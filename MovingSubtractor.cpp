#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

#include "BackgroundSubtractorSuBSENSE.h"
#include "MovingSubtractor.h"

using namespace std;

const int max_count = 50000;		// maximum number of features to detect
const double qlevel = 0.05;			// quality level for feature detection
const double minDist = 2;			// minimum distance between two feature points

MovingSubtractor::MovingSubtractor(bool flag): suBSENSE(), detailInformation(flag), mLastFrame() {
}

void MovingSubtractor::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
	outputInformation("initialize started\n");
	suBSENSE.initialize(oInitImg, oROI);
	mLastFrame = oInitImg.clone();
	outputInformation("initialize finished\n");
}

void MovingSubtractor::work(cv::InputArray _newFrame, cv::OutputArray fgmask, cv::Mat &delta, double learningRateOverride) {
	outputInformation("operate started\n");
	cv::Mat newFrame = _newFrame.getMat();
	vector<uchar> status; 	// status of tracked features
	vector<float> err;    	// error in tracking
	vector<cv::Point2f> features1,features2;

	cv::Mat grey0, grey1;
	cv::cvtColor(mLastFrame, grey0, CV_RGB2GRAY);
	cv::cvtColor(newFrame, grey1, CV_RGB2GRAY);
	// detect the features
	cv::goodFeaturesToTrack(grey0, 		// the image 
							features1,   		// the output detected features
							max_count,  		// the maximum number of features 
							qlevel,     		// quality level
							minDist);   		// min distance between two features
	outputInformation("features got\n");
	// 2. track features
	cv::calcOpticalFlowPyrLK(grey0, grey1,	// 2 consecutive images
							features1, 			// input point position in first image
							features2, 			// output point postion in the second image
							status,    			// tracking success
							err);      			// tracking error
	outputInformation("features traced\n");
	// remove tracking failed features
	int k=0;
	for( int i= 0; i < (int) features1.size(); i++ ) 
	{
		// do we keep this point?
		if (status[i] == 1) 
		{
			// keep this point in vector
			features1[k] = features1[i];
			features2[k++] = features2[i];
		}
	}
	features1.resize(k);
	features2.resize(k);
	outputInformation("featrues selected: k = ", k);

	// calculate result with vote based on cv::getAffineTransform()
	cv::Point2f pInput[3], pOuput[3];
	vector<double> dTransforms[2][3], dSortedTransforms[2][3];
	int isegLength = k / 3;
	double* pdata;
	for (int i = 0; i < isegLength; i ++) {
		pInput[0] = features1[i];
		pOuput[0] = features2[i];
		pInput[1] = features1[i + isegLength];
		pOuput[1] = features2[i + isegLength];
		pInput[2] = features1[k - i - 1];
		pOuput[2] = features2[k - i - 1];
		// get transform
		cv::Mat mTransform = cv::getAffineTransform(pInput, pOuput);
		for (int ii = 0; ii < 2; ii ++) {
			pdata = mTransform.ptr<double>(ii);
			for (int jj = 0; jj<3; jj ++) {
				dTransforms[ii][jj].push_back(pdata[jj]);
				dSortedTransforms[ii][jj].push_back(pdata[jj]);
			}
		}
	}
	outputInformation("matrix got k = ", (int) dTransforms[0][0].size());

	// sort values
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 3; j ++)
			sort(dSortedTransforms[i][j].begin(), dSortedTransforms[i][j].end());
	// find features
	vector<cv::Point2f> selectedFeaturesI, selectedFeaturesO;
	// to limit the range
	double limit = 3.8;
	for (; !selectedFeaturesI.size(); limit *= 1.1) {
		int iS = (int) isegLength / limit, iE = isegLength - iS;
		outputInformation("range iS = ", iS);
		outputInformation("range iE = ", iE);
		for (int i = 0; i < isegLength; i ++) {
			bool flag = true;
			for (int ii = 0; ii < 2; ii ++)
				if (flag) for (int jj = 0; jj < 3; jj ++)
					if (!(dTransforms[ii][jj][i] > dSortedTransforms[ii][jj][iS] && dTransforms[ii][jj][i] < dSortedTransforms[ii][jj][iE])) {
						flag = false;
						/*cout << i << " " << ii << " " << jj << endl;
						for (int ia = 0; ia < 2; ia ++) {
							for (int ja = 0; ja < 3; ja ++) cout << dTransforms[ia][ja][i] << " ";
							cout << endl;
						}*/
						break;
					}
			if (flag) {
				selectedFeaturesI.push_back(features1[i]);
				selectedFeaturesO.push_back(features2[i]);
				selectedFeaturesI.push_back(features1[i + isegLength]);
				selectedFeaturesO.push_back(features2[i + isegLength]);
				selectedFeaturesI.push_back(features1[k - i - 1]);
				selectedFeaturesO.push_back(features2[k - i - 1]);
			}
		}
		outputInformation("voted got k = ", selectedFeaturesI.size());
	}
	// count result
	std::vector<uchar> inliers(selectedFeaturesI.size());
	cv::Mat result = cv::findHomography(	cv::Mat(selectedFeaturesI),		// corresponding
											cv::Mat(selectedFeaturesO),		// points
											inliers,				// outputted inliers matches
											CV_RANSAC,				// RANSAC method
											0.1);					// max distance to reprojection point
	outputInformation("tansform matrix :\n", -1, &result);
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpPerspective(mLastFrame, mAfterTransform, result, newFrame.size());
	cv::Mat mBeforTransform;
	cv::warpPerspective(newFrame, mBeforTransform, result, newFrame.size(), cv::WARP_INVERSE_MAP & cv::INTER_LINEAR);
	cv::Mat grey2;
	cv::cvtColor(mAfterTransform, grey2, CV_RGB2GRAY);
	delta = grey2 - grey1;

	// use the result
	suBSENSE(mBeforTransform, fgmask, learningRateOverride);
	mLastFrame = newFrame.clone();
}

void MovingSubtractor::getBackgroundImage(cv::Mat oBackground) const {
	suBSENSE.getBackgroundImage(oBackground);
}

inline void MovingSubtractor::outputInformation(const string &sInfo, int num, cv::Mat* matrix) const {
	if (!detailInformation) return ;
	cout << sInfo << endl;
	if (num >= 0) printf("%d\n", num);
	if (matrix) cout << *matrix << endl;
}