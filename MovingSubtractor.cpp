#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

#include "BackgroundSubtractorSuBSENSE.h"
#include "MovingSubtractor.h"

using namespace std;

const int max_count = 50000;	  	// maximum number of features to detect
const double qlevel = 0.05;    		// quality level for feature detection
const double minDist = 2;   		// minimum distance between two feature points

MovingSubtractor::MovingSubtractor(bool flag): suBSENSE(), detailInformation(flag), mLastFrame() {
}

void MovingSubtractor::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
	if (detailInformation) cout << "initialize started" << endl;
	suBSENSE.initialize(oInitImg, oROI);
	mLastFrame = oInitImg.clone();
	if (detailInformation) cout << "initialize finished" << endl;
}

void MovingSubtractor::work(cv::InputArray _newFrame, cv::OutputArray fgmask, double learningRateOverride) {
	if (detailInformation) cout << "operate started" << endl;
	cv::Mat newFrame = _newFrame.getMat();
	vector<uchar> status; 	// status of tracked features
	vector<float> err;    	// error in tracking
	vector<cv::Point2f> features1,features2;
	// detect the features
	cv::goodFeaturesToTrack(mLastFrame, 		// the image 
							features1,   		// the output detected features
							max_count,  		// the maximum number of features 
							qlevel,     		// quality level
							minDist);   		// min distance between two features
	if (detailInformation) cout << "features got" << endl;
	// 2. track features
	cv::calcOpticalFlowPyrLK(mLastFrame, newFrame,	// 2 consecutive images
							features1, 			// input point position in first image
							features2, 			// output point postion in the second image
							status,    			// tracking success
							err);      			// tracking error
	if (detailInformation) cout << "features traced" << endl;
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
	if (detailInformation) {
		cout << "featrues selected" << endl;
		cout << "k = " << k <<endl;
	}

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
	if (detailInformation) cout << "matrix got k = " << (int) dTransforms[0][0].size() << endl;

	// sort values
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 3; j ++)
			sort(dSortedTransforms[i][j].begin(), dSortedTransforms[i][j].end());
	// find features
	vector<cv::Point2f> selectedFeaturesI, selectedFeaturesO;
	int iS = (int) isegLength / 3.8, iE = isegLength - iS;
	if (detailInformation) cout << "range: " << iS << " " << iE << endl;

	/*freopen("a.out", "w", stdout);

					for (int ia = 0; ia < 2; ia ++) {
						for (int ja = 0; ja < 3; ja ++) cout << dSortedTransforms[ia][ja][iS] << "," << dSortedTransforms[ia][ja][iE] << " ";
						cout << endl;
					}*/
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
	if (detailInformation) cout << "voted got k = " << selectedFeaturesI.size() << endl;
	// count result
	std::vector<uchar> inliers(selectedFeaturesI.size());
	cv::Mat result = cv::findHomography(	cv::Mat(selectedFeaturesI),		// corresponding
											cv::Mat(selectedFeaturesO),		// points
											inliers,				// outputted inliers matches
											CV_RANSAC,				// RANSAC method
											0.1);					// max distance to reprojection point
	if (detailInformation) cout << "tansform matrix :\n" << result <<endl;
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpPerspective(newFrame, mAfterTransform, result, newFrame.size());

	// use the result
	suBSENSE(mAfterTransform, fgmask, learningRateOverride);
	mLastFrame = newFrame;
}

void MovingSubtractor::getBackgroundImage(cv::Mat oBackground) {
	suBSENSE.getBackgroundImage(oBackground);
}