#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <time.h>

#include "BackgroundSubtractorSuBSENSE.h"
#include "MovingSubtractor.h"
#include "Timer.h"

using namespace std;

const int max_count = 50000;		// maximum number of features to detect
const double qlevel = 0.05;			// quality level for feature detection
const double minDist = 2;			// minimum distance between two feature points

MovingSubtractor::MovingSubtractor(bool flag, string path): suBSENSE(), detailInformation(flag), mLastFrame(), t(), frameIdx(1), sSaveP(path) {
}

void MovingSubtractor::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
	outputInformation("initialize started\n");
	t.reset();
	suBSENSE.initialize(oInitImg, oROI);
	mLastFrame = oInitImg.clone();
	outputInformation("initialize finished with time : ", t.getTime());
}

void MovingSubtractor::work(cv::InputArray _newFrame, cv::OutputArray fgmask, double learningRateOverride) {
	// Id count
	frameIdx ++;
	char num[100];
	sprintf(num, "%d", frameIdx);
	sNum = string(num);

	outputInformation("operate started\n");
	t.reset();
	cv::Mat newFrame = _newFrame.getMat();
	vector<uchar> status; 	// status of tracked features
	vector<float> err;    	// error in tracking
	vector<cv::Point2f> features1,features2;
	
	// to grey
	cv::Mat grey0, grey1;
	cv::cvtColor(mLastFrame, grey0, CV_RGB2GRAY);
	cv::cvtColor(newFrame, grey1, CV_RGB2GRAY);
	// detect the features
	cv::goodFeaturesToTrack(grey0, 		// the image 
							features1,   		// the output detected features
							max_count,  		// the maximum number of features 
							qlevel,     		// quality level
							minDist);   		// min distance between two features
	outputInformation("features got:", t.getTime());
	t.reset();
	// track features
	cv::calcOpticalFlowPyrLK(grey0, grey1,	// 2 consecutive images
							features1, 			// input point position in first image
							features2, 			// output point postion in the second image
							status,    			// tracking success
							err);      			// tracking error
	outputInformation("features traced:", t.getTime());
	
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
	
	t.reset();
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
	outputInformation("with time: ", t.getTime());
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpPerspective(mLastFrame, mAfterTransform, result, newFrame.size());
	cv::Mat mBeforTransform, resultInvert;
	cv::invert(result, resultInvert, cv::DECOMP_LU);
	cv::warpPerspective(newFrame, mBeforTransform, resultInvert, newFrame.size());
	cv::Mat grey2;
	cv::cvtColor(mAfterTransform, grey2, CV_RGB2GRAY);
	cv::Mat delta = grey2 - grey1;
	savePath("compare", delta);

	// use the result
	outputInformation("() operate :");
	t.reset();
	suBSENSE(mBeforTransform, fgmask, learningRateOverride);
	cv::warpPerspective(fgmask, fgmask, result, newFrame.size());
		
	savePath("AResult", fgmask.getMat());
	outputInformation("", t.getTime());
	if (frameIdx > STARTMATCH) {
		t.reset();
		outputInformation("patch match :");
		vector<cv::Point2i> ans;
		cv::Mat ansMat;
		suBSENSE.patch_match(newFrame, mLastFrame, ans, ansMat, resultInvert);
		outputInformation("", t.getTime());
		/*
		cv::Mat Rpatch;
		Rpatch.create(newFrame.size(), CV_8UC1);
		Rpatch = cv::Scalar(0);
		if (detailInformation) {
			string nnn = sSaveP + sNum + ".txt";
			FILE *f0 = fopen(nnn.c_str(), "w");
			for (int i = 0; i+patch_w < newFrame.rows; i++) {
				for (int j = 0; j+patch_w < newFrame.cols; j++) {
					double tttt = ansNum[i*newFrame.cols + j]/52;
					fprintf(f0, "%.3lf\t", tttt);
					if (tttt > 1) Rpatch.data[i*newFrame.cols + j] = 255;
				}
				fprintf(f0, "\n");
			}
			fclose(f0);
		}*/
		savePath("match", ansMat);
		
		t.reset();
		cv::warpPerspective(mLastMask, mLastMask, result, newFrame.size());
		outputInformation("max flow :");
		suBSENSE.randomField(newFrame, ansMat, mLastMask, fgmask);
		savePath("BResult", fgmask.getMat());
		outputInformation("", t.getTime());
	}
	t.reset();
	outputInformation("model update :");
	suBSENSE.update(newFrame, result);
	outputInformation("", t.getTime());
	if (frameIdx > 5) {
		suBSENSE.complete(fgmask);
		savePath("CResult", fgmask.getMat());
	}
	mLastFrame = newFrame.clone();
	mLastMask = fgmask.getMat().clone();
}

void MovingSubtractor::getBackgroundImage(cv::Mat oBackground) const {
	suBSENSE.getBackgroundImage(oBackground);
}

inline void MovingSubtractor::outputInformation(const string &sInfo, double num, cv::Mat* matrix) const {
	if (!detailInformation) return ;
	cout << sInfo ;
	if (num >= 0) printf("%.3lf\n", num);
	if (matrix) cout << *matrix << endl;
}
inline void MovingSubtractor::savePath(const string &sInfo, cv::Mat & pic) const {
	if (!detailInformation) return ;
	cv::imwrite(sSaveP + sInfo + sNum + ".jpg", pic);
}

void MovingSubtractor::patchmatch(const cv::Mat image, std::vector<cv::Point2i> &ans) {
	t.reset();
	outputInformation("patch match :");
	cv::Mat I = (cv::Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
	cv::Mat ansNum;
	suBSENSE.patch_match(image, mLastFrame, ans, ansNum, I);
	outputInformation("", t.getTime());
}

void MovingSubtractor::recover(cv::OutputArray &a, const cv::Mat &b, std::vector<cv::Point2i> &ans, double coverRate) {
	cv::Mat image = a.getMat();
	int aew = image.cols - patch_w + 1, aeh = image.rows - patch_w + 1, goa = int (patch_w * patch_w * coverRate);
	int ww = (aew - 1) / patch_w + 1;
	vector<bool> Wa, Bb;
	for (int ay = 0; ay < aeh; ay+=patch_w ) {
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int sWa = goa, sBb = goa;
			int bx = ans[ay * image.cols + ax].x, by = ans[ay * image.cols + ax].y;
			for (int ii = 0; ii < patch_w; ii ++)
				for (int jj = 0; jj < patch_w; jj ++) {
					if (b.data[(by + ii) * image.cols + bx + jj] == 0) sBb --;
					if (image.data[(ay + ii) * image.cols + ax + jj] == 255) sWa --;
				}
			Wa.push_back(sWa<=0? true:false);
			Bb.push_back(sBb<=0? true:false);
		}
	}

	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int idx = ay/patch_w*ww + ax/patch_w;
			int flag = int(Wa[idx]) * 2;
			if (ay > 0) flag += Wa[idx -ww];
			if (ay < aeh-patch_w) flag += Wa[idx + ww ];
			if (ax > 0) flag += Wa[idx - 1];
			if (ax < aew-patch_w) flag += Wa[idx + 1];
			if ((flag < 4) && Bb[idx]) 
				for (int ii = 0; ii < patch_w; ii ++)
					for (int jj = 0; jj < patch_w; jj ++)
						image.data[(ay + ii) * image.cols + ax + jj] = 0;
		}
	image.convertTo(a, CV_8U);
}
