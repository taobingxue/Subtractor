#include "BackgroundSubtractorSuBSENSE.h"
#include "highgui.h"
#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <windows.h>
#include "time.h"

using namespace std;

void test(cv::Mat aa) {
	IplImage* src = new IplImage(aa);
	cvNamedWindow("show_image", 1);
	cvShowImage("show_image", src);
	cout<<0<<endl;
	cvWaitKey(0);
	cout<<1<<endl;
	cvReleaseImage(&src);
	cout<<2<<endl;
	cvDestroyWindow("show_image");
}

const int LIMITK = 300;				// limit for k
const int LIMITSUM = 1000000;		// limit for times we calculate affine transform matrix

int main(int argc, char* argv[]) {
	cv::Mat aaaaaa = cv::imread("in000001.jpg", CV_8UC1);
	cv::Mat bbbbbb = cv::imread("in000002.jpg", CV_8UC1);
	int max_count = 50000;	  	// maximum number of features to detect
	double qlevel = 0.05;    	// quality level for feature detection
	double minDist = 2;   		// minimum distance between two feature points
	std::vector<uchar> status; 	// status of tracked features
	std::vector<float> err;    	// error in tracking

	std::vector<cv::Point2f> features1,features2;
	// detect the features
	cv::goodFeaturesToTrack(aaaaaa, 			// the image 
							features1,   		// the output detected features
							max_count,  		// the maximum number of features 
							qlevel,     		// quality level
							minDist);   		// min distance between two features
	// 2. track features
	cv::calcOpticalFlowPyrLK(aaaaaa, bbbbbb,	// 2 consecutive images
							features1, 			// input point position in first image
							features2, 			// output point postion in the second image
							status,    			// tracking success
							err);      			// tracking error
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
	cout << "k = " << k <<endl;

	SYSTEMTIME t;
	GetLocalTime(&t);
	int startTime = t.wMilliseconds;
	// calculate result directly with cv::findHomography
	std::vector<uchar> inliers(features1.size());
	cv::Mat result = cv::findHomography(	cv::Mat(features1),		// corresponding
											cv::Mat(features2),		// points
											inliers,				// outputted inliers matches
											CV_RANSAC,				// RANSAC method
											0.1);					// max distance to reprojection point
	cout << result <<endl;
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpPerspective(aaaaaa, mAfterTransform, result, aaaaaa.size());
	cv::imwrite("ou1.jpg", mAfterTransform); 
	GetLocalTime(&t);
	int nowTime = t.wMilliseconds;
	cout << "time1:" << nowTime - startTime << endl;
	startTime = nowTime;
	/*
	// calculate result directly with cv::estimateRigidTransform
	cv::Mat result = cv::estimateRigidTransform(cv::Mat(features1),		// corresponding
												cv::Mat(features2),		// points
												true);					// 6 degrees of freedom
	cout << result <<endl;
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpAffine(aaaaaa, mAfterTransform, result, aaaaaa.size());
	cv::imwrite("ou2.jpg", mAfterTransform); 
	*/

	/*
	// calculate result with vote based on cv::getAffineTransform()
	// limit the scale of k
	cv::Point2f pInput[3], pOuput[3];
	// vector of matrix
	// std::vector<cv::Mat> mTransforms;
	std::vector<double> dTransforms[2][3], dSortedTransforms[2][3];
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 2; j ++)
			dTransforms[i][j].reserve(LIMITSUM);

	int iSumK = k;
	if (k > LIMITK) iSumK = LIMITK;
	int iTimes = LIMITSUM / iSumK / (iSumK >> 1);
	if (!iTimes) iTimes = 1;
	cout << iSumK << " " << iTimes << endl;
	double* pdata;
	
	// iSumK = 5;
	int sss = 0;
	int segLength = LIMITK / 3;
	//for (int i=0; i < iSumK; i++) {
	for (int i = 0; i < segLength; i ++) {
		pInput[0] = features1[i];
		pOuput[0] = features2[i];
		for (int j=segLength; j < segLength << 1; j++) {
			pInput[1] = features1[j];
			pOuput[1] = features2[j];
			// int step = (iSumK - j) < iTimes? 1 : (iSumK - j) / iTimes;   // step should > 0s
			// int p = j+1;
			// for (int q=0; q<iTimes && p<iSumK; q++) {
			for (int p = segLength << 1; p < LIMITK; p ++) {
				pInput[2] = features1[p];
				pOuput[2] = features2[p];
				// get transform for i,j,q
				cv::Mat mTransform = cv::getAffineTransform(pInput, pOuput);
				// add to vector
				// cout << mTransform << endl;
				// mTransforms.push_back(mTransform);
				for (int ii = 0; ii < 2; ii ++) {
					pdata = mTransform.ptr<double>(ii);
					for (int jj = 0; jj<3; jj ++) {
						dTransforms[ii][jj].push_back(pdata[jj]);
						dSortedTransforms[ii][jj].push_back(pdata[jj]);
					}
				}
				// p += step;

				sss += 1;
				if (sss % 100 == 0)	cout << sss << endl;
			}
		}
	}
	cout << endl << sss << endl;
	cout << (int) dTransforms[0][0].size() << endl;
	// sort values
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 3; j ++)
			sort(dSortedTransforms[i][j].begin(), dSortedTransforms[i][j].end());
	// build result
	double resultData[2][3];
	int iMid = ((int) dTransforms[0][0].size()) >> 1;
	for (int i = 0; i < 2; i ++) {
		//pdata = result.ptr<double>(i);
		for (int j = 0; j<3; j ++) {
			//pdata[j] = dSortedTransforms[i][j][iMid];
			resultData[i][j] = dSortedTransforms[i][j][iMid];
			cout << dSortedTransforms[i][j][iMid] << " ";
		}
		cout << endl;
	}
	cv::Mat result(2, 3, CV_64FC1, resultData);
	cout << result << endl;
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpAffine(aaaaaa, mAfterTransform, result, aaaaaa.size());
	cv::imwrite("ou2.jpg", mAfterTransform);
	*/
	
	cv::Point2f pInput[3], pOuput[3];
	vector<double> dTransforms[2][3], dSortedTransforms[2][3];
	int sss = 0;
	int isegLength = k / 3;
	double* pdata;

	for (int i = 0; i < isegLength; i ++) {
		pInput[0] = features1[i];
		pOuput[0] = features2[i];
		pInput[1] = features1[i + isegLength];
		pOuput[1] = features2[i + isegLength];
		pInput[2] = features1[k - i - 1];
		pOuput[2] = features2[k - i - 1];

		cv::Mat mTransform = cv::getAffineTransform(pInput, pOuput);
		for (int ii = 0; ii < 2; ii ++) {
			pdata = mTransform.ptr<double>(ii);
			for (int jj = 0; jj<3; jj ++) {
				dTransforms[ii][jj].push_back(pdata[jj]);
				dSortedTransforms[ii][jj].push_back(pdata[jj]);
			}
		}
	}
	cout << (int) dTransforms[0][0].size() << endl;
	// sort values
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 3; j ++)
			sort(dSortedTransforms[i][j].begin(), dSortedTransforms[i][j].end());
	// find features
	vector<cv::Point2f> selectedFeaturesI, selectedFeaturesO;
	int iS = isegLength / 3.8, iE = isegLength - iS;
	cout << iS << " " << iE << endl;

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
	cout << selectedFeaturesI.size() << endl;
	
	/*
	// select the dense part
	vector<cv::Point2f> selectedFeaturesI, selectedFeaturesO;
	double selectRange = 0.5;
	for (; (int) selectedFeaturesI.size() < 4; selectRange *= 1.1) {
		selectedFeaturesI.resize(0);
		selectedFeaturesO.resize(0);
		double iS[2][3], iE[2][3];
		// determine the range to vote
		// size of segment
		int segSize = (int) isegLength * selectRange;
		for (int ia = 0; ia < 2; ia ++)
			for (int ja = 0; ja < 3; ja ++) {
				double range = 100;
				int sum = 0, maxSize = 0;
				vector<double> *data = &dSortedTransforms[ia][ja];
				for (; maxSize < segSize; range *= 0.85) {
					sum = 0; maxSize = 0;
					// step
					double step = ((*data)[isegLength-1] - (*data)[0]) / range, now = (*data)[0] + step;
					for (int i = 0; i < isegLength; i ++)
						if ((*data)[i] > now) {
							if (sum > maxSize) {
								maxSize = sum;
								iS[ia][ja] = now - step;
								iE[ia][ja] = now;
							}
							sum = 1;
							while ((*data)[i] > now) now += step;
						} else sum += 1;
					// last segment
					if (sum > maxSize) {
						maxSize = sum;
						iS[ia][ja] = now - step;
						iE[ia][ja] = now;
					}
				}
				outputInformation("voted iS[" + to_string(ia) + "][" + to_string(ja) + "]", iS[ia][ja]);
				outputInformation("voted iE[" + to_string(ia) + "][" + to_string(ja) + "]", iE[ia][ja]);
				outputInformation("voted segSize[" + to_string(ia) + "][" + to_string(ja) + "]", maxSize);
			}
		// get the vote result
		for (int i = 0; i < isegLength; i ++) {
			bool flag = true;
			for (int ia = 0; ia < 2; ia ++)
				if (flag) for (int ja = 0; ja < 3; ja ++)
					if (!(dTransforms[ia][ja][i] > iS[ia][ja] && dTransforms[ia][ja][i] < iE[ia][ja])) {
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
	*/	
	
	// count result
	std::vector<uchar> inlierss(selectedFeaturesI.size());
	result = cv::findHomography(	cv::Mat(selectedFeaturesI),		// corresponding
											cv::Mat(selectedFeaturesO),		// points
											inlierss,				// outputted inliers matches
											CV_RANSAC,				// RANSAC method
											0.1);					// max distance to reprojection point
	cout << result <<endl;
	// use the transform matrix
	//cv::Mat mAfterTransform;
	cv::warpPerspective(aaaaaa, mAfterTransform, result, aaaaaa.size());
	cv::imwrite("ou2.jpg", mAfterTransform); 
	
	GetLocalTime(&t);
	nowTime = t.wMilliseconds;
	cout << "time1:" << nowTime - startTime << endl;

	cv::Mat cccccc = aaaaaa.clone();
	cv::Scalar color( 255, 0, 0);
	for (int i = 0; i < (int)selectedFeaturesI.size(); i ++)
		cv::circle(cccccc, selectedFeaturesI[i], 3, color);
	cv::imwrite("Pointed.jpg", cccccc);  
}