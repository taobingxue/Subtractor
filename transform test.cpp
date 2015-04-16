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

const int LIMITK = 500;				// limit for k
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
	/*
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
	*/

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
	for (int i=0; i < iSumK; i++) {
		pInput[0] = features1[i];
		pOuput[0] = features2[i];
		for (int j=i+1; j < iSumK; j++) {
			pInput[1] = features1[j];
			pOuput[1] = features2[j];
			int step = (iSumK - j) < iTimes? 1 : (iSumK - j) / iTimes;   // step should > 0s
			int p = j+1;
			for (int q=0; q<iTimes && p<iSumK; q++) {
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
				p += step;

				sss += 1;
				if (sss % 100 == 0) cout << sss << endl;
			}
		}
	}

	cout << endl << sss << endl;
	cout << (int) dTransforms[0][0].size() << endl;
	// sort values
	for (int i = 0; i < 2; i ++)
		for (int j = 0; j < 2; j ++)
			sort(dSortedTransforms[i][j].begin(), dSortedTransforms[i][j].end());
	// build result
	double resultData[2][3];
	int iMid = LIMITSUM >> 1;
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
}