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
	cv::imwrite("ou1.jpg", mAfterTransform); */

	/*
	// calculate result directly with cv::estimateRigidTransform
	cv::Mat result = cv::estimateRigidTransform(cv::Mat(features1),		// corresponding
												cv::Mat(features2),		// points
												true);					// 6 degrees of freedom
	cout << result <<endl;
	// use the transform matrix
	cv::Mat mAfterTransform;
	cv::warpAffine(aaaaaa, mAfterTransform, result, aaaaaa.size());
	cv::imwrite("ou2.jpg", mAfterTransform);   */
}