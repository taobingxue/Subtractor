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

static void help() {
    printf("\nMinimalistic example of foreground-background segmentation in a video sequence using\n"
            "OpenCV's BackgroundSubtractor interface; will analyze frames from the default camera\n"
            "or from a specified file.\n\n"
            "Usage: \n"
            "  ./bgfg_segm [--camera]=<use camera, true/false>, [--file]=<path to file> \n\n");
}
const char* keys = {
	"{n  |filename |in       | file name		}"
    "{p  |filepath |         | file path		}"
	"{l  |numlength|6        | length of num	}"
	"{s  |numstart |0        | start of num     }"
};
string getpath(string fileName, int inumLength, int inumNow) {
	string snumNow = to_string(inumNow);
	int ilength = snumNow.length();
	if (inumLength > ilength) return fileName.append(inumLength - ilength, '0') + snumNow + ".jpg";
	else return fileName+snumNow;
}
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
	help();
    cv::CommandLineParser parser(argc, argv, keys);
	const string sFilePath = parser.get<string>("filepath");
	const string sFileName = parser.get<string>("filename");
	const int iNumLength = parser.get<int>("numlength");
	const int iNumStart = parser.get<int>("numstart");
	int iNumNow = iNumStart;
    cv::Mat oCurrInputFrame, oCurrSegmMask, oCurrReconstrBGImg, oROI;
	parser.printParams();
	// initialization
	oCurrInputFrame = cv::imread(sFilePath + getpath(sFileName, iNumLength, iNumNow));
	oCurrSegmMask.create(oCurrInputFrame.size(),CV_8UC1);
    oCurrReconstrBGImg.create(oCurrInputFrame.size(),oCurrInputFrame.type());
	oROI = cv::Mat(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255));
	
	BackgroundSubtractorSuBSENSE oSubtractor;
	oSubtractor.initialize(oCurrInputFrame, oROI);
	cout<< oCurrInputFrame.channels() <<endl;
	for (iNumNow++; ; iNumNow++) {
		// read new frame
		oCurrInputFrame = cv::imread(sFilePath + getpath(sFileName, iNumLength, iNumNow));
		if (oCurrInputFrame.empty() || (!oCurrInputFrame.data)) break;
		// motion compensate
		
		// subtractor work with new frame
		oSubtractor(oCurrInputFrame, oCurrSegmMask);
		oSubtractor.getBackgroundImage(oCurrReconstrBGImg);
		// save result
		cout<< iNumNow <<endl;
		cv::imwrite(sFilePath + getpath("ou", iNumLength, iNumNow), oCurrSegmMask);
		cv::imwrite(sFilePath + getpath("bg", iNumLength, iNumNow), oCurrReconstrBGImg);
		/*
		imshow("input",oCurrInputFrame);
        imshow("segmentation mask",oCurrSegmMask);
        imshow("reconstructed background",oCurrReconstrBGImg);
        if (cv::waitKey(1)==27) break;
		*/
	}
	return 0;
}