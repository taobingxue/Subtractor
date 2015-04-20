#include "MovingSubtractor.h"
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
    printf("\nforeground-background segmentation for moving (maybe Pan-Tilt-Zoom) camera.\n"
            "OpenCV's BackgroundSubtractor interface; will analyze frames from the file in the term of JPG pictures\n"
            "Usage: \n"
            "  ./bgfg_segm [--filepath]=<path to file>, [--filename]=<in common part of the name>, [--numlength]=<length for the number of each picture, -1 for no fixed length>, [--numstart]=<start number> \n\n");
}

const char* keys = {
	"{n  |filename |in       | file name		}"
    "{p  |filepath |         | file path		}"
	"{l  |numlength|6        | length of num	}"
	"{s  |numstart |0        | start of num     }"
};

// built the whole path for specific number
string getpath(string fileName, int inumLength, int inumNow) {
	string snumNow = to_string(inumNow);
	int ilength = snumNow.length();
	if (inumLength > ilength) return fileName.append(inumLength - ilength, '0') + snumNow + ".jpg";
	else return fileName+snumNow;
}
// show the Mat
void test(cv::Mat aa, string _filename = "show Picture") {
	cv::imshow(_filename, aa);
	cv::waitKey(0);
}
int main(int argc, char* argv[]) {
	help();
	// pass the ccommand line
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
	
	MovingSubtractor oSubtractor;
	oSubtractor.initialize(oCurrInputFrame, oROI);
	cout<< oCurrInputFrame.channels() <<endl;
	for (iNumNow++; ; iNumNow++) {
		// read new frame
		oCurrInputFrame = cv::imread(sFilePath + getpath(sFileName, iNumLength, iNumNow));
		if (oCurrInputFrame.empty() || (!oCurrInputFrame.data)) break;
		// motion compensate
		
		// subtractor work with new frame
		oSubtractor.work(oCurrInputFrame, oCurrSegmMask);
		oSubtractor.getBackgroundImage(oCurrReconstrBGImg);
		// save result
		cout<< iNumNow <<endl;
		cv::imwrite(sFilePath + getpath("ou", iNumLength, iNumNow), oCurrSegmMask);
		cv::imwrite(sFilePath + getpath("bg", iNumLength, iNumNow), oCurrReconstrBGImg);
	}
	return 0;
}