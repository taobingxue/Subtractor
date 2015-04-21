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
            "  ./bgfg_segm --filepath/-f=<path to file> --savepath/-s=<path to save> [--info/-i=<whether output infos, true/false>]\n\n");
}

const char* keys = {
    "{p  |filepath |         | file path		}"
	"{s  |savepath |         | save path		}"
	"{i  |info     |true    | whether output   }"
};

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
	const string sSavePath = parser.get<string>("savepath");
	const bool bOutputInfo = parser.get<bool>("info");
	if (bOutputInfo) cout << "^.^" << endl;
	cout << bOutputInfo << endl;
    cv::Mat oCurrInputFrame, oCurrSegmMask, oCurrReconstrBGImg, oDeltaImg, oROI;
	parser.printParams();

	cv::VideoCapture inputFile(sFilePath);
	if (!inputFile.isOpened()) {
		cout << "Failed to open the image sequence!\n" << endl;
		return 0;
	}
	
	// initialization
	inputFile >> oCurrInputFrame;
	oCurrSegmMask.create(oCurrInputFrame.size(),CV_8UC1);
    oCurrReconstrBGImg.create(oCurrInputFrame.size(),oCurrInputFrame.type());
	oDeltaImg = oCurrReconstrBGImg.clone();
	oROI = cv::Mat(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255));
	
	MovingSubtractor oSubtractor(bOutputInfo);
	oSubtractor.initialize(oCurrInputFrame, oROI);
	char num[100];
	for (int i = 0; ; i ++ ) {
		// read new frame
		inputFile >> oCurrInputFrame;
		if (oCurrInputFrame.empty() || (!oCurrInputFrame.data)) break;
		// subtractor work with new frame
		oSubtractor.work(oCurrInputFrame, oCurrSegmMask, oDeltaImg);
		oSubtractor.getBackgroundImage(oCurrReconstrBGImg);
		// save result
		printf("%d\n", i);
		sprintf(num, "%d", i);
		string ss = string(num) + ".jpg";
		cv::imwrite(sSavePath + "ou" + ss, oCurrSegmMask);
		cv::imwrite(sSavePath + "bg" + ss, oCurrReconstrBGImg);
		cv::imwrite(sSavePath + "compare" + ss, oDeltaImg);
	}
	return 0;
}