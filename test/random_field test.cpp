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
            "  ./bgfg_segm --filepath/-f=<path to file> --savepath/-s=<path to save> [--info/-i=<whether output infos, true/false>]\n\n"
			"eg. -p=./data/in%%06d.jpg -s=./output/ -i=true");
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
	cv::Mat aaaaaa = cv::imread("in000001.jpg");
	// initialization
	BackgroundSubtractorSuBSENSE oSubtractor;
	//cv::blur( oCurrInputFrame, oCurrInputFrame, cv::Size( 4, 4 ), cv::Point(-1,-1));
	oSubtractor.initialize(aaaaaa,cv::Mat(aaaaaa.size(),CV_8UC1,cv::Scalar_<uchar>(255)));

		vector<cv::Point2i> ans;
		Timer t;
		char name[100];
	for (int ii=3; ii<=28; ii++) {
		//printf("random field:");
		sprintf(name, "%06d", ii+2);
		cout << "in"+string(name)+".jpg" << endl;
		cv::Mat bbbbbb = cv::imread("in"+string(name)+".jpg");
		sprintf(name, "%d", ii);
		cout << "ou"+string(name)+".jpg"<<endl;
		cv::Mat cccccc = cv::imread("ou"+string(name)+".jpg", CV_8UC1);
		cv::imwrite("A.jpg", bbbbbb);
		cv::imwrite("B.jpg", cccccc);
		//printf("%d %d\n", cccccc.cols, cccccc.rows);
		// oSubtractor(bbbbbb, cccccc);
		oSubtractor.randomField(bbbbbb, cccccc);
		cout << "time:" << t.getTime() << endl;
		cv::imwrite("ou"+string(name)+"(2).jpg", cccccc);
	}
	return 0;
}