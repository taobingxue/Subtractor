#pragma once

#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BackgroundSubtractorSuBSENSE.h"
#include "Timer.h"
using namespace std;

// patch match from this frame
const int STARTMATCH = 9;
// whether is all black
const double COVERRATE = 0.95;

class MovingSubtractor {
public:
	MovingSubtractor(bool flag = false);
	void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
	void work(cv::InputArray image, cv::OutputArray fgmask, cv::Mat &delta, double learningRateOverride=0);
	void getBackgroundImage(cv::Mat oBackground) const;
	void patchmatch(const cv::Mat image, std::vector<cv::Point2i> &ans);
	void recover(cv::OutputArray &a, const cv::Mat &b, std::vector<cv::Point2i> &ans, double coverRate = 0.95);

private:
	// output detail information
	bool detailInformation;
	inline void outputInformation(const string &sInfo, double num = -1, cv::Mat* matrix = NULL) const;

	// subsense
	BackgroundSubtractorSuBSENSE suBSENSE;
	// last frame
	cv::Mat mLastFrame;
	// last fgmask
	cv::Mat mLastMask;
	// time counter
	Timer t;
	// count frame
	int frameIdx;
};