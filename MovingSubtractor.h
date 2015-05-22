#pragma once

#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BackgroundSubtractorSuBSENSE.h"
#include "Timer.h"
using namespace std;

class MovingSubtractor {
public:
	MovingSubtractor(bool flag = false);
	void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
	void work(cv::InputArray image, cv::OutputArray fgmask, cv::Mat &delta, double learningRateOverride=0);
	void getBackgroundImage(cv::Mat oBackground) const;

private:
	// output detail information
	bool detailInformation;
	inline void outputInformation(const string &sInfo, double num = -1, cv::Mat* matrix = NULL) const;

	// subsense
	BackgroundSubtractorSuBSENSE suBSENSE;
	// last frame
	cv::Mat mLastFrame;
	// time counter
	Timer t;
};