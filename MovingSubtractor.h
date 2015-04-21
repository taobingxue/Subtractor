#pragma once

#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BackgroundSubtractorSuBSENSE.h"
using namespace std;

class MovingSubtractor {
public:
	MovingSubtractor(bool flag = false);
	void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
	void work(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	void getBackgroundImage(cv::Mat oBackground) const;

private:
	inline void outputInformation(const string &sInfo, int num = -1, cv::Mat* matrix = NULL) const;
	BackgroundSubtractorSuBSENSE suBSENSE;
	bool detailInformation;
	cv::Mat mLastFrame;
};