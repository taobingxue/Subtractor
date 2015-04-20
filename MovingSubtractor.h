#pragma once

#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BackgroundSubtractorSuBSENSE.h"

class MovingSubtractor {
public:
	MovingSubtractor(bool flag = false);
	void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
	void work(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	void getBackgroundImage(cv::Mat oBackground);

private:
	BackgroundSubtractorSuBSENSE suBSENSE;
	bool detailInformation;
	cv::Mat mLastFrame;
};