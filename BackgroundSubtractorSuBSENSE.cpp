#include "BackgroundSubtractorSuBSENSE.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include "graph.h"
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

/*
 *
 * Intrinsic parameters for our method are defined here; tuning these for better
 * performance should not be required in most cases -- although improvements in
 * very specific scenarios are always possible.
 *
 */
//! defines the threshold value(s) used to detect long-term ghosting and trigger the fast edge-based absorption heuristic
#define GHOSTDET_D_MAX (0.010f) // defines 'negligible' change here
#define GHOSTDET_S_MIN (0.995f) // defines the required minimum local foreground saturation value
//! parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)
//! parameters used to adjust the variation step size of 'v(x)'
#define FEEDBACK_V_INCR  (1.000f)
#define FEEDBACK_V_DECR  (0.100f)
//! parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
//! parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN (0.100f)
#define UNSTABLE_REG_RDIST_MIN (3.000f)
//! parameters used to scale the relative LBSP intensity threshold used for internal comparisons
#define LBSPDESC_NONZERO_RATIO_MIN (0.100f)
#define LBSPDESC_NONZERO_RATIO_MAX (0.500f)
//! parameters used to define model reset/learning rate boosts in our frame-level component
#define FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD  (m_nMinColorDistThreshold/2)
#define FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO (8)

// local define used to display debug information
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the default frame size (320x240 = QVGA)
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET (m_nMinColorDistThreshold/5)
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET (m_nDescDistThresholdOffset)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorSuBSENSE::BackgroundSubtractorSuBSENSE(	 float fRelLBSPThreshold
															,size_t nDescDistThresholdOffset
															,size_t nMinColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples
															,size_t nSamplesForMovingAvgs)
	:	 BackgroundSubtractorLBSP(fRelLBSPThreshold)
		,m_nMinColorDistThreshold(nMinColorDistThreshold)
		,m_nDescDistThresholdOffset(nDescDistThresholdOffset)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nSamplesForMovingAvgs(nSamplesForMovingAvgs)
		,m_fLastNonZeroDescRatio(0.0f)
		,m_bLearningRateScalingEnabled(true)
		,m_fCurrLearningRateLowerCap(FEEDBACK_T_LOWER)
		,m_fCurrLearningRateUpperCap(FEEDBACK_T_UPPER)
		,m_nMedianBlurKernelSize(m_nDefaultMedianBlurKernelSize)
		,m_bUse3x3Spread(true) {
	CV_Assert(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples);
	CV_Assert(m_nMinColorDistThreshold>=STAB_COLOR_DIST_OFFSET);
}

BackgroundSubtractorSuBSENSE::~BackgroundSubtractorSuBSENSE() {
	if(m_aPxIdxLUT)
		delete[] m_aPxIdxLUT;
	if(m_aPxInfoLUT)
	    delete[] m_aPxInfoLUT;
}

void BackgroundSubtractorSuBSENSE::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
	// == init
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.isContinuous());
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		if(!cv::countNonZero((voInitImgChannels[0]!=voInitImgChannels[1])|(voInitImgChannels[2]!=voInitImgChannels[1])))
			std::cout << std::endl << "\tBackgroundSubtractorSuBSENSE : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
	}
	cv::Mat oNewBGROI;
	if(oROI.empty() && (m_oROI.empty() || oROI.size()!=oInitImg.size())) {
		oNewBGROI.create(oInitImg.size(),CV_8UC1);
		oNewBGROI = cv::Scalar_<uchar>(UCHAR_MAX);
	}
	else if(oROI.empty())
		oNewBGROI = m_oROI;
	else {
		CV_Assert(oROI.size()==oInitImg.size() && oROI.type()==CV_8UC1);
		CV_Assert(cv::countNonZero((oROI<UCHAR_MAX)&(oROI>0))==0);
		oNewBGROI = oROI.clone();
		cv::Mat oTempROI;
		cv::dilate(oNewBGROI,oTempROI,cv::Mat(),cv::Point(-1,-1),LBSP::PATCH_SIZE/2);
		cv::bitwise_or(oNewBGROI,oTempROI/2,oNewBGROI);
	}
	const size_t nOrigROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
	CV_Assert(nOrigROIPxCount>0);
	LBSP::validateROI(oNewBGROI);
	const size_t nFinalROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
	CV_Assert(nFinalROIPxCount>0);
	m_oROI = oNewBGROI;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nTotPxCount = m_oImgSize.area();
	m_nTotRelevantPxCount = nFinalROIPxCount;
	m_nFrameIndex = 0;
	m_nFramesSinceLastReset = 0;
	m_nModelResetCooldown = 0;
	m_fLastNonZeroDescRatio = 0.0f;
	const int nTotImgPixels = m_oImgSize.height*m_oImgSize.width;
	if(nOrigROIPxCount>=m_nTotPxCount/2 && (int)m_nTotPxCount>=DEFAULT_FRAME_SIZE.area()) {
		m_bLearningRateScalingEnabled = true;
		m_bAutoModelResetEnabled = true;
		m_bUse3x3Spread = !(nTotImgPixels>DEFAULT_FRAME_SIZE.area()*2);
		const int nRawMedianBlurKernelSize = std::min((int)floor((float)nTotImgPixels/DEFAULT_FRAME_SIZE.area()+0.5f)+m_nDefaultMedianBlurKernelSize,14);
		m_nMedianBlurKernelSize = (nRawMedianBlurKernelSize%2)?nRawMedianBlurKernelSize:nRawMedianBlurKernelSize-1;
		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
	}
	else {
		m_bLearningRateScalingEnabled = false;
		m_bAutoModelResetEnabled = false;
		m_bUse3x3Spread = true;
		m_nMedianBlurKernelSize = m_nDefaultMedianBlurKernelSize;
		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER*2;
		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER*2;
	}
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fCurrLearningRateLowerCap);
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oVariationModulatorFrame.create(m_oImgSize,CV_32FC1);
	m_oVariationModulatorFrame = cv::Scalar(10.0f); // should always be >= FEEDBACK_V_DECR
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);
	m_oDownSampledFrameSize = cv::Size(m_oImgSize.width/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO,m_oImgSize.height/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO);
	m_oMeanDownSampledLastDistFrame_LT.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanDownSampledLastDistFrame_ST.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);
	m_oUnstableRegionMask.create(m_oImgSize,CV_8UC1);
	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oDownSampledFrame_MotionAnalysis.create(m_oDownSampledFrameSize,CV_8UC((int)m_nImgChannels));
	m_oDownSampledFrame_MotionAnalysis = cv::Scalar_<uchar>::all(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oLastRawFGMask.create(m_oImgSize,CV_8UC1);
	m_oLastRawFGMask = cv::Scalar_<uchar>(0);
	m_oLastFGMask.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask = cv::Scalar_<uchar>(0);
	m_oLastFGMask_dilated.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask_dilated = cv::Scalar_<uchar>(0);
	m_oLastFGMask_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask_dilated_inverted = cv::Scalar_<uchar>(0);
	m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
	m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
	m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
	m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
	m_oCurrRawFGBlinkMask.create(m_oImgSize,CV_8UC1);
	m_oCurrRawFGBlinkMask = cv::Scalar_<uchar>(0);
	m_oLastRawFGBlinkMask.create(m_oImgSize,CV_8UC1);
	m_oLastRawFGBlinkMask = cv::Scalar_<uchar>(0);
	m_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
		m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
	}
	if(m_aPxIdxLUT)
		delete[] m_aPxIdxLUT;
	if(m_aPxInfoLUT)
	    delete[] m_aPxInfoLUT;
	m_aPxIdxLUT = new size_t[m_nTotRelevantPxCount];
	m_aPxInfoLUT = new PxInfoBase[m_nTotPxCount];
	if(m_nImgChannels==1) {
		CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
		CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold)/3);
		for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
			if(m_oROI.data[nPxIter]) {
				m_aPxIdxLUT[nModelIter] = nPxIter;
				m_aPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nModelIdx = nModelIter;
				m_oLastColorFrame.data[nPxIter] = oInitImg.data[nPxIter];
				const size_t nDescIter = nPxIter*2;
				LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[nPxIter],m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
				++nModelIter;
			}
		}
	}
	else { //m_nImgChannels==3
		CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*3 && m_oLastColorFrame.step.p[1]==3);
		CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold);
		for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
			if(m_oROI.data[nPxIter]) {
				m_aPxIdxLUT[nModelIter] = nPxIter;
				m_aPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
				m_aPxInfoLUT[nPxIter].nModelIdx = nModelIter;
				const size_t nPxRGBIter = nPxIter*3;
				const size_t nDescRGBIter = nPxRGBIter*2;
				for(size_t c=0; c<3; ++c) {
					m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
					LBSP::computeSingleRGBDescriptor(oInitImg,oInitImg.data[nPxRGBIter+c],m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
				}
				++nModelIter;
			}
		}
	}
	m_bInitialized = true;
	refreshModel(1.0f);
}

void BackgroundSubtractorSuBSENSE::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
	// == refresh
	CV_Assert(m_bInitialized);
	CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
	const size_t nModelsToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	if(m_nImgChannels==1) {
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
				for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X,nSampleImgCoord_Y,m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter] = m_oLastColorFrame.data[nSamplePxIdx];
						*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+nPxIter*2)) = *((ushort*)(m_oLastDescFrame.data+nSamplePxIdx*2));
					}
				}
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
				for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X,nSampleImgCoord_Y,m_aPxInfoLUT[nPxIter].nImgCoord_X,m_aPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						for(size_t c=0; c<3; ++c) {
							m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter*3+c] = m_oLastColorFrame.data[nSamplePxIdx*3+c];
							*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+(nPxIter*3+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*3+c)*2));
						}
					}
				}
			}
		}
	}
}

void BackgroundSubtractorSuBSENSE::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	// == process
	CV_Assert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_Assert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	CV_Assert(oInputImg.isContinuous());
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	size_t nNonZeroDescCount = 0;
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs/4);
	if(m_nImgChannels==1) {
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			const uchar nCurrColor = oInputImg.data[nPxIter];
			
			// avoid empty pixel after transform
			if (nCurrColor == 0) continue;
			
			const size_t nDescIter = nPxIter*2;
			const size_t nFloatIter = nPxIter*4;
			const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
			const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
			size_t nMinDescDist = s_nDescMaxDataRange_1ch;
			size_t nMinSumDist = s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
			float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
			float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
			float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+nDescIter));
			uchar& nLastColor = m_oLastColorFrame.data[nPxIter];
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET))/2;
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[nPxIter];
				{
					const size_t nColorDist = L1dist(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIter));
					const size_t nIntraDescDist = hdist(nCurrIntraDesc,nBGIntraDesc);
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					const size_t nInterDescDist = hdist(nCurrInterDesc,nBGIntraDesc);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
					const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					if(nMinDescDist>nDescDist)
						nMinDescDist = nDescDist;
					if(nMinSumDist>nSumDist)
						nMinSumDist = nSumDist;
					nGoodSamplesCount++;
				}
				failedcheck1ch:
				nSampleIdx++;
			}
			const float fNormalizedLastDist = ((float)L1dist(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)hdist(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[nPxIter] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
				}
			}
			else {
				// == background
				const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
				}
				int nSampleImgCoord_Y, nSampleImgCoord_X;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
					const size_t idx_rand_ushrt = idx_rand_uchar*2;
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
				}
			}
			if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
				if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
					*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
				*pfCurrLearningRate = m_fCurrLearningRateUpperCap;
			if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
				(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
			else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
				(*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
				if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
					(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
				(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
			else {
				(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount(nCurrIntraDesc)>=2)
				++nNonZeroDescCount;
			nLastIntraDesc = nCurrIntraDesc;
			nLastColor = nCurrColor;
		}
	}
	else { //m_nImgChannels==3
		for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
			const size_t nPxIter = m_aPxIdxLUT[nModelIter];
			const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
			const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
			const size_t nPxIterRGB = nPxIter*3;
			const uchar* const anCurrColor = oInputImg.data+nPxIterRGB;
			
			// avoid empty pixel after transform
			if (anCurrColor[0] == 0 && anCurrColor[1] == 0 && anCurrColor[2] == 0) continue;

			const size_t nDescIterRGB = nPxIterRGB*2;
			const size_t nFloatIter = nPxIter*4;			
			size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
			float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
			float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
			float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+nDescIterRGB));
			uchar* anLastColor = m_oLastColorFrame.data+nPxIterRGB;
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET));
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIterRGB);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+nPxIterRGB;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					const size_t nIntraDescDist = hdist(anCurrIntraDesc[c],anBGIntraDesc[c]);
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],nCurrImgCoord_X,nCurrImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					const size_t nInterDescDist = hdist(anCurrInterDesc[c],anBGIntraDesc[c]);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
					goto failedcheck3ch;
				if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;
				if(nMinTotSumDist>nTotSumDist)
					nMinTotSumDist = nTotSumDist;
				nGoodSamplesCount++;
				failedcheck3ch:
				nSampleIdx++;
			}
			const float fNormalizedLastDist = ((float)L1dist<3>(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist<3>(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[nPxIter] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
					}
				}
			}
			else {
				// == background
				const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
					}
				}
				int nSampleImgCoord_Y, nSampleImgCoord_X;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
					const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
					const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
				if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
					*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
				*pfCurrLearningRate = m_fCurrLearningRateUpperCap;
			if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
				(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
			else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
				(*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
				if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
					(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
				(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
			else {
				(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount<3>(anCurrIntraDesc)>=4)
				++nNonZeroDescCount;
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
		}
	}
#if DISPLAY_SUBSENSE_DEBUG_INFO
	std::cout << std::endl;
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
	cv::circle(oMeanRawSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
	cv::circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oVariationModulatorFrameNormalized; cv::normalize(m_oVariationModulatorFrame,oVariationModulatorFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::circle(oVariationModulatorFrameNormalized,dbgpt,5,cv::Scalar(255));
	cv::resize(oVariationModulatorFrameNormalized,oVariationModulatorFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("v(x)",oVariationModulatorFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      v(" << dbgpt << ") = " << m_oVariationModulatorFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/FEEDBACK_T_UPPER,-FEEDBACK_T_LOWER/FEEDBACK_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
	cv::bitwise_xor(oCurrFGMask,m_oLastRawFGMask,m_oCurrRawFGBlinkMask);
	cv::bitwise_or(m_oCurrRawFGBlinkMask,m_oLastRawFGBlinkMask,m_oBlinksFrame);
	m_oCurrRawFGBlinkMask.copyTo(m_oLastRawFGBlinkMask);
	oCurrFGMask.copyTo(m_oLastRawFGMask);
	// complete
	cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
	const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nTotRelevantPxCount;
	if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
	    for(size_t t=0; t<=UCHAR_MAX; ++t)
	        if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
	            --m_anLBSPThreshold_8bitLUT[t];
	}
	else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
	    for(size_t t=0; t<=UCHAR_MAX; ++t)
	        if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
	            ++m_anLBSPThreshold_8bitLUT[t];
	}
	m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
	if(m_bLearningRateScalingEnabled) {
		cv::resize(oInputImg,m_oDownSampledFrame_MotionAnalysis,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
		cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
		cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
		size_t nTotColorDiff = 0;
		for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
			const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
			for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
				const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
				nTotColorDiff += (m_nImgChannels==1)?
					(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
							:  //(m_nImgChannels==3)
						std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
							std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
										(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
			}
		}
		const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
		if(m_bAutoModelResetEnabled) {
			if(m_nFramesSinceLastReset>1000)
				m_bAutoModelResetEnabled = false;
			else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD && m_nModelResetCooldown==0) {
				m_nFramesSinceLastReset = 0;
				refreshModel(0.1f); // reset 10% of the bg model
				m_nModelResetCooldown = m_nSamplesForMovingAvgs/4;
				m_oUpdateRateFrame = cv::Scalar(1.0f);
			}
			else
				++m_nFramesSinceLastReset;
		}
		else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD*2) {
			m_nFramesSinceLastReset = 0;
			m_bAutoModelResetEnabled = true;
		}
		if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD/2) {
			m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
			m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
		}
		else {
			m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
			m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
		}
		if(m_nModelResetCooldown>0)
			--m_nModelResetCooldown;
	}
}

void BackgroundSubtractorSuBSENSE::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t s=0; s<m_nBGSamples; ++s) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
				const size_t nFloatIter = idx_nimg*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+nFloatIter);
				const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

void BackgroundSubtractorSuBSENSE::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_Assert(LBSP::DESC_SIZE==2);
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
				const size_t nFloatIter = idx_ndesc*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+nFloatIter);
				const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
			}
		}
	}
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

void BackgroundSubtractorSuBSENSE::update(const cv::Mat &newFrame, const cv::Mat &transmatrix) {
	m_oLastColorFrame = newFrame.clone();
	cv::warpPerspective(m_oLastDescFrame, m_oLastDescFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oLastFGMask, m_oLastFGMask, transmatrix, m_oImgSize);
	for (int i = 0; i < (int) m_voBGColorSamples.size(); i ++)
		cv::warpPerspective(m_voBGColorSamples[i], m_voBGColorSamples[i], transmatrix, m_oImgSize);
	for (int i = 0; i < (int) m_voBGDescSamples.size(); i ++)
		cv::warpPerspective(m_voBGDescSamples[i], m_voBGDescSamples[i], transmatrix, m_oImgSize);
	cv::warpPerspective(m_oUpdateRateFrame, m_oUpdateRateFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oDistThresholdFrame, m_oDistThresholdFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oVariationModulatorFrame, m_oVariationModulatorFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanLastDistFrame, m_oMeanLastDistFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_LT, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanMinDistFrame_ST, m_oMeanMinDistFrame_ST, transmatrix, m_oImgSize);
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::warpPerspective(m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_LT, transmatrix, m_oDownSampledFrameSize);
	cv::warpPerspective(m_oMeanDownSampledLastDistFrame_ST, m_oMeanDownSampledLastDistFrame_ST, transmatrix, m_oDownSampledFrameSize);
	//! pre-allocated matrix used to downsample the input frame when needed
	cv::warpPerspective(m_oDownSampledFrame_MotionAnalysis, m_oDownSampledFrame_MotionAnalysis, transmatrix, m_oDownSampledFrameSize);
	
	cv::warpPerspective(m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_LT, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanRawSegmResFrame_ST, m_oMeanRawSegmResFrame_ST, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_LT, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oMeanFinalSegmResFrame_ST, m_oMeanFinalSegmResFrame_ST, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oUnstableRegionMask, m_oUnstableRegionMask, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oBlinksFrame, m_oBlinksFrame, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oLastRawFGMask, m_oLastRawFGMask, transmatrix, m_oImgSize);
	// m_oLastRawFGMask = resultFrame.clone();
	//! pre-allocated CV_8UC1 matrices used to speed up morph ops
	cv::warpPerspective(m_oFGMask_PreFlood, m_oFGMask_PreFlood, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oFGMask_FloodedHoles, m_oFGMask_FloodedHoles, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oLastFGMask_dilated, m_oLastFGMask_dilated, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oLastFGMask_dilated_inverted, m_oLastFGMask_dilated_inverted, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oCurrRawFGBlinkMask, m_oCurrRawFGBlinkMask, transmatrix, m_oImgSize);
	cv::warpPerspective(m_oLastRawFGBlinkMask, m_oLastRawFGBlinkMask, transmatrix, m_oImgSize);
	
	// initialize empty pixel after transform
	if (m_nImgChannels == 1) {
		for (size_t nPxIter=0; nPxIter < m_nTotPxCount; nPxIter ++) {
			if (m_oROI.data[nPxIter] && m_oUpdateRateFrame.at<float>(nPxIter) < m_fCurrLearningRateLowerCap) {
				const size_t nDescIter = nPxIter*2;
				LBSP::computeGrayscaleDescriptor(newFrame, newFrame.data[nPxIter], m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, m_anLBSPThreshold_8bitLUT[newFrame.data[nPxIter]], *((ushort*)(m_oLastDescFrame.data+nDescIter)));
				m_oUpdateRateFrame.at<float>(nPxIter) = m_fCurrLearningRateLowerCap;
				m_oDistThresholdFrame.at<float>(nPxIter) = 1.0f;
				m_oVariationModulatorFrame.at<float>(nPxIter) = 10.0f;
				for (size_t nCurrModelIdx = 0; nCurrModelIdx < m_nBGSamples; nCurrModelIdx ++) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if (!m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter] = m_oLastColorFrame.data[nSamplePxIdx];
						*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+nPxIter*2)) = *((ushort*)(m_oLastDescFrame.data+nSamplePxIdx*2));
					}
				}
			}
		}
	}
	else { // m_nImgChannels == 3
		for (size_t nPxIter = 0; nPxIter < m_nTotPxCount; nPxIter++) {
			if (m_oROI.data[nPxIter] && m_oUpdateRateFrame.at<float>(nPxIter) < m_fCurrLearningRateLowerCap) {
				const size_t nPxRGBIter = nPxIter*3;
				const size_t nDescRGBIter = nPxRGBIter*2;
				for (size_t c = 0; c < 3; c ++) {
					LBSP::computeSingleRGBDescriptor(newFrame, newFrame.data[nPxRGBIter+c], m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, c, m_anLBSPThreshold_8bitLUT[newFrame.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
				}
				m_oUpdateRateFrame.at<float>(nPxIter) = m_fCurrLearningRateLowerCap;
				m_oDistThresholdFrame.at<float>(nPxIter) = 1.0f;
				m_oVariationModulatorFrame.at<float>(nPxIter) = 10.0f;
				for (size_t nCurrModelIdx = 0; nCurrModelIdx < m_nBGSamples; nCurrModelIdx ++) {
					int nSampleImgCoord_Y, nSampleImgCoord_X;
					getRandSamplePosition(nSampleImgCoord_X, nSampleImgCoord_Y, m_aPxInfoLUT[nPxIter].nImgCoord_X, m_aPxInfoLUT[nPxIter].nImgCoord_Y, LBSP::PATCH_SIZE/2, m_oImgSize);
					const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
					if (!m_oLastFGMask.data[nSamplePxIdx]) {
						const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
						for(size_t c = 0; c < 3; c ++) {
							m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter*3+c] = m_oLastColorFrame.data[nSamplePxIdx*3+c];
							*((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+(nPxIter*3+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*3+c)*2));
						}
					}
				}
			}
		}
	}	
}

void BackgroundSubtractorSuBSENSE::complete(cv::OutputArray &fgMask) {
	cv::Mat a = fgMask.getMat();
	imwrite("a0.jpg", a);
	cv::morphologyEx(a,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(a,m_oFGMask_FloodedHoles,a);
	imwrite("a1.jpg", a);
	cv::bitwise_or(a,m_oFGMask_PreFlood,a);
	imwrite("a2.jpg", a);
	cv::medianBlur(a,m_oLastFGMask,m_nMedianBlurKernelSize);
	cv::dilate(m_oLastFGMask,m_oLastFGMask_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_not(m_oLastFGMask_dilated,m_oLastFGMask_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
	m_oLastFGMask.copyTo(a);
	a.convertTo(fgMask, CV_8U);
}

int BackgroundSubtractorSuBSENSE::dist(const cv::Mat &a, const cv::Mat &b, int ax, int ay, int bx, int by, int cutoff) const{
	int ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			size_t anPxIter = ay * m_oImgSize.width + ax, bnPxIter = by * m_oImgSize.width + bx;
			if (a.channels() == 1) {
				int d = a.data[anPxIter] - b.data[bnPxIter];
				ans += d*d;
			} else {
				anPxIter *= 3, bnPxIter *= 3;
				for (size_t c = 0; c < 3; c++) {
					int d = a.data[anPxIter+c] - b.data[bnPxIter+c];
					ans += d*d;
				}
			}
		}
		if (ans >= cutoff) { return cutoff; }
	}
	return ans;
}

void BackgroundSubtractorSuBSENSE::improve_guess(const cv::Mat &a, const cv::Mat &b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by) {
	int d = dist(a, b, ax, ay, bx, by, dbest);
	if (d < dbest) {
		dbest = d;
		xbest = bx;
		ybest = by;
	}
}

void BackgroundSubtractorSuBSENSE::patch_match(const cv::Mat &a, const cv::Mat &b, std::vector<cv::Point2i> &ans, cv::Mat & ansMat, cv::Mat matrix) {
	/* Initialize with random nearest neighbor field (NNF). */
	ans.clear();
	ans.resize(m_nTotPxCount);
	std::vector<double> annd;
	annd.resize(m_nTotPxCount);
	ansMat.create(a.size(), CV_8UC1);
	ansMat = cv::Scalar(0);

	int aew = a.cols - patch_w2 + 1, aeh = a.rows - patch_w2 + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = a.cols - patch_w2 + 1, beh = a.rows - patch_w2 + 1;
	std::vector<cv::Point2f> p1, p2;
	for (int ay = 0; ay < aeh; ay++)
		for (int ax = 0; ax < aew; ax++) p1.push_back(cv::Point2f(ax, ay));
	p2.reserve(p1.size());
	cv::perspectiveTransform(p1, p2, matrix);

	int p = 0;
	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			int idx = ay*m_oImgSize.width + ax;
			int bx = int(p2[p].x);
			int by = int(p2[p++].y);
			if (!(0<=bx && bx<m_oImgSize.width && 0<=by && by<m_oImgSize.height)) {
				bx = rand()%bew; by = rand()%beh;
				ans[idx] = cv::Point2i(bx, by);
				annd[idx] = dist(a, b, ax, ay, bx, by);
				for (int ii = 0; ii < pm_iters; ii++) {
					bx = rand()%bew; by = rand()%beh;
					int disnow = dist(a, b, ax, ay, bx, by);
					if (disnow < annd[idx]) {
						ans[idx] = cv::Point2i(bx, by);
						annd[idx] = disnow;
					}
				}
			} else {
				ans[idx] = cv::Point2i(bx, by);
				annd[idx] = dist(a, b, ax, ay, bx, by);
			}
			annd[idx] /= patch_w2 * patch_w2;
			if (annd[idx] / 52 >= 1) ansMat.data[idx] = 255;
			else ansMat.data[idx] = int(annd[idx] / 52.0 * 255);
		}
	}
	
	/*
	for (int iter = 0; iter < pm_iters; iter++) {
		// In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. 
		int ystart = 0, yend = aeh, ychange = 1;
		int xstart = 0, xend = aew, xchange = 1;
		if (iter % 2 == 1) {
			xstart = xend-1; xend = -1; xchange = -1;
			ystart = yend-1; yend = -1; ychange = -1;
		}
		for (int ay = ystart; ay != yend; ay += ychange)
			for (int ax = xstart; ax != xend; ax += xchange) { 
				// Current (best) guess. 
				size_t nPxIter = ay*m_oImgSize.width + ax;
				cv::Point2i v = ans[nPxIter];
				int xbest = v.x, ybest = v.y;
				int dbest = annd[nPxIter];

				// Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). 
				if ((unsigned) (ax - xchange) < (unsigned) aew) {
					cv::Point2i vp = ans[nPxIter-xchange];
					int xp = vp.x + xchange, yp = vp.y;
					if ((unsigned) xp < (unsigned) bew)
						improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
				}
				if ((unsigned) (ay - ychange) < (unsigned) aeh) {
					cv::Point2i vp = ans[(ay-ychange) * m_oImgSize.width + ax];
					int xp = vp.x, yp = vp.y + ychange;
					if ((unsigned) yp < (unsigned) beh)
						improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
				}

				// Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. 
				int rs_start = rs_max;
				if (rs_start > MAX(m_oImgSize.width, m_oImgSize.height)) rs_start = MAX(m_oImgSize.width, m_oImgSize.height);
				for (int mag = rs_start; mag >= 1; mag /= 2) {
					// Sampling window 
					int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
					int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);
					int xp = xmin+ ((xmax-xmin) == 0? 0 : rand()%(xmax-xmin));
					int yp = ymin+ ((ymax-ymin) == 0? 0 : rand()%(ymax-ymin));
					improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
				}

				ans[nPxIter] = cv::Point2i(xbest, ybest);
				annd[nPxIter] = dbest;
			}
    }
	*/
}

void BackgroundSubtractorSuBSENSE::cover(cv::Mat &a, const cv::Mat &b, std::vector<cv::Point2i> &ans) {
	int aew = m_oImgSize.width - patch_w + 1, aeh = m_oImgSize.height - patch_w + 1;
	int bew = m_oImgSize.width - patch_w + 1, beh = m_oImgSize.height - patch_w + 1;
	for (int ay = 0; ay < aeh; ay+=patch_w )
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int bx = ans[ay * m_oImgSize.width + ax].x, by = ans[ay * m_oImgSize.width + ax].y;
			for (int ii = 0; ii < patch_w; ii ++)
				for (int jj = 0; jj < patch_w; jj ++) {
					size_t anPxIter = (ay + ii) * m_oImgSize.width + ax + jj, bnPxIter = (by + ii) * m_oImgSize.width + bx + jj;
					if (m_nImgChannels==1) {
						a.data[anPxIter] = b.data[bnPxIter];
					} else {
						anPxIter *= 3, bnPxIter *= 3;
						for (size_t c = 0; c < 3; c++)
							a.data[anPxIter+c] = b.data[bnPxIter+c];
					}
				}
		}
}

void BackgroundSubtractorSuBSENSE::randomField(cv::Mat & image, cv::Mat & ansMat, cv::Mat & lastMat, cv::OutputArray & fgMask) {
	cv::Mat a = fgMask.getMat();
	cv::addWeighted(a, 0.5, ansMat, 0.5, 0, a);
	cv::addWeighted(a, 0.8, lastMat, 0.2, 0, a);
    int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;
	int ww = (aew - 1) / patch_w + 1, hh = (aeh - 1) / patch_w + 1;
	int size = ww * hh;
	Graph *graph = new Graph(size);
	
	std::vector<double> foreNum;
	foreNum.reserve(size);
	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			float ss = 0; //, s2 = 0;
			for (int ii = 0; ii < patch_w; ii ++)
				for (int jj = 0; jj < patch_w; jj ++)
					ss += 1 - a.data[(ay + ii) * image.cols + ax + jj]/255.0;
					//if (a.data[(ay + ii) * image.cols + ax + jj] == 0) ss ++;
			foreNum.push_back(ss);

			// float ps = L1 * MIN(1.0f, (ss + 0.0) / patch_area / 0.5) + L2 * MAX(0.0f, MIN(1.0, ((s2 + 0.0) / patch_area - 0.3) / 0.4));
			float ps = (ss + 0.0) / patch_area;
			// if (s2 < 8) ps = MAX(0.0f, ps - 0.4);
			float d = MIN(1.0f, ps * 2);
			d = MAX(1e-20f, d);
			float d1 = -log(d);
			d = MAX(1e-20f, 1 - d);
			float d2 = -log(d);

			if (d1 > d2) graph->add_edge(graph->getS(), foreNum.size()-1, d1 - d2, 0);
			else graph->add_edge(foreNum.size()-1, graph->getT(), d2 - d1, 0);
		}

	double avgDistance = 0;
	std::vector<double> edgeLen;
	double len;
	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			if (ay>0) {
				len = dist(image, image, ax, ay, ax, ay - patch_w);
				edgeLen.push_back(len);
				avgDistance += len;
			}
			if (ax>0) {
				len = dist(image, image, ax, ay, ax - patch_w, ay);
				edgeLen.push_back(len);
				avgDistance += len;
			}
			if (ax>0 && ay>0) {
				len = dist(image, image, ax, ay, ax - patch_w, ay - patch_w);
				edgeLen.push_back(len);
				avgDistance += len;
			}
			if (ay>0 && ax + patch_w < aew) {
				len = dist(image, image, ax, ay, ax + patch_w, ay - patch_w);
				edgeLen.push_back(len);
				avgDistance += len;
			}
		}
    avgDistance /= edgeLen.size();

	float lmd1 = 0.3;
    float lmd2 = 0.3;
	float cap;
	int countIdx = 0;
	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int idx = ay/patch_w * ww + ax/patch_w;
			if (ay>0) {
				cap = lmd1 + lmd2 * exp(-edgeLen[countIdx++] / 2 / avgDistance);
				graph->add_edge(idx, idx-ww, cap, cap);
			}
			if (ax>0) {
				cap = lmd1 + lmd2 * exp(-edgeLen[countIdx++] / 2 / avgDistance);
				graph->add_edge(idx, idx-1, cap, cap);
			}
			if (ax>0 && ay>0) {
				cap = lmd1 + lmd2 * exp(-edgeLen[countIdx++] / 2 / avgDistance);
				graph->add_edge(idx, idx-ww-1, cap, cap);
			}
			if (ay>0 && ax + patch_w < aew) {
				cap = lmd1 + lmd2 * exp(-edgeLen[countIdx++] / 2 / avgDistance);
				graph->add_edge(idx, idx-ww+1, cap, cap);
			}
		}

    graph->dinic(graph->getS());
	graph->extended_path();
	std::vector<int> lebal;
	lebal.resize(size);

	cv::Mat aaa = a.clone();
	aaa = cv::Scalar(0);
	cv::Mat bbb = aaa.clone();
	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int idx = ay/patch_w * ww + ax/patch_w;
			if (graph->check_type(idx))
				for (int ii = 0; ii < patch_w; ii ++)
					for (int jj = 0; jj < patch_w; jj ++) {
						size_t anPxIter = (ay + ii) * m_oImgSize.width + ax + jj;
						bbb.data[anPxIter] = 255;
					}
			bool f1 = false, f2 = false;
			if (ax > 0) {
				if (graph->check_type(idx-1)) f1 = true;
				else f2 = true;
			}
			if (ay > 0) {
				if (graph->check_type(idx-ww)) f1 = true;
				else f2 = true;
			}
			if (ax + patch_w < aew) {
				if (graph->check_type(idx+1)) f1 = true;
				else f2 = true;
			}
			if (ay + patch_w < aeh) {
				if (graph->check_type(idx+ww)) f1 = true;
				else f2 = true;
			}
			if (f1 && f2) {
				for (int ii = 0; ii < patch_w; ii ++)
					for (int jj = 0; jj < patch_w; jj ++) {
						size_t anPxIter = (ay + ii) * m_oImgSize.width + ax + jj;

					aaa.data[anPxIter] = 255;
				}
				lebal[idx] = 1;
				continue;
			}
		}
	cv::imwrite("A.jpg", bbb);
	cv::imwrite("B.jpg", aaa);

	for (int ay = 0; ay < aeh; ay+=patch_w)
		for (int ax = 0; ax < aew; ax+=patch_w) {
			int idx = ay/patch_w * ww + ax/patch_w;
			bool flag = lebal[idx];
			if (ax > 0 && lebal[idx-1]) flag = true;
			if (ay > 0 && lebal[idx-ww]) flag = true;
			if (ax + patch_w < aew && lebal[idx+1]) flag = true;
			if (ay + patch_w < aeh && lebal[idx+ww]) flag = true;
			if (flag) {
				// on the edge
				for (int ii = 0; ii < patch_w; ii ++)
					for (int jj = 0; jj < patch_w; jj ++) {
						size_t anPxIter = (ay + ii) * m_oImgSize.width + ax + jj;
						a.data[anPxIter] = a.data[anPxIter]>155? 255:0;
					}
			} else {
				// full the patch
				int goa = graph->check_type(idx)? 255:0;
				for (int ii = 0; ii < patch_w; ii ++)
					for (int jj = 0; jj < patch_w; jj ++) {
						size_t anPxIter = (ay + ii) * m_oImgSize.width + ax + jj;
						a.data[anPxIter] = goa;
						aaa.data[anPxIter] = 255;
					}
			}
		}
	cv::imwrite("C.jpg", aaa);
	a.convertTo(fgMask, CV_8U);
}