//
// Created by trongan93 on 2/27/24.
//

#ifndef INFERENCE_LL_ENHANCEMENT_ALGORITHM_H
#define INFERENCE_LL_ENHANCEMENT_ALGORITHM_H


#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

void cvReadImg(std::string fileName);
void cvOutputImg();
void qNorm();
void qNorm_256();
void qDownSample();
void qConv1st();
void qConv2nd();
void qConv3rd();
void qConv3rdV2();
void qUpSample();
void qEnhance();
void qEnhance_256();
void qEnhance_256_q8();
void qEnhance_256_q4(); // Not applicable

void writeImageData(std::string fileName);
//void qDCENet();
void qLoadParam();

void initialMem();
void cleanMem();



#endif //INFERENCE_LL_ENHANCEMENT_ALGORITHM_H
