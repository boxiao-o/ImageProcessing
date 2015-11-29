//
//  imageProcess.h
//  OpenCVPlatform2
//
//  Created by xiaobo on 14-10-8.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#ifndef __OpenCVPlatform2__imageProcess__
#define __OpenCVPlatform2__imageProcess__

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cvaux.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

cv::Mat vignette(cv::Mat input, double power);
cv::Mat grayWorld(cv::Mat input);
cv::Mat ycrcbWhiteBalance(cv::Mat input);
cv::Mat medianFilter(cv::Mat input);
cv::Mat gaussianBlur(cv::Mat input, int radius);
cv::Mat saturation(cv::Mat input, int increment);
cv::Mat sharpen(cv::Mat input);
cv::Mat liquify(cv::Mat input, float pointx, float pointy, float vx, float vy, float radius);
cv::Mat beautify(cv::Mat input, int skinretouchpower, int whitenpower);
cv::Mat lomo(cv::Mat input, int type);
#endif /* defined(__OpenCVPlatform2__imageProcess__) */
