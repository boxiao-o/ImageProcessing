//
//  NSImage+OpenCV.h
//  OpenCVPlatform2
//
//  Created by xiaobo on 14-10-12.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#ifndef __OpenCVPlatform2__NSImage_OpenCV__
#define __OpenCVPlatform2__NSImage_OpenCV__

#include <stdio.h>
#import <AppKit/AppKit.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cvaux.hpp"

@interface NSImage (NSImage_OpenCV) {
    
}

+(NSImage*)imageWithCVMat:(const cv::Mat&)cvMat;
-(id)initWithCVMat:(const cv::Mat&)cvMat;

@property(nonatomic, readonly) cv::Mat CVMat;
@property(nonatomic, readonly) cv::Mat CVGrayscaleMat;

@end
#endif /* defined(__OpenCVPlatform2__NSImage_OpenCV__) */
