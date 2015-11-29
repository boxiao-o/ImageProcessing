//
//  AppDelegate.h
//  XB_OpenCVPlat
//
//  Created by xiaobo on 14-10-17.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cvaux.hpp"
#import "XBImageContainerView.h"

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (strong) IBOutlet XBImageContainerView *imageContainer;
@property cv::Mat inputImage;
@property cv::Mat outputImage;
- (IBAction)exportImage:(id)sender;
- (IBAction)importImage:(id)sender;
@property (weak) IBOutlet NSSlider *vignetteSlider;
- (IBAction)slideVignette:(id)sender;
@property (weak) IBOutlet NSMatrix *imageFilters;
- (IBAction)chooseImageFilters:(id)sender;
@property (weak) IBOutlet NSSlider *gaussianSlider;
- (IBAction)slideGaussian:(id)sender;
@property (weak) IBOutlet NSSlider *saturationSlider;
- (IBAction)slideSaturation:(id)sender;
@property (weak) IBOutlet NSImageView *imageView;
@property NSSize imageSize;
@property (weak) IBOutlet NSSlider *liquifySlider;
- (void) transferCord:(NSNotification *)notification;
@property (weak) IBOutlet NSSlider *retouchSlider;
@property (weak) IBOutlet NSSlider *whitenSlider;
- (IBAction)slideRetouch:(id)sender;
- (IBAction)slideWhiten:(id)sender;

@property (weak) IBOutlet NSMatrix *lomoTypes;

- (IBAction)selectLomoType:(id)sender;
@property (weak) IBOutlet NSProgressIndicator *hub;
@end

