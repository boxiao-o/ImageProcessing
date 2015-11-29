//
//  AppDelegate.m
//  XB_OpenCVPlat
//
//  Created by xiaobo on 14-10-17.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#import "AppDelegate.h"
#import "NSImage+OpenCV.h"
#import "imageProcess.h"
#include <string>
@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;

@end

@implementation AppDelegate
@synthesize imageContainer;
@synthesize inputImage;
@synthesize outputImage;
@synthesize vignetteSlider;
@synthesize imageFilters;
@synthesize gaussianSlider;
@synthesize saturationSlider;
@synthesize imageView;
@synthesize imageSize;
@synthesize liquifySlider;
@synthesize retouchSlider;
@synthesize whitenSlider;
@synthesize lomoTypes;
@synthesize hub;

static const int IMAGE_FILTER_ORIGIN = 0;
static const int IMAGE_FILTER_VIGNETTE = 1;
static const int IMAGE_FILTER_WHITE_BALANCE = 2;
static const int IMAGE_FILTER_MEDIAN_FILTER = 3;
static const int IMAGE_FILTER_GAUSSIAN_BLUR = 4;
static const int IMAGE_FILTER_SATURATION = 5;
static const int IMAGE_FILTER_SHARPEN = 6;
static const int IMAGE_FILTER_LIQUIFY = 7;
static const int IMAGE_FILTER_BEAUTIFY = 8;
static const int IMAGE_FILTER_LOMO = 9;


- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Insert code here to initialize your application
    [hub setHidden:YES];
   
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Insert code here to tear down your application
}

- (IBAction)exportImage:(id)sender {
    if (imageView.image == nil) {
        return;
    }
    if (inputImage.empty()) {
        inputImage = [imageView.image CVMat];
        outputImage = inputImage;
    }
    NSSavePanel*    panel = [NSSavePanel savePanel];
    [panel setNameFieldStringValue:@"Untitle"];
    [panel setMessage:@"Choose the path to save the document"];
    [panel setAllowsOtherFileTypes:YES];
    [panel setAllowedFileTypes:@[@"png", @"bmp", @"jpg", @"jpeg"]];
    [panel setExtensionHidden:NO];
    [panel setCanCreateDirectories:YES];
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSInteger result){
        if (result == NSFileHandlingPanelOKButton)
        {
            NSString *path = [[panel URL] path];
            const char *char_gurl = [path UTF8String];
            cv::imwrite(char_gurl, outputImage);
        }
    }];
    
}

- (IBAction)importImage:(id)sender {
    [imageFilters selectCellWithTag:IMAGE_FILTER_ORIGIN];
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedFileTypes:[NSImage imageFileTypes]];
    [panel beginSheetModalForWindow:nil completionHandler: (^(NSInteger result){
        if(result == NSOKButton) {
            NSArray *fileURLs = [panel URLs];
            NSImage *image = [[NSImage alloc] initWithContentsOfURL:
                              [fileURLs objectAtIndex:0]];
            NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithData:[image TIFFRepresentation]];
            NSSize size = NSMakeSize([rep pixelsWide], [rep pixelsHigh]);
            [image setSize: size];
            imageSize = size;
            NSLog(@"%@", [fileURLs objectAtIndex:0]);
            inputImage = [image CVMat];
            outputImage = inputImage;
            NSImage* outputNSImage;
            outputNSImage = [NSImage imageWithCVMat:outputImage];
            [imageView setImage:outputNSImage];
        }
    })];
}

- (IBAction)slideVignette:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_VIGNETTE ) {
        double vignettepower = vignetteSlider.integerValue/100.0;
        outputImage = vignette(inputImage, vignettepower);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
    }
}

- (IBAction)chooseImageFilters:(id)sender {
    [hub setHidden:NO];
    [hub startAnimation:sender];
    
    [[NSNotificationCenter defaultCenter] removeObserver:self name:@"changeCord" object:nil];
    
    if (imageView.image == nil) {
        return;
    }
    if (inputImage.empty()) {
        inputImage = [imageView.image CVMat];
        outputImage = inputImage;
    }
    NSButtonCell *selCell = [sender selectedCell];
    switch ([selCell tag]) {
        case IMAGE_FILTER_ORIGIN:
            outputImage = inputImage;
            break;
        case IMAGE_FILTER_VIGNETTE:{
                double vignettepower = vignetteSlider.integerValue/100.0;
                outputImage = vignette(inputImage, vignettepower);
            }
            break;
        case IMAGE_FILTER_WHITE_BALANCE:
            outputImage = ycrcbWhiteBalance(inputImage);
            break;
        case IMAGE_FILTER_MEDIAN_FILTER:
            outputImage = medianFilter(inputImage);
            break;
        case IMAGE_FILTER_GAUSSIAN_BLUR:
            outputImage = gaussianBlur(inputImage, (int)gaussianSlider.integerValue);
            break;
        case IMAGE_FILTER_SATURATION:
            outputImage = saturation(inputImage, (int)saturationSlider.integerValue);
            break;
        case IMAGE_FILTER_SHARPEN:
            outputImage = sharpen(inputImage);
            break;
        case IMAGE_FILTER_LIQUIFY:
            outputImage = inputImage;
             [[NSNotificationCenter defaultCenter]addObserver:self selector:@selector(transferCord:) name:@"changeCord" object:nil];
            break;
        case IMAGE_FILTER_BEAUTIFY:
            outputImage = beautify(inputImage, (int)retouchSlider.integerValue, (int)whitenSlider.integerValue);
            break;
        case IMAGE_FILTER_LOMO:
            outputImage = lomo(inputImage, (int)lomoTypes.selectedTag);
            break;
        default:
            
            break;
    }
    NSImage* outputNSImage;
    outputNSImage = [NSImage imageWithCVMat:outputImage];
    [imageView setImage:outputNSImage];
    [hub stopAnimation:sender];
    [hub setHidden:YES];

}
- (IBAction)slideGaussian:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_GAUSSIAN_BLUR ) {
        [hub setHidden:NO];
        [hub startAnimation:sender];
        outputImage = gaussianBlur(inputImage, (int)gaussianSlider.integerValue);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
        [hub stopAnimation:sender];
        [hub setHidden:YES];
    }
}
- (IBAction)slideSaturation:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_SATURATION ) {
        [hub setHidden:NO];
        [hub startAnimation:sender];
        outputImage = saturation(inputImage, (int)saturationSlider.integerValue);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
        [hub stopAnimation:sender];
        [hub setHidden:YES];
    }
}

- (void) transferCord:(NSNotification *)notification{
    float pointx = [[[notification userInfo] valueForKey:@"x"] floatValue];
    float pointy = [[[notification userInfo] valueForKey:@"y"] floatValue];
    float vx = [[[notification userInfo] valueForKey:@"vx"] floatValue];
    float vy = [[[notification userInfo] valueForKey:@"vy"] floatValue];
    
    NSPoint location;
    location.x = pointx;
    location.y = pointy;
    
    NSRect drawingRect = [imageView.cell drawingRectForBounds: imageView.bounds];
    location.x -= drawingRect.origin.x;
    location.y -= drawingRect.origin.y;
    
    NSSize frameSize = drawingRect.size;
    float frameAspect = frameSize.width/frameSize.height;
    float imageAspect = imageSize.width/imageSize.height;
    float scaleFactor = 1.0f;
    
    if(imageAspect > frameAspect) {
        
        ///in this case image.width == frame.width
        scaleFactor = imageSize.width / frameSize.width;
        
        float imageHeightinFrame = imageSize.height / scaleFactor;
        
        float imageOffsetInFrame = (frameSize.height - imageHeightinFrame)/2;
        
        location.y -= imageOffsetInFrame;
        
    } else {
        ///in this case image.height == frame.height
        scaleFactor = imageSize.height / frameSize.height;
        
        float imageWidthinFrame = imageSize.width / scaleFactor;
        
        float imageOffsetInFrame = (frameSize.width - imageWidthinFrame)/2;
        
        location.x -= imageOffsetInFrame;
    }
    location.x *= scaleFactor;
    location.y *= scaleFactor;
    location.y = imageSize.height - location.y;
    vx *= scaleFactor;
    vy *= -scaleFactor;
    outputImage = liquify(outputImage, location.x, location.y, vx, vy, liquifySlider.floatValue);
    NSImage* outputNSImage;
    outputNSImage = [NSImage imageWithCVMat:outputImage];
    [imageView setImage:outputNSImage];
}

- (IBAction)slideRetouch:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_BEAUTIFY ) {
        [hub setHidden:NO];
        [hub startAnimation:sender];
        outputImage = beautify(inputImage, (int)retouchSlider.integerValue, (int)whitenSlider.integerValue);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
        [hub stopAnimation:sender];
        [hub setHidden:YES];
    }
}

- (IBAction)slideWhiten:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_BEAUTIFY ) {
        [hub setHidden:NO];
        [hub startAnimation:sender];
        outputImage = beautify(inputImage, (int)retouchSlider.integerValue, (int)whitenSlider.integerValue);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
        [hub stopAnimation:sender];
        [hub setHidden:YES];
    }
}
- (IBAction)selectLomoType:(id)sender {
    NSButtonCell *selCell = [imageFilters selectedCell];
    if ([selCell tag] == IMAGE_FILTER_LOMO ) {
        [hub setHidden:NO];
        [hub startAnimation:sender];
        outputImage = lomo(inputImage, (int)lomoTypes.selectedTag);
        NSImage* outputNSImage;
        outputNSImage = [NSImage imageWithCVMat:outputImage];
        [imageView setImage:outputNSImage];
        [hub stopAnimation:sender];
        [hub setHidden:YES];
    }
}
@end
