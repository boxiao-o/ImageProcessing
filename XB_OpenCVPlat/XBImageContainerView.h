//
//  XBImageContainerView.h
//  XB_OpenCVPlat
//
//  Created by xiaobo on 14/11/19.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@interface XBImageContainerView : NSView{
    NSPoint location;
    // private variables that track state
    BOOL dragging;
    NSPoint lastDragLocation;
}

- (id)initWithFrame:(NSRect)frame;

// -----------------------------------
// Handle Mouse Events
// -----------------------------------

-(void)mouseDown:(NSEvent *)event;
-(void)mouseDragged:(NSEvent *)event;
-(void)mouseUp:(NSEvent *)event;

// -----------------------------------
// First Responder Methods
// -----------------------------------

- (BOOL)acceptsFirstResponder;



@end
