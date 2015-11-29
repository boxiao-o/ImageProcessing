//
//  XBImageContainerView.m
//  XB_OpenCVPlat
//
//  Created by xiaobo on 14/11/19.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#import "XBImageContainerView.h"



@implementation XBImageContainerView
// -----------------------------------
// Initialize the View
// -----------------------------------

- (id)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
//        [self setItemPropertiesToDefault:self];
    }
    return self;
}

// -----------------------------------
// Release the View
// -----------------------------------

- (void)dealloc
{
}


// -----------------------------------
// Handle Mouse Events
// -----------------------------------

-(void)mouseDown:(NSEvent *)event
{
    NSPoint clickLocation;
    
    // convert the click location into the view coords
    clickLocation = [self convertPoint:[event locationInWindow]
                              fromView:nil];
    dragging=YES;
    
    // store the starting click location;
    lastDragLocation=clickLocation;
    
    // set the cursor to the closed hand cursor
    // for the duration of the drag
    [[NSCursor closedHandCursor] push];
}

-(void)mouseDragged:(NSEvent *)event
{
    if (dragging) {
        NSPoint newDragLocation=[self convertPoint:[event locationInWindow]
                                          fromView:nil];
        float vx = newDragLocation.x - lastDragLocation.x;
        float vy = newDragLocation.y - lastDragLocation.y;
        
        lastDragLocation = newDragLocation;
        [[self window] invalidateCursorRectsForView:self];
        NSDictionary* dic = @{@"x": [NSNumber numberWithFloat:lastDragLocation.x], @"y":[NSNumber numberWithFloat:lastDragLocation.y], @"vx":[NSNumber numberWithFloat:vx], @"vy":[NSNumber numberWithFloat:vy]};
        [[NSNotificationCenter defaultCenter] postNotificationName:@"changeCord" object:self userInfo:dic];
    }
}

-(void)mouseUp:(NSEvent *)event
{
    dragging=NO;

    [NSCursor pop];
    
    NSPoint newDragLocation=[self convertPoint:[event locationInWindow]
                                      fromView:nil];
    float vx = newDragLocation.x - lastDragLocation.x;
    float vy = newDragLocation.y - lastDragLocation.y;

    
    [[self window] invalidateCursorRectsForView:self];
    NSDictionary* dic = @{@"x": [NSNumber numberWithFloat:lastDragLocation.x], @"y":[NSNumber numberWithFloat:lastDragLocation.y], @"vx":[NSNumber numberWithFloat:vx], @"vy":[NSNumber numberWithFloat:vy]};
    [[NSNotificationCenter defaultCenter] postNotificationName:@"changeCord" object:self userInfo:dic];
}


// -----------------------------------
// First Responder Methods
// -----------------------------------

- (BOOL)acceptsFirstResponder
{
    return YES;
}
@end
