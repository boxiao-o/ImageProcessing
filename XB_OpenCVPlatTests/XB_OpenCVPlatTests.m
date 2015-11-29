//
//  XB_OpenCVPlatTests.m
//  XB_OpenCVPlatTests
//
//  Created by 小白 on 14-10-17.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

@interface XB_OpenCVPlatTests : XCTestCase

@end

@implementation XB_OpenCVPlatTests

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void)testExample {
    // This is an example of a functional test case.
    XCTAssert(YES, @"Pass");
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
