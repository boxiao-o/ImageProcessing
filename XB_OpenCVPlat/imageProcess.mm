//
//  imageProcess.cpp
//  OpenCVPlatform2
//
//  Created by xiaobo on 14-10-8.
//  Copyright (c) 2014年 xiaobo. All rights reserved.
//

#include "imageProcess.h"
#include <math.h>
#include <vector>
#define PI 3.141592653
#define EPSILON 0.00001
double dist(cv::Point a, cv::Point b)
{
    return sqrt(pow((double) (a.x - b.x), 2) + pow((double) (a.y - b.y), 2));
}
float dist(float x1, float y1, float x2, float y2){
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}
float biLine(float f00, float f01, float f10, float f11, float u, float v){
    return (1-u)*(1-v)*f00+(1-u)*v*f01+u*(1-v)*f10 + u*v*f11;
}

void generateGradient(cv::Mat& mask, double power)
{
    cv::Point center = cv::Point(mask.size().width/2, mask.size().height/2);
    cv::Point corner = cv::Point(0, 0);
    double center2corner = dist(center, corner);
    
    mask.setTo(cv::Scalar(1));
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double distance = dist(center, cv::Point(j, i)) / center2corner;
            distance *= power;
            mask.at<double>(i, j) = pow(cos(distance), 4);
        }
    }
}
cv::Mat vignette(cv::Mat input, double power){
    cv::Mat output;
    cv::Mat maskImage(input.size(), CV_64F);
    
    generateGradient(maskImage, power);
    
    cv::Mat labImage(input.size(), CV_8UC3);
    cv::cvtColor(input, labImage, CV_BGR2Lab);
    
    for (int row = 0; row < labImage.size().height; row++)
    {
        for (int col = 0; col < labImage.size().width; col++)
        {
            cv::Vec3b value = labImage.at<cv::Vec3b>(row, col);
            value.val[0] *= maskImage.at<double>(row, col);
            labImage.at<cv::Vec3b>(row, col) =  value;
        }
    }
    
    cv::cvtColor(labImage, output, CV_Lab2BGR);
    return output;
}
int sign(double x){
    if (x > 0)
        return 1;
    else if(x < 0)
        return -1;
    else
        return 0;
}
cv::Mat ycrcbWhiteBalance(cv::Mat input){
    cv::Mat output(input.size(), CV_8UC3);
    cv::Mat YCrCbImage(input.size(), CV_8UC3);
    cv::cvtColor(input, YCrCbImage, CV_BGR2YCrCb);
    std::vector<cv::Mat> ycrcb(3);
    cv::split(YCrCbImage, ycrcb);
    double Mcr = cv::sum(ycrcb[1])[0];
    double Mcb = cv::sum(ycrcb[2])[0];
    int width = input.size().width;
    int height = input.size().height;
    Mcr /= width*height*1.0;
    Mcb /= width*height*1.0;
    cv::Mat temp;
    cv::subtract(ycrcb[1], Mcr, temp);
    double Dr = cv::sum(temp)[0]/(width*height*1.0);
    cv::subtract(ycrcb[2], Mcb, temp);
    double Db = cv::sum(temp)[0]/(width*height*1.0);
    std::vector<int> ciny;
    ciny.clear();
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            double b1 = ycrcb[2].at<uint8_t>(row, col)-(Mcb + Db * sign(Mcb));
            double b2 = ycrcb[1].at<uint8_t>(row, col)-(1.5*Mcr + Dr * sign(Mcr));
            if (b1 < fabs(1.5*Db)&b2<fabs(1.5*Dr)) {
                ciny.push_back(ycrcb[0].at<uint8_t>(row, col));
            }
        }
    }
    std::sort(ciny.begin(), ciny.end());
    int nn = round(ciny.size()/10);
    double mn = ciny[ciny.size()- nn];
    int tstSize = 0;
    double Rav = 0, Gav = 0, Bav = 0;
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            if (ycrcb[0].at<uint8_t>(row, col) > mn) {
                tstSize ++;
                Bav += 1.0*input.at<cv::Vec3b>(row, col)[0];
                Gav += 1.0*input.at<cv::Vec3b>(row, col)[1];
                Rav += 1.0*input.at<cv::Vec3b>(row, col)[2];
            }
        }
    }
    Bav /= width*height*1.0;
    Gav /= width*height*1.0;
    Rav /= width*height*1.0;
    double Ymax, Ymin;
    cv::minMaxLoc(ycrcb[0], &Ymin, &Ymax);
    Ymax /= 15.0;
    double Bgain = Ymax/Bav;
    double Ggain = Ymax/Gav;
    double Rgain = Ymax/Rav;
    double maxB = 0, maxG = 0, maxR = 0;
    cv::Mat outIm(input.size(), CV_64FC3);
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            outIm.at<cv::Vec3d>(row, col)[0] = input.at<cv::Vec3b>(row, col)[0]*Bgain;
            outIm.at<cv::Vec3d>(row, col)[1] = input.at<cv::Vec3b>(row, col)[1]*Ggain;
            outIm.at<cv::Vec3d>(row, col)[2] = input.at<cv::Vec3b>(row, col)[2]*Rgain;
            if (maxB < outIm.at<cv::Vec3d>(row, col)[0]) {
                maxB = outIm.at<cv::Vec3d>(row, col)[0];
            }
            if (maxG < outIm.at<cv::Vec3d>(row, col)[1]) {
                maxG = outIm.at<cv::Vec3d>(row, col)[1];
            }
            if (maxR < outIm.at<cv::Vec3d>(row, col)[2]) {
                maxR = outIm.at<cv::Vec3d>(row, col)[2];
            }
        }
    }
//    float factor = MAX(MAX(maxR, maxG), maxB)*1.0;
//    if (factor > 255) {
//        outIm.convertTo(outIm, CV_64FC3, 1.0/factor*255.0);
//    }
    outIm.convertTo(output, CV_8UC3);
    return output;
}

cv::Mat grayWorld(cv::Mat input){
    cv::Mat output;
    cv::Mat im(input.size(), CV_32FC3);
    input.convertTo(im, CV_32FC3, 1/255.0);
    float avgR=0, avgG=0, avgB=0;
    for (int row = 0; row < im.size().height; row ++) {
        for (int col = 0; col < im.size().width; col ++) {
            avgB += im.at<cv::Vec3f>(row, col)[0];
            avgG += im.at<cv::Vec3f>(row, col)[1];
            avgR += im.at<cv::Vec3f>(row, col)[2];
        }
    }
    avgB = avgB / (im.size().width*im.size().height*1.0);
    avgG = avgG / (im.size().width*im.size().height*1.0);
    avgR = avgR / (im.size().width*im.size().height*1.0);
    float avgGray = (avgB + avgG + avgR) / 3;
    cv::Mat outIm(input.size(), CV_32FC3);
    float maxR = 0, maxG = 0, maxB = 0;
    for (int row = 0; row < im.size().height; row ++) {
        for (int col = 0; col < im.size().width; col ++) {
            if (fabs(avgB) < EPSILON) {
                outIm.at<cv::Vec3f>(row, col)[0] = im.at<cv::Vec3f>(row, col)[0];
            }
            else{
                outIm.at<cv::Vec3f>(row, col)[0] = (avgGray/avgB)*im.at<cv::Vec3f>(row, col)[0];
            }
            if (fabs(avgG) < EPSILON) {
                outIm.at<cv::Vec3f>(row, col)[1] = im.at<cv::Vec3f>(row, col)[1];
            }
            else{
                outIm.at<cv::Vec3f>(row, col)[1] = (avgGray/avgG)*im.at<cv::Vec3f>(row, col)[1];
            }
            if (fabs(avgR) < EPSILON) {
                outIm.at<cv::Vec3f>(row, col)[2] = im.at<cv::Vec3f>(row, col)[2];
            }
            else{
                outIm.at<cv::Vec3f>(row, col)[2] = (avgGray/avgR)*im.at<cv::Vec3f>(row, col)[2];
            }
            
            if (maxB < outIm.at<cv::Vec3f>(row, col)[0]) {
                maxB = outIm.at<cv::Vec3f>(row, col)[0];
            }
            if (maxG < outIm.at<cv::Vec3f>(row, col)[1]) {
                maxG = outIm.at<cv::Vec3f>(row, col)[1];
            }
            if (maxR < outIm.at<cv::Vec3f>(row, col)[2]) {
                maxR = outIm.at<cv::Vec3f>(row, col)[2];
            }
        }
    }
    float factor = MAX(MAX(maxR, maxG), maxB);
    if (factor > 1) {
        outIm.convertTo(outIm, CV_32FC3, 1/factor);
    }
    outIm.convertTo(output, CV_8UC3, 255);
    return output;
}

cv::Mat medianFilter(cv::Mat input){
    int radius = 2;
    cv::Mat output(input.size(), CV_8UC3);
    cv::Mat labImage(input.size(), CV_8UC3);
    cv::cvtColor(input, labImage, CV_BGR2Lab);
    output = labImage;
    std::vector<int> window;
    for (int row = 0; row < labImage.size().height; row ++) {
        for (int col = 0; col < labImage.size().width; col ++) {
            for (int i = row - radius; i < row + radius + 1; i ++) {
                for (int j = col - radius; j < col + radius + 1; j ++) {
                    if(i >= 0 && i < labImage.size().height && j >= 0 && j < labImage.size().width)
                        window.push_back(labImage.at<cv::Vec3b>(i, j)[0]);
                }
            }
            std::sort(window.begin(), window.end());
            output.at<cv::Vec3b>(row, col)[0] = window[(window.size()-1)/2];
            window.clear();
        }
    }
    cv::cvtColor(output, output, CV_Lab2BGR);
    return output;
}
double gaussian(double x, double mu, double sigma){
    return (1/(sqrt(2*PI)*sigma)*exp(-pow(x-mu,2)/(2*pow(sigma, 2))));
}
double* genGaussianKernel(int radius){
    double *kernel = new double[2*radius + 1];
    double sigma = radius/3.0;
    double sum = 0;
    for (int i = 0; i < 2*radius+1; i ++) {
        kernel[i] = gaussian(i, radius, sigma);
            sum += kernel[i];
    }
    for (int i = 0; i < 2*radius + 1; i ++) {
            kernel[i] /= sum;
    }
    return kernel;
}

cv::Mat gaussianBlur(cv::Mat input, int radius){
    cv::Mat temp(input.size(), CV_64FC3);
    input.convertTo(temp, CV_64FC3, 1/255.0);
    cv::Mat output = cv::Mat::zeros(input.size(), CV_64FC3);
    double* kernel = genGaussianKernel(radius);
    for (int row = 0; row < output.size().height; row ++) {
        for (int col = 0; col < output.size().width; col ++) {
            double b= 0, g = 0, r = 0, sum = 0;
            for (int i = 0; i < 2*radius+1; i ++) {
                if((row - radius + i >= 0)&&(row-radius+i < output.size().height)){
                    b += temp.at<cv::Vec3d>(row-radius+i, col)[0] * kernel[i];
                    g += temp.at<cv::Vec3d>(row-radius+i, col)[1] * kernel[i];
                    r += temp.at<cv::Vec3d>(row-radius+i, col)[2] * kernel[i];
                    sum += kernel[i];
                }
            }
            output.at<cv::Vec3d>(row, col)[0] = b/sum;
            output.at<cv::Vec3d>(row, col)[1] = g/sum;
            output.at<cv::Vec3d>(row, col)[2] = r/sum;
        }
    }
    for (int row = 0; row < output.size().height; row ++) {
        for (int col = 0; col < output.size().width; col ++) {
            double b= 0, g = 0, r = 0, sum = 0;
            for (int j = 0; j < 2*radius+1; j ++) {
                if((col - radius + j >= 0)&&(col - radius + j < output.size().width)){
                    b += output.at<cv::Vec3d>(row, col - radius + j)[0] * kernel[j];
                    g += output.at<cv::Vec3d>(row, col - radius + j)[1] * kernel[j];
                    r += output.at<cv::Vec3d>(row, col - radius + j)[2] * kernel[j];
                    sum += kernel[j];
                }
            }
            output.at<cv::Vec3d>(row, col)[0] = b/sum;
            output.at<cv::Vec3d>(row, col)[1] = g/sum;
            output.at<cv::Vec3d>(row, col)[2] = r/sum;
        }
    }

    cv::Mat output2(input.size(), CV_8UC3);
    output.convertTo(output2, CV_8UC3, 255.0);
    return output2;
}

cv::Mat saturation(cv::Mat input, int increment_){
    cv::Mat output(input.size(), CV_8UC3);
    input.copyTo(output);
    double increment = increment_ / 100.0;
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            int maxRGB = MAX(input.at<cv::Vec3b>(row, col)[0], MAX(input.at<cv::Vec3b>(row, col)[1], input.at<cv::Vec3b>(row, col)[2]));
            int minRGB = MIN(input.at<cv::Vec3b>(row, col)[0], MIN(input.at<cv::Vec3b>(row, col)[1], input.at<cv::Vec3b>(row, col)[2]));
            double delta = (maxRGB - minRGB)/255.0;
            if (fabs(delta) < EPSILON) {
                continue;
            }
            double value = (maxRGB + minRGB)/255.0;
            double L = value / 2;
            double S, alpha;
            if (L < 0.5) {
                S = delta/value;
            }else{
                S = delta/(2-value);
            }
            if (increment > 0) {
                if ((increment + S) >= 1) {
                    alpha = S;
                }else{
                    alpha = 1 - increment;
                }
                alpha = 1/alpha - 1;
                output.at<cv::Vec3b>(row, col)[0] = input.at<cv::Vec3b>(row, col)[0] + (input.at<cv::Vec3b>(row, col)[0] - L*255)*alpha;
                output.at<cv::Vec3b>(row, col)[1] = input.at<cv::Vec3b>(row, col)[1] + (input.at<cv::Vec3b>(row, col)[1] - L*255)*alpha;
                output.at<cv::Vec3b>(row, col)[2] = input.at<cv::Vec3b>(row, col)[2] + (input.at<cv::Vec3b>(row, col)[2] - L*255)*alpha;
            }else{
                alpha = increment;
                output.at<cv::Vec3b>(row, col)[0] = L*255 + (input.at<cv::Vec3b>(row, col)[0] - L*255)*(1+alpha);
                output.at<cv::Vec3b>(row, col)[1] = L*255 + (input.at<cv::Vec3b>(row, col)[1] - L*255)*(1+alpha);
                output.at<cv::Vec3b>(row, col)[2] = L*255 + (input.at<cv::Vec3b>(row, col)[2] - L*255)*(1+alpha);
            }
        }
    }
    return output;
}
cv::Mat sharpen(cv::Mat input){
    cv::Mat output;
    cv::Mat lab, outputlab;
    cv::cvtColor(input, lab, CV_BGR2Lab);
    lab.copyTo(outputlab);
    int w[3][3] = {{0,-1,0},{-1,5,-1},{0,-1,0}};
    for (int row = 1; row < input.size().height - 1; row ++) {
        for (int col = 1; col < input.size().width - 1; col ++) {
            int value = lab.at<cv::Vec3b>(row-1, col-1)[0]*w[0][0] + lab.at<cv::Vec3b>(row-1, col)[0]*w[0][1] + lab.at<cv::Vec3b>(row-1, col+1)[0]*w[0][2] + lab.at<cv::Vec3b>(row, col-1)[0]*w[1][0] + lab.at<cv::Vec3b>(row, col)[0]*w[1][1] + lab.at<cv::Vec3b>(row, col+1)[0]*w[1][2] + lab.at<cv::Vec3b>(row+1, col-1)[0]*w[2][0] + lab.at<cv::Vec3b>(row+1, col)[0]*w[2][1] + lab.at<cv::Vec3b>(row+1, col+1)[0]*w[2][2];

            outputlab.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(value);
        }
    }
    cv::cvtColor(outputlab, output, CV_Lab2BGR);
   return output;
}

int setValid(int x, int up, int down){
    if (x < down) {
        x = down;
    }
    if (x > up) {
        x = up;
    }
    return x;
}

cv::Mat liquify(cv::Mat input, float pointx, float pointy, float vx, float vy, float radius){
    cv::Mat output, tempImage;
    input.copyTo(output);
    input.copyTo(tempImage);
    tempImage.convertTo(tempImage, CV_32FC3, 1/255.0);
    output.convertTo(output, CV_32FC3, 1/255.0);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            float distXtoC = dist(col, row, pointx, pointy);
            if (distXtoC > radius) {
                continue;
            }
            float pointUx, pointUy;
            float radius2minusDist2 = radius*radius - distXtoC*distXtoC;
            float distMtoC = vx*vx + vy*vy;
            float temp = pow(radius2minusDist2/(radius2minusDist2 + 100*distMtoC), 2);
            pointUx = col*1.0 - temp * vx;
            pointUy = row*1.0 - temp * vy;
            int x00 = (int)pointUx;
            int y00 = (int)pointUy;
            int x01 = (int)(pointUx+0.5);
            int y01 = (int)pointUy;
            int x10 = (int)pointUx;
            int y10 = (int)(pointUy+0.5);
            int x11 = (int)(pointUx+0.5);
            int y11 = (int)(pointUy+0.5);
            setValid(x00, input.size().width - 1, 0);
            setValid(x01, input.size().width - 1, 0);
            setValid(x10, input.size().width - 1, 0);
            setValid(x11, input.size().width - 1, 0);
            setValid(y00, input.size().height - 1, 0);
            setValid(y01, input.size().height - 1, 0);
            setValid(y10, input.size().height - 1, 0);
            setValid(y11, input.size().height - 1, 0);
            
            output.at<cv::Vec3f>(row, col)[0] = biLine(tempImage.at<cv::Vec3f>(y00, x00)[0],tempImage.at<cv::Vec3f>(y01, x01)[0],tempImage.at<cv::Vec3f>(y10, x10)[0],tempImage.at<cv::Vec3f>(y11, x11)[0], pointUx - (int)pointUx, pointUy - (int)pointUy);
            output.at<cv::Vec3f>(row, col)[1] = biLine(tempImage.at<cv::Vec3f>(y00, x00)[1],tempImage.at<cv::Vec3f>(y01, x01)[1],tempImage.at<cv::Vec3f>(y10, x10)[1],tempImage.at<cv::Vec3f>(y11, x11)[1], pointUx - (int)pointUx, pointUy - (int)pointUy);
            output.at<cv::Vec3f>(row, col)[2] = biLine(tempImage.at<cv::Vec3f>(y00, x00)[2],tempImage.at<cv::Vec3f>(y01, x01)[2],tempImage.at<cv::Vec3f>(y10, x10)[2],tempImage.at<cv::Vec3f>(y11, x11)[2], pointUx - (int)pointUx, pointUy - (int)pointUy);
        }
    }
    output.convertTo(output, CV_8UC3, 255);
    return output;
}
cv::Mat findContours(cv::Mat mask){
    mask.convertTo(mask, CV_8UC1);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3), cv::Point(-1,-1));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);
    cv::vector< cv::vector<cv::Point> > contours;   // 轮廓
    cv::vector< cv::vector<cv::Point> > filterContours; // 筛选后的轮廓
    cv::vector< cv::Vec4i > hierarchy;    // 轮廓的结构信息
    contours.clear();
    hierarchy.clear();
    filterContours.clear();
    
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // 去除伪轮廓
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (fabs(contourArea(cv::Mat(contours[i]))) > 800/*&&fabs(arcLength(Mat(contours[i]),true))<2000*/)  //判断手进入区域的阈值
            filterContours.push_back(contours[i]);
    }
    
    mask.setTo(0);
    drawContours(mask, filterContours, 0, cv::Scalar(255,0,0), CV_FILLED); //8, hierarchy);
    
    mask.convertTo(mask, CV_8U);
    return mask;
}
cv::Mat skinDetector(cv::Mat input){
    cv::Mat mask(input.size(), CV_8U);
    mask.zeros(input.size(), CV_8U);
    cv::Mat yCrBr, newImage;
    newImage = medianFilter(input);
    cv::cvtColor(newImage, yCrBr, CV_BGR2YCrCb);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            int y = yCrBr.at<cv::Vec3b>(row, col)[0];
            int cr = yCrBr.at<cv::Vec3b>(row, col)[1];
            int cb = yCrBr.at<cv::Vec3b>(row, col)[2];
            cb -= 109;
            cr -= 152;
            int x1 = (819*cr-614*cb)/32 + 51;
            int y1 = (819*cr+614*cb)/32 + 77;
            x1 = x1*41/1024;
            y1 = y1*73/1024;
            int value = x1*x1+y1*y1;
            if (y < 100) {
                mask.at<uchar>(row, col) = (value < 1200)?255:0;
            }else{
                mask.at<uchar>(row, col) = (value < 1350)?255:0;
            }
        }
    }
    return mask;
}
cv::Mat faceDetector(cv::Mat input){
    std::vector<int> rect_x_vec;
    std::vector<int> rect_y_vec;
    std::vector<int> rect_width_vec;
    std::vector<int> rect_height_vec;

    const char *pstrCascadeFileName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    CvHaarClassifierCascade *pHaarCascade = NULL;
    pHaarCascade = (CvHaarClassifierCascade*)cvLoad(pstrCascadeFileName);
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    IplImage grayImage = gray;
    if (pHaarCascade != NULL)
    {
        CvMemStorage *pcvMStorage = cvCreateMemStorage(0);
        cvClearMemStorage(pcvMStorage);
        // 识别
        CvSeq *pcvSeqFaces = cvHaarDetectObjects(&grayImage, pHaarCascade, pcvMStorage);
        
        // 标记
        for(int i = 0; i <pcvSeqFaces->total; i++)
        {
            CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);
            rect_x_vec.push_back(r->x);
            rect_y_vec.push_back(r->y);
            rect_width_vec.push_back(r->width);
            rect_height_vec.push_back(r->height);
        }

        cvReleaseMemStorage(&pcvMStorage);
    }
    
    cv::Mat mask(input.size(), CV_8U);
    mask.zeros(input.size(), CV_8U);
    cv::Mat yCrBr, newImage;
    newImage = medianFilter(input);
    cv::cvtColor(newImage, yCrBr, CV_BGR2YCrCb);
//    if(false){
    if (rect_x_vec.size() != 0) {
        for (int i = 0; i < rect_x_vec.size(); i ++) {
            
            int width = rect_width_vec.at(i);
            int height = rect_height_vec.at(i);
            int start_x = rect_x_vec.at(i) - width*0.2;;
            int start_y = rect_y_vec.at(i) - height*0.1;
            for (int row = start_y; row <start_y+height*1.6; row ++) {
                for (int col = start_x; col < start_x + width*1.2; col ++) {
                    if (row < 0 || row > input.size().width || col < 0 || col > input.size().width) {
                        continue;
                    }
                    int y = yCrBr.at<cv::Vec3b>(row, col)[0];
                    int cr = yCrBr.at<cv::Vec3b>(row, col)[1];
                    int cb = yCrBr.at<cv::Vec3b>(row, col)[2];
                    cb -= 109;
                    cr -= 152;
                    int x1 = (819*cr-614*cb)/32 + 51;
                    int y1 = (819*cr+614*cb)/32 + 77;
                    x1 = x1*41/1024;
                    y1 = y1*73/1024;
                    int value = x1*x1+y1*y1;
                    if (y < 100) {
                        mask.at<uchar>(row, col) = (value < 1300)?255:0;
                    }else{
                        mask.at<uchar>(row, col) = (value < 1450)?255:0;
                    }
                }
            }
        }
    }else{
        for (int row = 0; row < input.size().height; row ++) {
            for (int col = 0; col < input.size().width; col ++) {
                int y = yCrBr.at<cv::Vec3b>(row, col)[0];
                int cr = yCrBr.at<cv::Vec3b>(row, col)[1];
                int cb = yCrBr.at<cv::Vec3b>(row, col)[2];
                cb -= 109;
                cr -= 152;
                int x1 = (819*cr-614*cb)/32 + 51;
                int y1 = (819*cr+614*cb)/32 + 77;
                x1 = x1*41/1024;
                y1 = y1*73/1024;
                int value = x1*x1+y1*y1;
                if (y < 100) {
                    mask.at<uchar>(row, col) = (value < 850)?255:0;
                }else{
                    mask.at<uchar>(row, col) = (value < 1000)?255:0;
                }
            }
        }
        
    }
//    mask=findContours(mask);
    return mask;
    
}


double gaussian2D(double x, double y, double sigma){
//    return (1/(2*PI*sigma*sigma))*exp(-(x*x+y*y)/(2*sigma*sigma));
    return (exp(-(x-y)*(x-y)/(2*sigma*sigma)));
}
cv::Mat beeps(cv::Mat input, double lamda){
    double sigma = 0.25;
    cv::Mat output(input.size(), CV_64FC3);
    cv::Mat output2(input.size(), CV_64FC3);
    cv::Mat src(input.size(), CV_64FC3);
    input.convertTo(src, CV_64FC3, 1/255.0);
    cv::Mat progressive(input.size(), CV_64FC3);
    input.convertTo(progressive, CV_64FC3, 1/255.0);
    cv::Mat regressive(input.size(), CV_64FC3);
    input.convertTo(regressive, CV_64FC3, 1/255.0);
    double gaussiank, gaussiank_r;
    
    for (int row = 0; row < input.size().height; row ++) {
        for (int col= 1; col < input.size().width; col ++) {
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[0], progressive.at<cv::Vec3d>(row, col - 1)[0], sigma);
            progressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[0];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[1], progressive.at<cv::Vec3d>(row, col - 1)[1], sigma);
            progressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[1];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[2], progressive.at<cv::Vec3d>(row, col - 1)[2], sigma);
            progressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = input.size().width - 2; col >= 0; col --) {
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[0], regressive.at<cv::Vec3d>(row, col+1)[0], sigma);
            regressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[0];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[1], regressive.at<cv::Vec3d>(row, col+1)[1], sigma);
            regressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[1];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[2], regressive.at<cv::Vec3d>(row, col+1)[2], sigma);
            regressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output2.at<cv::Vec3d>(row, col)[0] = (progressive.at<cv::Vec3d>(row, col)[0]-(1-lamda)*src.at<cv::Vec3d>(row, col)[0]+regressive.at<cv::Vec3d>(row, col)[0])/(1+lamda);
            output2.at<cv::Vec3d>(row, col)[1] = (progressive.at<cv::Vec3d>(row, col)[1]-(1-lamda)*src.at<cv::Vec3d>(row, col)[1]+regressive.at<cv::Vec3d>(row, col)[1])/(1+lamda);
            output2.at<cv::Vec3d>(row, col)[2] = (progressive.at<cv::Vec3d>(row, col)[2]-(1-lamda)*src.at<cv::Vec3d>(row, col)[2]+regressive.at<cv::Vec3d>(row, col)[2])/(1+lamda);
        }
    }
    src = output2;
    for (int col = 0; col < input.size().width; col ++) {
        for (int row = 1; row < input.size().height; row ++) {
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[0], progressive.at<cv::Vec3d>(row - 1, col)[0], sigma);
            progressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[0];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[1], progressive.at<cv::Vec3d>(row - 1, col)[1], sigma);
            progressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[1];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[2], progressive.at<cv::Vec3d>(row - 1, col)[2], sigma);
            progressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[2];
        }
    }
    for (int col = 0; col < input.size().width; col ++) {
        for (int row = input.size().height - 2; row >= 0; row --) {
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[0], regressive.at<cv::Vec3d>(row + 1, col)[0], sigma);
            regressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[0];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[1], regressive.at<cv::Vec3d>(row + 1, col)[1], sigma);
            regressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[1];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[2], regressive.at<cv::Vec3d>(row + 1, col)[2], sigma);
            regressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output2.at<cv::Vec3d>(row, col)[0] = (progressive.at<cv::Vec3d>(row, col)[0]-(1-lamda)*src.at<cv::Vec3d>(row, col)[0]+regressive.at<cv::Vec3d>(row, col)[0])/(1+lamda);
            output2.at<cv::Vec3d>(row, col)[1] = (progressive.at<cv::Vec3d>(row, col)[1]-(1-lamda)*src.at<cv::Vec3d>(row, col)[1]+regressive.at<cv::Vec3d>(row, col)[1])/(1+lamda);
            output2.at<cv::Vec3d>(row, col)[2] = (progressive.at<cv::Vec3d>(row, col)[2]-(1-lamda)*src.at<cv::Vec3d>(row, col)[2]+regressive.at<cv::Vec3d>(row, col)[2])/(1+lamda);
        }
    }

    
    input.convertTo(src, CV_64FC3, 1/255.0);
    for (int col = 0; col < input.size().width; col ++) {
        for (int row = 1; row < input.size().height; row ++) {
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[0], progressive.at<cv::Vec3d>(row - 1, col)[0], sigma);
            progressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[0];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[1], progressive.at<cv::Vec3d>(row - 1, col)[1], sigma);
            progressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[1];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[2], progressive.at<cv::Vec3d>(row - 1, col)[2], sigma);
            progressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank*lamda*progressive.at<cv::Vec3d>(row - 1, col)[2];
        }
    }
    for (int col = 0; col < input.size().width; col ++) {
        for (int row = input.size().height - 2; row >= 0; row --) {
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[0], regressive.at<cv::Vec3d>(row + 1, col)[0], sigma);
            regressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[0];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[1], regressive.at<cv::Vec3d>(row + 1, col)[1], sigma);
            regressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[1];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[2], regressive.at<cv::Vec3d>(row + 1, col)[2], sigma);
            regressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row + 1, col)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output.at<cv::Vec3d>(row, col)[0] = (progressive.at<cv::Vec3d>(row, col)[0]-(1-lamda)*src.at<cv::Vec3d>(row, col)[0]+regressive.at<cv::Vec3d>(row, col)[0])/(1+lamda);
            output.at<cv::Vec3d>(row, col)[1] = (progressive.at<cv::Vec3d>(row, col)[1]-(1-lamda)*src.at<cv::Vec3d>(row, col)[1]+regressive.at<cv::Vec3d>(row, col)[1])/(1+lamda);
            output.at<cv::Vec3d>(row, col)[2] = (progressive.at<cv::Vec3d>(row, col)[2]-(1-lamda)*src.at<cv::Vec3d>(row, col)[2]+regressive.at<cv::Vec3d>(row, col)[2])/(1+lamda);
        }
    }
    
    src = output;
    for (int row = 0; row < input.size().height; row ++) {
        for (int col= 1; col < input.size().width; col ++) {
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[0], progressive.at<cv::Vec3d>(row, col - 1)[0], sigma);
            progressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[0];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[1], progressive.at<cv::Vec3d>(row, col - 1)[1], sigma);
            progressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[1];
            gaussiank = gaussian2D(src.at<cv::Vec3d>(row, col)[2], progressive.at<cv::Vec3d>(row, col - 1)[2], sigma);
            progressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank*lamda*progressive.at<cv::Vec3d>(row, col - 1)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = input.size().width - 2; col >= 0; col --) {
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[0], regressive.at<cv::Vec3d>(row, col+1)[0], sigma);
            regressive.at<cv::Vec3d>(row, col)[0] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[0]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[0];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[1], regressive.at<cv::Vec3d>(row, col+1)[1], sigma);
            regressive.at<cv::Vec3d>(row, col)[1] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[1]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[1];
            gaussiank_r = gaussian2D(src.at<cv::Vec3d>(row, col)[2], regressive.at<cv::Vec3d>(row, col+1)[2], sigma);
            regressive.at<cv::Vec3d>(row, col)[2] = (1 - gaussiank_r*lamda)*src.at<cv::Vec3d>(row, col)[2]+gaussiank_r*lamda*regressive.at<cv::Vec3d>(row, col+1)[2];
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output.at<cv::Vec3d>(row, col)[0] = (progressive.at<cv::Vec3d>(row, col)[0]-(1-lamda)*src.at<cv::Vec3d>(row, col)[0]+regressive.at<cv::Vec3d>(row, col)[0])/(1+lamda);
            output.at<cv::Vec3d>(row, col)[1] = (progressive.at<cv::Vec3d>(row, col)[1]-(1-lamda)*src.at<cv::Vec3d>(row, col)[1]+regressive.at<cv::Vec3d>(row, col)[1])/(1+lamda);
            output.at<cv::Vec3d>(row, col)[2] = (progressive.at<cv::Vec3d>(row, col)[2]-(1-lamda)*src.at<cv::Vec3d>(row, col)[2]+regressive.at<cv::Vec3d>(row, col)[2])/(1+lamda);
        }
    }
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output.at<cv::Vec3d>(row, col)[0] = (output.at<cv::Vec3d>(row, col)[0] + output2.at<cv::Vec3d>(row, col)[0])/2;
            output.at<cv::Vec3d>(row, col)[1] = (output.at<cv::Vec3d>(row, col)[1] + output2.at<cv::Vec3d>(row, col)[1])/2;
            output.at<cv::Vec3d>(row, col)[2] = (output.at<cv::Vec3d>(row, col)[2] + output2.at<cv::Vec3d>(row, col)[2])/2;
        }
    }
    output.convertTo(output, CV_8UC3, 255);
    return output;
}
cv::Mat whiten(cv::Mat input, double beta){
    cv::Mat output;
    input.convertTo(output, CV_64FC3, 1/255.0);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output.at<cv::Vec3d>(row, col)[0] = log(output.at<cv::Vec3d>(row, col)[0]*(beta - 1)+1)/log(beta);
            output.at<cv::Vec3d>(row, col)[1] = log(output.at<cv::Vec3d>(row, col)[1]*(beta - 1)+1)/log(beta);
            output.at<cv::Vec3d>(row, col)[2] = log(output.at<cv::Vec3d>(row, col)[2]*(beta - 1)+1)/log(beta);
        }
    }
    output.convertTo(output, CV_8UC3, 255);
    return output;
}
cv::Mat beautify(cv::Mat input, int skinretouchpower, int whitenpower){
    double sp = skinretouchpower / 200.0 + 0.45;
    double wp = whitenpower/50.0 +1.1;
    cv::Mat output, mask, output1;
    
    output = whiten(input, wp);
    mask = skinDetector(output);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            if (mask.at<uchar>(row, col) == 0) {
                output.at<cv::Vec3b>(row, col)[0] = input.at<cv::Vec3b>(row, col)[0];
                output.at<cv::Vec3b>(row, col)[1] = input.at<cv::Vec3b>(row, col)[1];
                output.at<cv::Vec3b>(row, col)[2] = input.at<cv::Vec3b>(row, col)[2];
            }
        }
    }
    
    output.copyTo(output1);
    
    mask = faceDetector(output);
    output = beeps(output, sp);
    
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            if (mask.at<uchar>(row, col) == 0) {
                output.at<cv::Vec3b>(row, col)[0] = output1.at<cv::Vec3b>(row, col)[0];
                output.at<cv::Vec3b>(row, col)[1] = output1.at<cv::Vec3b>(row, col)[1];
                output.at<cv::Vec3b>(row, col)[2] = output1.at<cv::Vec3b>(row, col)[2];
            }
        }
    }

//    cv::cvtColor(mask, output, CV_GRAY2BGR);
    return output;

}
cv::Mat lomoMemory(cv::Mat input){
    cv::Mat output;
    input.copyTo(output);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            output.at<cv::Vec3b>(row, col)[0] = 10*sqrt(output.at<cv::Vec3b>(row, col)[0]);
        }
    }    return output;
}
cv::Mat lomoPurple(cv::Mat input){
    cv::Mat output;
    input.copyTo(output);
    for (int row = 0; row < input.size().height; row ++) {
        for (int col = 0; col < input.size().width; col ++) {
            double R = output.at<cv::Vec3b>(row, col)[2]/255.0;
            R  = 1 / (1 + exp(-(R - 0.5)/0.15));
            R *= 255;
            output.at<cv::Vec3b>(row, col)[2] = (uchar)R;
        }
    }    return output;
}

cv::Mat lomo(cv::Mat input, int type){
    cv::Mat output;
    switch (type) {
        case 1:
            output = lomoMemory(input);
            break;
        case 2:
            output = lomoPurple(input);
            break;
        default:
            input.copyTo(output);
            break;
    }
    output = vignette(output, 0.7);
    return output;
}