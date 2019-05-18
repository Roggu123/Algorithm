//
//  main.cpp
//  VideoCapture
//
//  Created by Ruogu Lu on 2019/4/9.
//  Copyright © 2019 Ruogu Lu. All rights reserved.
//  Reference: 代码参考 [C/C++ OpenCV读取视频与调用摄像头](https://blog.csdn.net/qq78442761/article/details/54173104)
//             配置参考 [xcode用c++调用opencv打开摄像头。Info.plist缺少NSCameraUsageDescription的值](https://blog.csdn.net/qqq2018/article/details/86992355)

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    //读取视频或摄像头
    VideoCapture capture(0);

    while (true)
    {
        Mat frame;
        capture >> frame;
        imshow("get video", frame);
        waitKey(30);    //延时30
    }
    return 0;
}

//int main()
//{
//    //读取视频或摄像头
//    VideoCapture capture("/Users/ruogulu/Desktop/Video/720P_1500K_177918601.mp4");
//
//    while (true)
//    {
//        Mat frame;
//        capture >> frame;
//        imshow("读取视频", frame);
//        waitKey(30);    //延时30
//    }
//    return 0;
//}
