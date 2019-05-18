//  main.cpp
//  FaceDetect
//
//  Created by Ruogu Lu on 2019/1/1.
//  Copyright © 2019 Ruogu Lu. All rights reserved.

#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <sstream>
#include <iomanip>
#include <chrono>

using namespace cv;
using namespace std;

CascadeClassifier faceCascade;

void save_img(Mat img)
{
    imshow("CamerFaces", img);
    //          获取系统当前时间并转换为字符串
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%F %T");
    std::string str = ss.str();

    //        保存图片至本地
    string saveName = "/Users/ruogulu/Desktop/Results/"+str+".png";
    imwrite(saveName,img);

    waitKey();
}

Mat ReadImg(Mat img)
{
    VideoCapture capture(0);
    if (capture.isOpened())
    {
        capture >> img;
    }
    else
    {
        img=imread("/Users/ruogulu/Desktop/mask10.jpeg");
    }
    return img;
}

void DrawFaces(Mat img, Mat imgGray, vector<Rect> faces)
{
    while(1)
    {
        if(img.empty())
        {
            continue;
        }

        if(img.channels() == 3)
        {
            cvtColor(img, imgGray, CV_RGB2GRAY);
        }
        else
        {
            imgGray = img;
        }

        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0)); //检测人脸

        if(faces.size()>0)
        {
            for(int i =0; i<faces.size(); i++)
            {
                rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
            }
            save_img(img);
        }

        if(faces.size()<=0)
        {
            cout << "There is no face" << endl;
            putText(img,"Can not detect the face",Point(500,400),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),4,8);//在图片上写文字
            save_img(img);
        }

        if(waitKey(1) > 0)        // delay ms 等待按键退出
        {
            break;
        }

    }
}



int main()
{
//    faceCascade.load("haarcascade_frontalface_default.xml");
    if( !faceCascade.load("/usr/local/Cellar/opencv/3.4.1_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    ){
        printf("--(!)Error loading face cascade\n");
        return -1;
    };

    Mat img, imgGray;
    vector<Rect> faces;
    VideoCapture capture(0);

    img = ReadImg(img);

    DrawFaces(img, imgGray,faces);

    return 0;
}




