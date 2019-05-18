//
//  main.cpp
//  Image_Read_Show
//
//  Created by Ruogu Lu on 2019/1/3.
//  Copyright © 2019 Ruogu Lu. All rights reserved.
//

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

int main(){
    /*Version:opencv2;Use:Mat,imread(),nameWindow(),imshow()*/
    Mat img1=imread("/Users/ruogulu/Desktop/mask10.jpeg"); //注意路径名及文件后缀，此处后缀应为.jpeg,但写成了.jpg,不会报错但也不会显示
    if(!img1.data) {cout<<"error1";return -1;}
    namedWindow("image1");
    imshow("image1",img1);
    //cvWaitKey(30);
    waitKey(0);
    
    /*Version:opencv; Use:IplImage,cvLoadImage(),cvNamedWindow(),cvShowImage()*/
    IplImage* img2=cvLoadImage("/Users/ruogulu/Desktop/mask10.jpeg");
    if(!img2)  {cout<<"error2";return -1;}
    cvNamedWindow("image2");
    cvShowImage("image2",img2);
    cvWaitKey(0);
    cvReleaseImage(&img2);
    cvDestroyWindow("image2");
    
    return 0;
}
