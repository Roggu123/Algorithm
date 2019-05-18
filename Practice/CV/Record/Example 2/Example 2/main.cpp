//
//  main.cpp
//
//  Created by Ruogu Lu on 2018/12/30.
//  Copyright © 2018 Ruogu Lu. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  First Program
//  A simple OpenCV program that loads an image from disk and display it on the Screen
//  从磁盘载入图片并在屏幕上显示

//#include <iostream>
//#include <opencv2/opencv.hpp>       //include file for every supported OpenCV function
/*
 头文件作用
 C/C++编译采用的是分离编译模式（若干个源文件共同实现，而每个源文件单独编译生成目标文件，最后将所有目标文件连接起来形成单一的可执行文件的过程）
 （1）加强类型检查，提高类型安全性。
 （2）减少公用代码的重复书写，提高编程效率。
 （3）提供保密和代码重用的手段。
 opencv.hpp实现opencv中的不同库文件间的公用数据类型一致，公用代码重用，保密
 */

//
//Example2-1
//

//int main(int argc, const char * argv[]) {
//    /*
//     main函数中的参数 argc 表示导入参数的数目， argv数组存储真正参数（索引从1开始，main函数本身占据索引0）
//     */
//    cv:: Mat img = cv::imread(argv[1],-1);  //loads the image
//    /*imread()是一种高级方法，根据文件名字，确定文件格式，然后为图片自动分配内存，该函数可读取多种类型图片包括 BMP，DIB，JPEG， JPE, PNG, PBM, PGM, PPM, SR, RAS, and TIFF.
     
//       Mat 是一种数据结构，执行上诉语句返回数据结构Mat，这是 OpenCV 中最常用的概念，该数据结构可以处理单通道，多通道，整型，浮点型等众多类型图片，当该数据结构（图片）超出使用范围后，会被自动回收
//     */
//    if (img.empty()) return -1;     //checks if an image was in fact read

//   cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);   //open window that contain and display image
//    /* namedWindow() 是由 HighGUI 库提供的函数，显示包含图片的窗口并为之命名（如“Example1”）, 在之后与该窗口交互时将使用该名字，第二个参数定义窗口的属性，1.取默认值0时，窗口将应用同样的尺寸而不考虑图片大小，图片根据窗口大小调整尺寸，
//                          2.取值cv::WINDOWS_AUTORISE时，窗口尺寸则根据图片尺寸改变大小，或由用户自己调整。
//     */
//   cv::imshow("Example1", img);    //display image in cv::Mat structure in an existing window
     /* imshow() 在窗口中显示图片，如果有窗口，它会刷新窗口，若无窗口，则会新建一个窗口
      */
//   cv::waitKey(0);
     /* waitKey() 要求程序停止，并等待一个键盘输入，参数为正时，会等待相应的毫秒数然后继续，参数为负或零时，则一致等待直到有键盘敲击
      */
//   cv::destroyWindow("Example1");      //destroyWindow() 表示销毁窗口，并且将与之有关的内存占用释放，小程序可以省略此步，大型复杂程序需确保处理窗口避免内存泄漏。
//   //std::cout << "Hello, World!\n";
//    return 0;
//}

/*
 **************小结*****************
 
 OpenCV 中的函数属于命名空间 cv, （1）在调用 OpenCV 函数时必须要告知编译器具体的命名空间 （如 cv::function），
                             （2）在主函数前声明： using namespace cv; 告知编译器下面使用的函数属于该命名空间
 在上述代码中引入头文件 opencv.hpp, 为提高编译速度可以仅包含入需要用到的头文件
 
 **********************************
*/

//
//Example 2-2
//

/*
 
#include <iostream>
#include <opencv2/highgui/highgui.hpp>      //UPDATE
 using namespace cv;                        //UPDATE
int main(int argc, const char * argv[]) {
    Mat img = imread(argv[1],-1);
    if (img.empty()) return -1;
    namedWindow("Example1", cv::WINDOW_AUTOSIZE);
    imshow("Example1", img);
    waitKey(0);
    destroyWindow("Example1");
    return 0;
}
 
*/

/*
 ****************小结****************
 
 在使用函数时添加命名空间是更好的编程习惯
 
************************************
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Second Program
//
//Example2-3
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
int main(int argc, char** argv){
    cv::namedWindow("Example3",cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    /* VideoCapture 是对象，被实例化，这个对象可以打开或关闭多种类型的视频文件
     */
    cap.open(std::string(argv[1]));
    cv::Mat frame;
    /* open() 给予对象一个字符串（其中包括视频的路径和名字， s对象中会包括被读入的视频的所用信息即使是状态信息）
     * frame 中包含视频的每一个画面
     */
    for(;;){
        cap>>frame;
        if(frame.empty()) break;
        cv::imshow("Example3", frame);
        if(cv::waitKey(33)>=0) break;
    }
    /* 该循环中视频文件被从对象 cap 一幅画面一幅画面地读取
     * 等待33ms,如果未敲击键盘则继续循环，若敲击键盘则立即退出，结束循环
     */
    return 0;
}




