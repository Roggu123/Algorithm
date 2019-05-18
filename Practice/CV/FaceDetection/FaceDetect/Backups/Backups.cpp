//
//  Backups.cpp
//  FaceDetect
//
//  Created by Ruogu Lu on 2019/4/12.
//  Copyright © 2019 Ruogu Lu. All rights reserved.
//

////
////FIRST BACKUPS ON 12TH April, PM 5:26
////
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

//int main()
//{
//    faceCascade.load("haarcascade_frontalface_default.xml");
//
//    if( !faceCascade.load("/usr/local/Cellar/opencv/3.4.1_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
//       ){
//        printf("--(!)Error loading face cascade\n");
//        return -1;
//    };
//
//    VideoCapture capture(0);
//    Mat img, imgGray;
//    vector<Rect> faces;
//
//    if (capture.isOpened())
//    {
//        capture >> img;
//    }
//
//    else
//    {
//        img=imread("/Users/ruogulu/Desktop/mask10.jpeg");
//    }
//
//    while(1)
//    {
//        if(img.empty())
//        {
//            continue;
//        }
//
//        if(img.channels() == 3)
//        {
//            cvtColor(img, imgGray, CV_RGB2GRAY);
//        }
//        else
//        {
//            imgGray = img;
//        }
//
//        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0)); //检测人脸
//
//        if(faces.size()>0)
//        {
//            for(int i =0; i<faces.size(); i++)
//            {
//                rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
//            }
//        }
//        if(faces.size()<=0)
//        {
//            cout << "There is no face" << endl;
////            图片中添加说明文字“Can not detect the face”
//            putText(img,"Can not detect the face",Point(500,400),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),4,8);//在图片上写文字
//            imshow("CamerFace", img);
//            waitKey();
//        }
//        imshow("CamerFace", img); //显示
//
//        ////        保存图片至本地
//        //        string saveName = "/Users/ruogulu/Desktop/1.png";
//        //        imwrite(saveName,img);
//
//        if(waitKey(1) > 0)        // delay ms 等待按键退出
//        {
//            break;
//        }
//
//    }
//    //    //    获取系统当前时间并转换为字符串
//    //    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//    //    std::stringstream ss;
//    //    ss << std::put_time(std::localtime(&t), "%F %T");
//    //    std::string str = ss.str();
//    //    //    以下用于实现将识别到的图片保存并自动编号，
//    //    imwrite("/Users/ruogulu/Desktop/Results/"+to_string(i)+".png", img );
//    //    imwrite("/Users/ruogulu/Desktop/Results/"+str+".png", img );
//    //    imwrite("/Users/ruogulu/Desktop/Results/.png", img );
//
//    //    for(size_t i=0; i<faces.size();i++){
//    //        stringstream buffer;
//    //        buffer<<i;
//    //        string saveName = "/Users/ruogulu/Desktop/"+ buffer.str() + ".png";
//
//    //        Rect roi = faces[i];
//
//    return 0;
//}

////
////  SECOND BACKUPS ON 12TH April, PM 5:30
////  Saved the picture locally
////
//
////  将摄像头拍到的图片保存至本地
//void save_img(Mat img)
//{
////  在保存图片前显示图片
//    imshow("CamerFaces", img);
//
////    保存图片至本地
//    //    获取系统当前时间并转换为字符串，将作为保存图片的名字
//    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//    std::stringstream ss;
//    ss << std::put_time(std::localtime(&t), "%F %T");
//    std::string str = ss.str();
//    //    保存图片至本地
//    string saveName = "/Users/ruogulu/Desktop/Results/"+str+".png";
//    imwrite(saveName,img);
//    //    等待按键，将循环暂停在这里，否则将会不断循环，保存过多重复图片
//    waitKey();
//}
//
//int main()
//{
////    加载xml文档，是一种格式规范，作为识别人脸的参考标准
//    faceCascade.load("haarcascade_frontalface_default.xml");
////    如果加载不成功，报错“Error loading face cascade”
//    if( !faceCascade.load("/usr/local/Cellar/opencv/3.4.1_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
//       ){
//        printf("--(!)Error loading face cascade\n");
//        return -1;
//    };
//
////    定义数据
//    Mat img, imgGray;
//    vector<Rect> faces;
////    打开摄像头
//    VideoCapture capture(0);
//
////    判断摄像头是否打开
////      若打开则将拍摄的图片保存至img
////      否则打开本地图片
//    if (capture.isOpened())
//    {
//        capture >> img;
//    }
//
//    else
//    {
//        img=imread("/Users/ruogulu/Desktop/mask10.jpeg");
//    }
//
////    循环读取图片并识别人脸
//    while(1)
//    {
//        //       图片数据img为空进行下一次循环，再读一次图片
//        if(img.empty())
//        {
//            continue;
//        }
//        //        判断图片是否灰度图，是则存为imgGray,否则通过cvtColor转为灰度并存入imgGray
//        if(img.channels() == 3)
//        {
//            cvtColor(img, imgGray, CV_RGB2GRAY);
//        }
//        else
//        {
//            imgGray = img;
//        }
//
////        识别人脸
//        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0));
//        //        若识别出人脸则框出
//        if(faces.size()>0)
//        {
//            for(int i =0; i<faces.size(); i++)
//            {
//                rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
//            }
//            save_img(img);
//        }
//        //        未识别出人脸则在图片标注“Can not detect the face”
//        if(faces.size()<=0)
//        {
//            cout << "There is no face" << endl;
//            putText(img,"Can not detect the face",Point(500,400),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),4,8);//在图片上写文字
//            save_img(img);
//        }
//
////         delay ms 等待按键退出循环
//        if(waitKey(1) > 0)
//        {
//            break;
//        }
//
//    }
//    return 0;
//}

////
////  SECOND BACKUPS ON 16TH April, PM 21:22
////  Modulizing Functions
////
//
////  保存图片至本地
//void save_img(Mat img)
//{
//    imshow("CamerFaces", img);
//    //          获取系统当前时间并转换为字符串
//    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//    std::stringstream ss;
//    ss << std::put_time(std::localtime(&t), "%F %T");
//    std::string str = ss.str();
//
//    //        保存图片至本地
//    string saveName = "/Users/ruogulu/Desktop/Results/"+str+".png";
//    imwrite(saveName,img);
//
//    waitKey();
//}
//
////  读取图片
//Mat ReadImg(Mat img)
//{
//    VideoCapture capture(0);
//    if (capture.isOpened())
//    {
//        capture >> img;
//    }
//    else
//    {
//        img=imread("/Users/ruogulu/Desktop/mask10.jpeg");
//    }
//    return img;
//}
//
////  识别人脸
//void DrawFaces(Mat img, Mat imgGray, vector<Rect> faces)
//{
//    while(1)
//    {
//        if(img.empty())
//        {
//            continue;
//        }
//
//        if(img.channels() == 3)
//        {
//            cvtColor(img, imgGray, CV_RGB2GRAY);
//        }
//        else
//        {
//            imgGray = img;
//        }
//
//        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0)); //检测人脸
//
//        if(faces.size()>0)
//        {
//            for(int i =0; i<faces.size(); i++)
//            {
//                rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
//            }
//            save_img(img);
//        }
//
//        if(faces.size()<=0)
//        {
//            cout << "There is no face" << endl;
//            putText(img,"Can not detect the face",Point(500,400),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),4,8);//在图片上写文字
//            save_img(img);
//        }
//
//        if(waitKey(1) > 0)        // delay ms 等待按键退出
//        {
//            break;
//        }
//
//    }
//}
//
////  主函数
//int main()
//{
//    //    faceCascade.load("haarcascade_frontalface_default.xml");
//    if( !faceCascade.load("/usr/local/Cellar/opencv/3.4.1_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
//       ){
//        printf("--(!)Error loading face cascade\n");
//        return -1;
//    };
//
//    Mat img, imgGray;
//    vector<Rect> faces;
//    VideoCapture capture(0);
//
//    img = ReadImg(img);
//
//    DrawFaces(img, imgGray,faces);
//
//    return 0;
//}

//
//  THIRD BACKUPS ON 24TH April, PM 16:54
//  Add Functions about age and gender, this is a copy not original, it can not run due to errors of mdel
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
using namespace cv;
using namespace cv::dnn;
using namespace std;

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);
    
    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);
    
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");
    
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    vector<vector<int>> bboxes;
    
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        
        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }
    
    return make_tuple(frameOpenCVDNN, bboxes);
}

int main(int argc, char** argv)
{
    string faceProto = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/opencv_face_detector.pbtxt";
    string faceModel = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/opencv_face_detector_uint8.pb";
    
    string ageProto = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/age_deploy.prototxt";
    string ageModel = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/age_net.caffemodel";
    string genderProto = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/gender_deploy.prototxt";
    string genderModel = "/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/gender_net.caffemodel";
    
    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);
    
    vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
        "(38-43)", "(48-53)", "(60-100)"};
    
    vector<string> genderList = {"Male", "Female"};
    
    // Load Network
    //Net ageNet = readNetFrom(ageModel, ageProto);
    //Net genderNet = readNetFromCaffe(genderModel, genderProto);
    //Net faceNet = readNetFromCaffe(faceModel, faceProto);
    Net ageNet = dnn::readNetFromCaffe(ageProto, ageModel);
    Net genderNet = readNetFromCaffe(genderProto, genderModel);
    //Net faceNet = readNetFromCaffe(faceProto,faceModel);
    //Net faceNet = readNetFromTensorflow(faceProto, faceModel);
    //Net faceNet = readNetFromTensorflow(faceModel, faceProto);
    //Net faceNet = readNetFromTensorflow('/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/opencv_face_detector_uint8.pb','/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/opencv_face_detector.pbtxt');
    Net faceNet = readNetFromTensorflow( 'opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt');
    
    VideoCapture cap;
    if (argc > 1)
        cap.open(argv[1]);
    else
        cap.open(0);
    int padding = 20;
    while(waitKey(1) < 0) {
        // read frame
        Mat frame;
        cap.read(frame);
        if (frame.empty())
        {
            waitKey();
            break;
        }
        
        vector<vector<int>> bboxes;
        Mat frameFace;
        tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);
        
        if(bboxes.size() == 0) {
            cout << "No face detected, checking next frame." << endl;
            continue;
        }
        for (auto it = begin(bboxes); it != end(bboxes); ++it) {
            Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
            Mat face = frame(rec); // take the ROI of box on the frame
            
            Mat blob;
            blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
            genderNet.setInput(blob);
            // string gender_preds;
            vector<float> genderPreds = genderNet.forward();
            // printing gender here
            // find max element index
            // distance function does the argmax() work in C++
            int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
            string gender = genderList[max_index_gender];
            cout << "Gender: " << gender << endl;
            
            /* // Uncomment if you want to iterate through the gender_preds vector
             for(auto it=begin(gender_preds); it != end(gender_preds); ++it) {
             cout << *it << endl;
             }
             */
            
            ageNet.setInput(blob);
            vector<float> agePreds = ageNet.forward();
            /* // uncomment below code if you want to iterate through the age_preds
             * vector
             cout << "PRINTING AGE_PREDS" << endl;
             for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
             cout << *it << endl;
             }
             */
            
            // finding maximum indicd in the age_preds vector
            int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
            string age = ageList[max_indice_age];
            cout << "Age: " << age << endl;
            string label = gender + ", " + age; // label
            cv::putText(frameFace, label, Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
            imshow("Frame", frameFace);
            imwrite("out.jpg",frameFace);
        }
        
    }
}




