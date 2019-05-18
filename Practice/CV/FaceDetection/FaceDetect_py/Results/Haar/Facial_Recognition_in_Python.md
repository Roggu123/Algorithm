#  <center>人脸识别（三）</center>
## 3.1 准备工作  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其实和之前用C++实现人脸识别的[准备工作](https://blog.csdn.net/lrglgy/article/details/89317878)差不多，而且还更方便。  
1. 会自动开启本地摄像头，不必添加别的文件；  
2. 如果之前下载安装了opencv3，这里直接导入就好。有时`import cv2`可能会提示`No module named cv2`，这时选择`install package opencv-pyhton`就好；  
3. xml文档还是要导入的，下载到项目文件夹就好；  

## 3.2 代码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;代码注释还是比较详细的，不需再多说什么了。有不好的地方欢迎评论提出。
    
````python
# -*- coding: utf-8 -*-
# @Author   作者 : BogeyDa！！
FaceDetect_Haar.py
# @Software 创建文件的IDE : PyCharm
# @Blog     博客地址 : https://blog.csdn.net/lrglgy
# @Time     创建时间 : 2019-05-06 11:42
#
# @reference参考 : 代码：https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_dlib_hog.py#L48
#                  博客：https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
# @Log      代码说明：利用Haar特征对面部进行识别，代码成功运行

from __future__ import division
import cv2
import time
import sys

def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
    # 修改帧大小为高度300，宽度据情况而变（frameWidth/frameHeight）* inHeight
    # 这里只是对帧大小进行调整，但还未真正落实
    frameOpenCVHaar = frame.copy()
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    # 获取高度和宽度的缩放比例
    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    # 将帧进行缩放，真正落实帧的改变frameOpenCVHaarSmall，并将帧灰度化frameGray，但原始帧frameOpenCVHaar并未改变
    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    # 利用Haar探测器找到人脸并在原始帧上框出来
    faces = faceCascade.detectMultiScale(frameGray)
    bboxes = []
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        bboxes.append(cvRect)
        # 框出人脸，参数依次表示被识别帧、框一角的坐标、框另一角坐标、框的颜色、框宽度、框类型
        # 参考：[python opencv cv2.rectangle 參數含義](https://www.twblogs.net/a/5c22417dbd9eee16b3daf922)
        cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frameHeight / 150)), 4)
    return frameOpenCVHaar, bboxes


# 从命令行读取参数，即获取要进行面部识别的文件
# 加载Haar特征，Harr特征文件可根据自身需求导入不同文件，文件链接：https://github.com/opencv/opencv/tree/master/data/haarcascades
if __name__ == "__main__":
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    # haarcascade_frontalface_default.xml is in models directory.
    faceCascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    # faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # 读取进行识别的文件，source表示文件路径，当为0时打开本地摄像头
    # hasFrame表示是否读取到图片，frame表示读取到的每一帧图片
    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    # 参考：[OpenCV学习笔记（四十七）——VideoWriter生成视频流highgui](https://blog.csdn.net/yang_xian521/article/details/7440190)
    # 使用video_writer类，将图片序列保存为视频文件，参数依次为视频文件名、视频编码格式、帧速率、帧大小
    vid_writer = cv2.VideoWriter('output-haar-{}.avi'.format(str(source).split(".")[0]),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]), False)
    # vid_writer = cv2.VideoWriter('output-haar-test.avi',
    #                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]),
    #                              False)

    # 读取并保存摄像头中的每一帧至frame
    frame_count = 0
    tt_opencvHaar = 0
    while (1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        # 获取摄像头拍摄时的帧速率，而VideoWriter中的帧速率15表示存入视频的帧速率
        # 利用Haar探测器探测人脸并框出
        t = time.time()
        outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
        tt_opencvHaar += time.time() - t
        fpsOpencvHaar = frame_count / tt_opencvHaar

        label = "OpenCV Haar ; FPS : {:.2f}".format(fpsOpencvHaar)
        cv2.putText(outOpencvHaar, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", outOpencvHaar)

        vid_writer.write(outOpencvHaar)
        if frame_count == 1:
            tt_opencvHaar = 0

        key = cv2.waitKey(1) & 0xFF
        # click keyboard 'q' to exit
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vid_writer.release()
````

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;项目文件目录如下图：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190509182744174.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xyZ2xneQ==,size_16,color_FFFFFF,t_70)
 <center>图4-1 目录结构</center>
  
 ## 3.3 使用
+ IDE（Pycharm等）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;直接点击运行按钮，会打开本地摄像头。
+ Terminal(终端)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如下图所示：
![识别本地摄像](https://img-blog.csdnimg.cn/20190509185709308.png) 
<center>图4-2 摄像头</center> 
 
![识别本地视频](https://img-blog.csdnimg.cn/20190509191903735.png)
<center>图4-3 本地视频</center>
 


## 3.3 结果  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当未指定文件时，本地摄像头会自动打开，摄像头影像中的人脸会被自动识别框出，同时左上角会显示当前帧频率，停止运行程序后，人脸识别视频会被保存至本地，并命名为`output-haar-0.avi`。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当指定文件后，会打开指定的本地视频并识别其中的人脸，然后保存视频到本地。  

## 3.4 参考文献
1. [FaceDetectionComparison/face_detection_dlib_hog.py](https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_dlib_hog.py#L48)
2. [face-detection-opencv-dlib-and-deep-learning-c-python](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)
3. [OpenCV学习笔记（四十七）——VideoWriter生成视频流highgui](https://blog.csdn.net/yang_xian521/article/details/7440190)
4. [python opencv cv2.rectangle 參數含義](https://www.twblogs.net/a/5c22417dbd9eee16b3daf922)
