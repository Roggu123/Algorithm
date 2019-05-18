#  <center>人脸识别（四）</center>
## 3.1 前言  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;续上一篇博客[《人脸识别（三）》](https://blog.csdn.net/lrglgy/article/details/90046125)，感觉只是学习跑一遍他人的Demo太无趣，根据个人兴趣又小小修改了一下，可以保存视频中的一副图片到本地，也可以自主选择是否在本地保存视频。

## 3.2 代码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;代码注释比较详细，不需再多说什么了。有不好的地方欢迎评论交流。
    
````python
# -*- coding: utf-8 -*-
# @Author   作者 : BogeyDa！！
FaceDetect_Haar.py
# @Software 创建文件的IDE : PyCharm
# @Blog     博客地址 : https://blog.csdn.net/lrglgy
# @Time     创建时间 : 2019-05-06 11:42
#
# @reference参考 : 代码：https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_haar.py
#                  博客：https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
# @Log      代码说明：利用Haar特征对面部进行识别，代码成功运行

from __future__ import division
import cv2
import time
import sys

from pip._vendor.distlib.compat import raw_input


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
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

    # 对视频帧数与时间进行计数
    frame_count = 0
    tt_opencvHaar = 0

    # 选择是否保存视频
    print("Do you want to save video?")
    print("1.Yes              2.No")
    str = input("Input:\n")
    if str == '1':
        flag = 1
    else:
        flag = 0

    # 显示摄像头内容并框出人脸
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

        # 修改代码，添加保存为图片功能，且视频保存为可选操作
        # 参考：[python实现拍照，视频保存，录像，剪辑，分帧等操作](https://blog.csdn.net/xiao__run/article/details/77362888)
        if frame_count == 20:
            cv2.imwrite('FaceDetect0.jpg', outOpencvHaar)
        # 存储视频：判断flag是否为1，是则进行视频保存，否则不保存
        if flag == 1:
            vid_writer.write(outOpencvHaar)

        # 当只拍摄一帧时，时间计为0，等待键盘输入，若键盘有输入且为q时退出程序
        if frame_count == 1:
            tt_opencvHaar = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vid_writer.release()
````

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;项目文件目录见[《人脸识别（三）》](https://blog.csdn.net/lrglgy/article/details/90046125)中的图4-1。
  
 ## 3.3 使用
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;项目文件目录见[《人脸识别（三）》](https://blog.csdn.net/lrglgy/article/details/90046125)中的第3节。
 


## 3.3 结果  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当运行程序后，会先提示是否本地保存视频，选择是则会在本地保存接下来的视频及图片，否则只会保存图片，不保存视频。其它结果和[《人脸识别（三）》](https://blog.csdn.net/lrglgy/article/details/90046125)中的一样。  

## 3.4 参考文献
1. [FaceDetectionComparison/face_detection_dlib_hog.py](https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_dlib_hog.py#L48)
2. [face-detection-opencv-dlib-and-deep-learning-c-python](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)
3. [OpenCV学习笔记（四十七）——VideoWriter生成视频流highgui](https://blog.csdn.net/yang_xian521/article/details/7440190)
4. [python opencv cv2.rectangle 參數含義](https://www.twblogs.net/a/5c22417dbd9eee16b3daf922)
5. [python实现拍照，视频保存，录像，剪辑，分帧等操作](https://blog.csdn.net/xiao__run/article/details/77362888)
6. [Python input() 函数](http://www.runoob.com/python/python-func-input.html)
