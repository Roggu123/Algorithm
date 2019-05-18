# -*- coding: utf-8 -*-
# @Author   作者 : BogeyDa！！
# @FileName 文件名 : FaceDetect_Dnn.py
# @Software 创建文件的IDE : PyCharm
# @Blog     博客地址 : https://blog.csdn.net/lrglgy
# @Time     创建时间 : 2019-05-15 11:23
#
# @reference参考 : 博客：[face-detection-opencv-dlib-and-deep-learning-c-python](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)
#                 代码：[face_detection_opencv_dnn.py](https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_dnn.py)
# @Log      代码说明：利用DNN网络识别人脸，他人Demo

## First Backups on 18th May
## Just practice the Demo
#from __future__ import division
#import cv2
#import time
#import sys
#
#def detectFaceOpenCVDnn(net, frame):
#    frameOpencvDnn = frame.copy()
#    frameHeight = frameOpencvDnn.shape[0]
#    frameWidth = frameOpencvDnn.shape[1]
#    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
#
#    net.setInput(blob)
#    detections = net.forward()
#    bboxes = []
#    for i in range(detections.shape[2]):
#        confidence = detections[0, 0, i, 2]
#        if confidence > conf_threshold:
#            x1 = int(detections[0, 0, i, 3] * frameWidth)
#            y1 = int(detections[0, 0, i, 4] * frameHeight)
#            x2 = int(detections[0, 0, i, 5] * frameWidth)
#            y2 = int(detections[0, 0, i, 6] * frameHeight)
#            bboxes.append([x1, y1, x2, y2])
#            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
#    return frameOpencvDnn, bboxes
#
#if __name__ == "__main__" :
#
#    # OpenCV DNN supports 2 networks.
#    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
#    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
#    # 选择进行人脸识别的网络：1.Tensorflow  2.Caffe
#    print("Please choose the net you prefer:")
#    print("1.Tensorflow        2.Caffe")
#    DNN = input("Input:\n")
#    if DNN == '2':
#        modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
#        configFile = "model/deploy.prototxt"
#        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
#    else:
#        modelFile = "model/opencv_face_detector_uint8.pb"
#        configFile = "model/opencv_face_detector.pbtxt"
#        # net = cv2.dnn.readNetFromTensorflow(configFile, modelFile)
#        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
#
#    conf_threshold = 0.7   #设定神经网络阈值
#
#    # 确定进行人脸识别的文件来源source, source=0表示摄像头，否则为指定路径的文件
#    source = 0
#    if len(sys.argv) > 1:
#        source = sys.argv[1]
#
#    # 定义人脸识别中涉及的参数及操作，
#    # cap--被识别文件，frame--视频帧，frame_count--视频帧数，tt_opencvDnn--进行人脸识别的总时间
#    # vid_writer--保存视频
#    cap = cv2.VideoCapture(source)
#    hasFrame, frame = cap.read()
#    vid_writer = cv2.VideoWriter('Results/Dnn/output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
#    frame_count = 0
#    tt_opencvDnn = 0
#
#    # 通过循环读取视频每一帧并进行人脸识别
#    while(1):
#        hasFrame, frame = cap.read()
#        if not hasFrame:
#            break
#        frame_count += 1
#
#        t = time.time()
#        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
#        tt_opencvDnn += time.time() - t
#        fpsOpencvDnn = frame_count / tt_opencvDnn
#        label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
#        cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
#
#        cv2.imshow("Face Detection Comparison", outOpencvDnn)
#
#        vid_writer.write(outOpencvDnn)
#        if frame_count == 1:
#            tt_opencvDnn = 0
#
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    cv2.destroyAllWindows()
#    vid_writer.release()

# Second Backups on 13th May
# Add codes which can compute the accuracy of detector
def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    flag = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            flag = 1
    return frameOpencvDnn, bboxes, flag

if __name__ == "__main__" :
    
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    # 选择进行人脸识别的网络：1.Tensorflow  2.Caffe
    print("Please choose the net you prefer:")
    print("1.Tensorflow        2.Caffe")
    DNN = input("Input:\n")
    if DNN == '2':
        modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "model/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "model/opencv_face_detector_uint8.pb"
        configFile = "model/opencv_face_detector.pbtxt"
        # net = cv2.dnn.readNetFromTensorflow(configFile, modelFile)
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
    conf_threshold = 0.7   #设定神经网络阈值
    
    # 确定进行人脸识别的文件来源source, source=0表示摄像头，否则为指定路径的文件
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    # 定义人脸识别中涉及的参数及操作，
    # cap--被识别文件，frame--视频帧，frame_count--视频帧数，tt_opencvDnn--进行人脸识别的总时间
    # vid_writer--保存视频
    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('Results/Dnn/output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
    frame_count = 0
    tt_opencvDnn = 0
    reg = 0
    
    # 通过循环读取视频每一帧并进行人脸识别
    while(1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1
        
        t = time.time()
        outOpencvDnn, bboxes, flag = detectFaceOpenCVDnn(net,frame)
        if(flag==1):
            reg += 1
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        accuracy = reg/frame_count
        label1 = "OpenCV DNN ; FPS : {:.2f}：".format(fpsOpencvDnn)
        label2 = "OpenCV DNN ; Acc : {:.2f}".format(accuracy)
        cv2.putText(outOpencvDnn, label1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(outOpencvDnn, label2, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        
        cv2.imshow("Face Detection Comparison", outOpencvDnn)
        
        vid_writer.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print("The overall accuracy is:", accuracy)
cv2.destroyAllWindows()
vid_writer.release()
