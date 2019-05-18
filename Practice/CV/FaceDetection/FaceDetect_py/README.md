# FaceDetection_py
学习人脸识别过程中的**Python**实战项目积累
## Catalog

1. **FaceDetect_Haar**--------------- 利用Haar特征实现人脸识别
2. **FaceDetect_Dnn**----------- 利用DNN网络实现人脸是识别
3. **model**----------------------- 人脸识别过程中用到的模型文件  
4. **Face**------------------ 人脸识别中用到的本地测试文件
4. **Results**----------------------- 代码运行后的结果 
5. **Record**---------------------- 博客中使用的过程记录截图 
6. **Backups**-------------------- 代码的备份文件

## Reference   
1. Code: [face_detection_opencv_haar.py](https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_haar.py
)  
2. Blog：[face-detection-opencv-dlib-and-deep-learning-c-python](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/) 

## Error
`FAILED: fs.is_open(). Can't open "models/opencv_face_detector_uint8.pb" in function 'ReadProtoFromBinaryFile'`  
经过检查发现是自己的文件路径写错了；
            