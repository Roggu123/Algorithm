# age_gender
通过识别人脸判断其年龄性别
## 1.1 目录

1. **age_deploy.prototxt** ------------------------- 年龄模型定义  
*模型定义，定义了每层的结构信息*
2. **age_net.caffemodel** ------------------------- 年龄模型权重文件  
*定义了模型各节点的权重* 
3. **gender_deploy.protoxt** --------------------- 性别模型定义
4. **gender_net.caffemodel** -------------------- 年龄模型权重文件  
5. **opencv_face_detector_uint8.pb**------------ 人脸检测Tensorflow模型  
*打开文件后发现它似乎也是定义了模型权重* 
6. **opencv_face_detector.pbtxt**--------- 人脸检测模型定义 
7. **Photo**------------------------------------------- 需识别的图片  
8. **FileofTensorflow.md**---------------- Tensorflow中三种文件辨析

## 1.2 参考  
报错积累：  

1. [OpenCV调用TesorFlow预训练模型](https://www.tinymind.cn/articles/615)  


python+opencv有关讲解：  

1. [Deep learning: How OpenCV’s blobFromImage works](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)  
2. [Drawing Functions](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.rectangle)

