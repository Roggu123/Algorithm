# <center>TensorFlow的三种文件</center>
## 1.1 前言
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在使用tensorflow时，`.pb`、`.pbtxt`、`.proto`这三种文件比较常见，本文将对这几种文件的区别及联系进行简单介绍,由于知识水平有限，目前只能提供一种感性认知。  

## 1.2 定义比较  
+ **.pbtxt**  
tensorflow配置文件，定义了网络结构，存储的是结构化数据，定义神经网络中每个节点要储存的信息，具体实例如下：
  
```python
child_info {
    name: 'Jerry';
    age: 3;
    sex: 0;
}

child_info {
    name: 'Kerry';
    age: 4;
    sex: 0;
}
......
```
  
+ **.proto**   
`protobuf`是谷歌开源的数据存储语言，其文件以`.proto`为后缀，可以帮助读取结构化语言，具体实例如下：  

```python
syntax = "proto3";

package proto_test;

message Child {
    string name = 1;
    int32 age = 2;
    int32 sex = 3;        
}

message ChildInfo {
    repeated Child child_info = 1;        
}
```  

+ **.pb**  
为二进制文件，是tensorflow的预训练模型，包含protobuf序列化后的数据，计算图，所有运算符细节，不包含Variable节点，但将其转化为常量，具体实例如下：  

```txt
0a20 0a04 6461 7461 120b 506c 6163 6568
6f6c 6465 722a 0b0a 0564 7479 7065 1202
3001 0a58 0a23 666c 6174 7465 6e5f 392f
5265 7368 6170 652f 7368 6170 652f 5f31
315f 5f63 665f 5f31 3112 0543 6f6e 7374
2a1d 0a05 7661 6c75 6512 1442 1208 0312
0412 0208 0222 0801 0000 00ff ffff ff2a
0b0a 0564 7479 7065 1202 3003 0a58 0a23
666c 6174 7465 6e5f 382f 5265 7368 6170
652f 7368 6170 652f 5f31 305f 5f63 665f
5f31 3012 0543 6f6e 7374 2a1d 0a05 7661
6c75 6512 1442 1208 0312 0412 0208 0222
0801 0000 00ff ffff ff2a 0b0a 0564 7479
7065 1202 3003 0a56 0a21 666c 6174 7465
6e5f 372f 5265 7368 6170 652f 7368 6170
```  


  
## 1.3 作用比较  
上述三种文件的使用流程如下：  
  
![Alt text](Process1.png)  

1. 通过终端命令`$ ***/protos/*.proto -- python_out=.`可将`protos`文件夹下的`.proto`文件转化为`.py`文件；  
2. `.pbtxt`文件是结构化数据，使用传统的数据读取方式且要快速方便的解析出其中的信息，并不容易。而使用protobuf则非常便捷，所以借助`.py`文件读取`.pbtxt`文件；  
3. 将`.pb`文件和读取后的`.pbtxt`文件结合起来进行相应的运算；  
  
## 1.4 类比学习  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面略带专业性的描述似乎还是有点难懂，与现实生活场景斗胆类比一下似乎更便于理解一些。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我将`.proto`，`.pbtxt`，`.pb`三个文件与工厂进行一个类比，运行模型的过程就像工厂加工产品。`.proto`文件就像工厂设计师，它可以看懂工厂的简略设计图`.pbtxt`。工厂的简略设计图`.pbtxt`只定义了工厂有哪些厂房，以及厂房的功能。而`.pb`文件则定义了产品在各厂房间的流转顺序，以及各厂房如何实现自己的功能。输入的变量就像原料，经过各工厂间的流转，输出加工完成的产品。

## 2.5 Reference
[一文看懂Protocol Buffer](https://zhuanlan.zhihu.com/p/36554982)  
[TensorFlow 简单 pb (pbtxt) 文件读写](https://www.jianshu.com/p/3de6ffc490a9)  
[TensorFlow 到底有几种模型格式？](https://cloud.tencent.com/developer/article/1009979)