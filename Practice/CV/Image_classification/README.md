# 参考：
1.Tensorflow教程实例：https://www.tensorflow.org/tutorials/keras/basic_classification
## 代码重点：
<br>1.代码需在TensorFlow中运行，而非传统集成开发环境（如pycharm等）；
<br>2.个人认为本次代码中的重点为神经网络模型的构建[model=keras.Sequential()]与编译[model.compile()], Note 文件会对神经网络模型构建，编译，以及每层的含义作用进行研究，Optimizer文件和loss文件会对编译模型中用到的优化器和损失函数进行研究。
## 使用Tensorflow:
一.利用pip安装tensorflow,并在virtualenv中使用tensorflow
<br> 1.使用命令安装pip, virtualenv
sudo easy_install pip
pip3 install - -upgrade virtualenv
<br>2.安装完毕，对环境进行激活（在每个新shell中使用tensorflow）
如果Virtualenv没有被激活，则命令提示行将不显示tensorflow对应的目录
输入如下命令，对环境进行激活：
<br>UserMacBook-Pro:~ ****&#36; cd tensorflow（tensorflow是我的安装目录，据个人情况有所不同）
<br>UserMacBook-Pro:tensorflow ****&#36; source ./bin/activate  
###if using bash, sh,ksh, or zsh
UserMacBook-Pro:tensorflow ****&#36; source ./bin/activate.csh
#### if using csh or tcsh
输入如上命令后，命令行将转变为如下，表示tensorflow环境被激活：
(tensorflow) UserMacBook-Pro:tensorflow ****&#36; 
（最关键的是有括号及内部目录名如（tensorflow），其它据个人电脑情况不同而不同）
使用完tensorflow后可以输入如下命令，停止激活环境：
(tensorflow) UserMacBook-Pro:tensorflow ****&#36; deactivate

二.利用Docker安装Tensorflow
<br>1.从应用商店下载并安装Docker
<br>2.运行Docker
<br>3.打开终端，开始在Docker中使用Tensorflow， 
<br>（在终端中运行TensorFlow）输入：&#36; docker run -it tensorflow/tensorflow bash
<br>（在Jupyternotebook中运行）输入：&#36; docker run -it -p 8888:8888 tensorflow/tensorflow
