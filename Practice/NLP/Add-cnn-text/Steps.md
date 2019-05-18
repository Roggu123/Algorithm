# <center>第2章 Steps(*步骤*)</center>
**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

This is the flow to run code is this project.（*这是执行项目代码的过程记录。*）

## 2.1 Python文件作用  
+  <font color=#0000ff>**data\_helpers.py**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;辅助作用，在其它文件中被执行。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;提供有关**<font color=#FF0000>数据的处理的方法</font>**，加载数据（*load\_data\_and\_labels()*）、清理数据（*clean\_str()*），定义批迭代器（每次训练时使用不同的批次）（*batch\_iter()*）。  
使用pad_sequences()将多个句子截断或补齐为相同长度， 
+ <font color=#0000ff>**text_cnn.py**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;辅助作用，在其它文件中被执行。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>搭建了一个最基础的CNN模型</font>**，有输入层（*input layer*），卷积层（*convolutional layer*），最大池化层（*max-pooling layer*）和最后输出的softmax layer。
+ <font color=#0000ff>**train.py**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;真正被执行的文件，导入data_helpers.py和text_cnn.py文件。   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>训练模型及真正处理数据</font>**，其中包含：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**数据准备**`preprocess()`，即数据加载、字典生成、打乱数据集、拆分训练集和测试集；  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**训练**`train()`，即定义图（Graph）、定义会话(Session)、<font color=#0000ff>在会话中建立并编译cnn模型</font>、对训练进行汇总（定义结果存储路径）；  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**定义单独训练步**`train_step()`，在一批数据上训练模型，并相应地更新参数。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**定义单独评估步**`dev_step()`，即评估模型在一批数据上的表现。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**定义训练循环**`batches`，一批一批的在数据上迭代，调用train_step函数，然后评估并存储检查点。
+ <font color=#0000ff>**eval.py**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;真正被执行的文件，导入 data\_helpers.py 和 text\_cnn.py 文件。   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>对模型进行评价</font>**，其中包含：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**数据准备**，即数据加载、映射到词典；  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**获取并执行模型**，即获取数据、定义评价循环、获取保存预测值、获取打印模型精确度；  


## 2.2 执行步骤  
### 2.2.1 加载数据（*load data*）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用的数据集为 Movie Review data from Rotten Tomatoes，该数据集包含10,662个示例评论句子，正面和负面评论各占一半。数据集的单词数量约为20k。请注意，由于此数据集非常小，我们使用强大的模型可能会过拟合。此外，该数据集没有官方的训练/测试拆分，因此我们只使用10％的数据作为开发集，用于模型参数调优。数据集存储在目录`data/rt-polaritydata/`中。Python文件**`./train.py`**中的<b>*preprocess()*</b>函数真正执行该步骤。
  
### 2.2.2 数据预处理（*data pre-processing*）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对数据集进行清洗，如将部分字母或符号进行替换；建立字典，如统一句子长度，将句子用向量来表示；打乱数据集；将数据集拆分为训练集和测试集。Python文件`./data_helper.py`实现数据清洗，`./train.py`实现字典建立，打乱数据集，数据集拆分。由于`train.py`包含`data_helpers.py`,所以运行**`train.py`**可以直接执行该步骤。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据预处理的详细知识总结可参考笔记 [第二章 数据预处理](https://blog.csdn.net/lrglgy/article/details/87882746)。
[^_^]: # ((哈哈，传github时使用这个)../../../MyNote/DataPreprocess.md)

### 2.2.3 建立及编译模型
text_cnn.py中提供底层支持，**`train.py`**真正执行该步骤。

### 2.2.4 训练模型

### 2.2.5 评价模型 

## 2.3 函数解析
### 2.3.1 data_helpers.py
1. **clean\_str(string)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;替换非字母，非数字，非标点符号的字符为空格；替换缩写字符；替换标点符号等。  
<font color=FF0000>除了正则表达式可用于字符匹配，大小写转换外，还有什么办法可进行上述操作？</font>   
[关于python 的re.sub用法](https://blog.csdn.net/lovemianmian/article/details/8867613)  
[正则表达式中各种字符的含义](https://www.cnblogs.com/afarmer/archive/2011/08/29/2158860.html)  
[常用正则表达式](https://www.jianshu.com/p/0cb001fe3572)  
2. **load_data_and_labels(positive_data_file, negative_data_file)**  
3. **batch_iter(data, batch_size, num_epochs, shuffle=True)**

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)原始论文
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)原始论文导读
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)原始代码解析
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) 理解卷积神经网络 
- [用Tensorflow实现CNN文本分类(详细解释及TextCNN代码解释)](http://www.voidcn.com/article/p-sjhkchtl-bmr.html)对代码中textCNN的理解  
- [CNN 实现文本分类](https://github.com/fengxqinx/TextCNN)  主要讲解卷积神经网络在文本分类中的实现(代码其参考)
- [文本分类(下)-卷积神经网络(CNN)在文本分类上的应用](https://juejin.im/post/5b584ae1e51d4517580dfd56)主要讲解卷积神经网络对文本分类原理及实现
