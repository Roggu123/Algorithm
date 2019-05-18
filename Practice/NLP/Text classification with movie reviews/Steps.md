# Steps-步骤
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;搭建影评文本分类神经网络的步骤；
## 1 <font color=#7CFC00 face="黑体">下载数据</font>$\longrightarrow$<font color=#FF4500 face="黑体">load</font>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将数据导入训练集与测试集中，如（train),(test)=***.load_data()。
## 2 <font color=#7CFC00 face="黑体">探索数据</font>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;探索数据，了解数据格式,获取数据集长度，数据类型等。
## 3 <font color=#7CFC00 face="黑体">数据预处理</font>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;修改数据的维度，格式等，使之能够作为神经网络的输入，如本例中将输入数据集转换为一样的长度。建立函数，实现数据可视化或使数据能以更直观形式呈现的函数。
## 4 <font color=#7CFC00 face="黑体">构建模型</font>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;神经网络通过堆叠层创建而成，这需要做出三个方面的主要决策:  

* 在模型中使用**什么样的层**？
* 要在模型中使用**多少个层**？ 
* 要针对每个层使用多少个**隐藏单元**？

<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Keras中，模型其实有两种，一种叫<font color=#FF4500 face="黑体">**Sequential**</font>，称为<font color=#FF4500 face="黑体">**序贯模型**</font>，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，跨层连接统统没有。这种模型编译速度快，操作上也比较简单。第二种模型称为<font color=#FF4500 face="黑体">**Graph**</font>，即<font color=#FF4500 face="黑体">**图模型**</font>，这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢。可以看到，Sequential其实是Graph的一个特殊情况。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;定义好模型后需要设置模型的层，层的个数，层的类型等。  

1. 第一层是 **Embedding** 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。生成的维度为：(batch, sequence, embedding)。详细介绍见 **Embedding.md**。
2. 接下来，一个 GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。这样，模型便能够以尽可能简单的方式处理各种长度的输入。
3. 该长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）
4. 最后一层与单个输出节点密集连接。应用 sigmoid 激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平。（参考链接：http://www.ituring.com.cn/book/tupubarticle/16624）

