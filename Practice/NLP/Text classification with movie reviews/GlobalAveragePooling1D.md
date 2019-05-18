#  <font color=#7CFC00 face="黑体">第2章 GlobalAveragePooling1D层</font>
```python
keras.layers.pooling.GlobalAveragePooling1D()
```
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;添加一个 GlobalAveragePooling1D 层，它将平均整个序列的词嵌入，将输入的词向量序列相加在求平均，整合成一个向量。
## 2.1 参数解释
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;无参数，不解释。
## 2.2 方法流程解析：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;特征图全局平均一下输出一个值，也就是把*W\*H\*D*的一个张量变成*1\*1\*D*的张量,操作非常简单，在文本分类中只需判断文本属于哪一类，因此可以将所有文本词向量求和平均为一个向量,从而更便于分类。某种程度上，池化也是一种降维操作。经常被用于神经网络的最后，便于获得可以在全连接层工作的形状。

![Alt text](https://img-blog.csdn.net/20180921180920791?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 "darknet-53") 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Darknet-53在平均池化前的张量输出是8x8x1024，对每个8x8的特征图做一个平均池化(取一个平均数)，就可以得到1024个标量了，然后在进入一个1000结点的全连接层，最后通过softmax输出。这就是一个分类网络的主干了。
## 2.3 参考
[参考1](https://zhuanlan.zhihu.com/p/48574887)  
[参考2](https://www.jianshu.com/p/e15bda2e4ebf)

