# <center><font color=#7CFC00 face="黑体">第2章 Flatten 层</font></center>
```python
keras.layers.core.Flatten()
```
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>Flatten(扁平)层</font>**用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。<font color=#FF0000>（batch在神经网络中指什么？参考：[一些概念](https://www.jianshu.com/p/872b07813bff)）</font>
## 2.1 参数解释
+ **data_format**：字符串，“channels\_first” 或 “channels\_last” 之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tensorflowf”，即Keras使用tensorflow作为后端引擎（处理张量的库），“channels_first”对应原本的“theano”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。这一参数的作用为保存权重的顺序，当数据形式发生改变时。<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**代码例子:**

```Python
    model = Sequential()
    model.add(Conv2D(64, (3, 3),input_shape=(3, 32, 32), padding='same',))
    # now: model.output_shape == (None, 64, 32, 32)
    model.add(Flatten())
    # now: model.output_shape == (None, 65536)
```

该嵌入层仅使用了前两个参数，设置输入数据（input\_dim）为1000个，每个单词用16维向量表示（output\_dim。
## 2.2 方法流程解析：
### 2.2.1 神经网络可视化：
![Alt text](https://img-blog.csdn.net/2018062910431435?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3Byb2dyYW1fZGV2ZWxvcGVy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 "Flatten层可视化")  
输入的数据格式为(3,32,32),个人理解为3行32列32通道的数据（通道个人理解为特征），经过卷积层后变为3行32列64通道的数据，经过Flatten层后，数据变为3$\times$32$\times$64=6144维的数据。

## 2.3 参考文献：
莫言：[”Keras中文文档“] (https://keras-cn.readthedocs.io/en/latest/layers/core_layer/#flatten "With a Title")， https://keras-cn.readthedocs.io/en/latest/layers/ core_layer/#flatten （2019/2/11）

Microstrong0305：[”深度学习中Flatten层的作用“] (https://blog.csdn.net/program_developer/article/details/80853425 "With a Title")， https://blog.csdn.net/program_developer/article/details/80853425 （2019/2/11）


