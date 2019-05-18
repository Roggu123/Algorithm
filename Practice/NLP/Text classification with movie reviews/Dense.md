#  <font color=#7CFC00 face="黑体">Dense层(全连接层)</font>
```python
keras.layers.core.Dense(units, activation=None,  
	use_bias=True, kernel_initializer='glorot_uniform',  
	bias_initializer='zeros', kernel_regularizer=None,  
	bias_regularizer=None, activity_regularizer=None,
	kernel_constraint=None, bias_constraint=None)
```
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在神经网络中最常见的网络层就是**全连接层**，在这个层中实现对神经网络里面的神经元的激活。比如：y = g(x′w + b)，其中w 是该层的权重向量，b 是偏置项，g() 是激活函数。如果use_bias 选项设置为False，那么偏置项为0。
## 1 参数解释
+ **units**：大于0的整数，代表该层的输出维度;
+ **activation**：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）;
+ **use_bias**：: 布尔值，是否使用偏置项;
+ **kernel_initializer**: 权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers；  
+ **bias_initializer**偏置向量初始化方法，为预定义初始化方法名的字符串，或用于初始化偏置向量的初始化器。参考initializers;
+ **kernel_regularizer**: 施加在权重上的正则项，为Regularizer对象;
+ **bias_regularizer**：施加在偏置向量上的正则项，为Regularizer对象;
+ **activity_regularizer**：施加在输出上的正则项，为Regularizer对象;
+ **kernel_constraints**：施加在权重上的约束项，为Constraints对象;
+ **bias_constraints**：施加在偏置上的约束项，为Constraints对象;

<font color=#87CEEB face="黑体">约束项与正则项的区别？  
[参考](http://cking0821.club/2018/06/12/Keras/ "With a Title").   
初步感觉正则化项为变量，约束项为常量。</font>

<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**对应代码解析:**

```Python
    model.add(keras.layers.Dense(16,activation=tf.nn.relu)  
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
```

<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;神经网络的最后两层都使用全连接层，第一个全连接层输出16维向量（即有16个隐藏单元），使用relu作为激活函数；第二个全连接层为输出层，由于 IMDB 情感数据集只有正负两个类别，因此全连接层是只有一个神经元的二元分类，使用 sigmoid 激活函数。

[参考](https://keras-cn.readthedocs.io/en/latest/layers/core_layer/ "With a Title"). 
## 2 方法流程解析：
[参考](https://blog.csdn.net/LK274857347/article/details/70246055 "With a Title"). 
[参考](https://www.cnblogs.com/ymjyqsx/p/9451739.html "With a Title"). 
[最佳参考](https://blog.csdn.net/l691899397/article/details/52267166)
### 2.1 全连接层的前向计算：

![Alt text](https://img-blog.csdn.net/20160821142608048 "全连接层") 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上图中连线最密集的2个地方就是全连接层，显而易见，全连接层的参数非常多。在前向计算过程，也就是一个**线性的加权求和**的过程，全连接层的每一个输出都可以看成前一层的每一个结点乘以一个权重系数W，最后加上一个偏置值b得到，即 。如上图中第一个全连接层，输入有50\*4\*4个神经元结点，输出有500个结点(这500个结点代表500个特征)，则一共需要50\*4\*4\*500=400000个权值参数W和500个偏置参数b。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;下图为具体介绍：
<center>
![Alt text](https://img-blog.csdn.net/20160821142639705 "具体介绍")
</center> 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中，x1、x2、x3为全连接层的输入，a1、a2、a3为输出，有
<center>
![Alt text](https://img-blog.csdn.net/20160821142804582 "具体介绍") 
</center>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;写成矩阵形式如下：
<center>
![Alt text](https://img-blog.csdn.net/20160821142838207)
</center>
#### 2.2 全连接层的反向转播：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于需要对$W$和$b$进行更新，还要逆向传递梯度，所以我们需要计算如下三个偏导数。
#### 2.2.1 对当前层输入(上一层输出)求导：
以第一个全连接层为例，有50\*4\*4个输入结点和500个输出结点：
<center>
![Alt text](https://img-blog.csdn.net/20160821143055227)
</center>

<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若我们已知转递到该层的梯度**$\frac{\partial loss}{\partial a}$**，则我们可以通过链式法则求得loss对x的偏导数。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先需要求得该层的输出$a_i$对输入$x_j$的偏导数
<center>
$\frac{\partial a_i}{\partial x_j}=\sum_j^{800}\frac{ w_{ij}*x_j}{\partial x_j}=w_{ij}$</center>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;再通过链式法则求得loss对x的偏导数

<center>
$\frac{\partial loss}{\partial x_j}=\sum_i^{500} \frac{\partial loss}{\partial a_i}\frac{\partial a_i}{\partial x_j}=\sum_i^{500}\frac{\partial loss}{\partial a_i}w_{ij}$
</center>
#### 2.2.2 对权重系数w求导
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据之前的逆梯度计算公式，
<center>
![Alt text](https://img-blog.csdn.net/20160821142804582 "具体介绍") 
</center>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可得$\frac{\partial a_i}{\partial w_{ij}}=x_j$,所以$\frac{\partial loss}{\partial w_{ij}}=\frac{\partial loss}{\partial a_i}x_j$。
#### 2.2.3 对偏置系数b求导
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据之前的逆梯度计算公式得$\frac{\partial a_i}{\partial b_i}=1$。
所以$\frac{\partial loss}{\partial b_i}=\frac{\partial loss}{\partial a_i}$。
即即loss对偏置系数的偏导数等于对上一层输出的偏导数。