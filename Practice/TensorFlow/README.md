# Tensorflow目录
学习使用 Tensorflow
## <div id="11-Tensorflow基础概念">1.1 Tensorflow基础概念</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tensorflow是一个用于数值计算的强大开源软件库，非常适合大型机器学习，是世界上最受欢迎的开源机器学习框架。Tensorflow能把代码转换成操作图。而真正运行的正是这种图。将模型以图的形式展现出来后，工作者可以推迟或者删除不必要的操作。甚至重用部分结果，可以减少代码量。  

## 1.2 tensorflow常用包及函数介绍
### 1.2.1 计算操作有关
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与Tensorflow中数据的数学计算操作有关的节点及函数的定义，使用。
   
1. `tf.matmul(x,y)`  
**作用**：  
表示矩阵x与y相乘,注意与`tf.multiply()`区分  
**参数解释**：  
注意矩阵x与y的类型相同且尺寸符合乘法规则。
  
2. `tf.square(x)`  
**作用**  
计算元素x的平方。
  
3. `reduce_mean(input_tensor,axis=None,keep_dims=False,name=None)`  
**作用**  
沿张量(input_tensor)的某一维度求元素的平均值，这类操作也被称作降维。  
**参数解释**  
input_tensor: 被降维的张量;  
axis: axis=none, 求全部元素的平均值；axis=0, 按列降维，求每列     平均值；axis=1，按行降维，求每行平均值;  
keep_dims: 若值为True，可多行输出平均值;  
name: 自定义操作的名称。 

4. `sess.run()`

5. `变量名.eval`  

6. `gradients=tf.gradients(mse,[theta])[0]`  
**作用**  
自动微分，即自动计算梯度。  
**参数解释**  
mse：操作符，计算均方误差；  
theta：参数列表；  
创建一个操作符的列表计算每个变量的梯度。  

### 1.2.2 数据有关
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对tensorflow中各种不同类型数据节点的总结比较。数据类型转换，数据整理，数据获取。

1. `tf.constant=(value, dtype=None, shape=None, name="Const", verify_shape=False)`  
**作用**：  
将数据转换为常量，即不可更改的常量。  
**参数解释**：  
value: 符合tf中定义的数据类型的常数值或者常数列表;  
dtype：数据类型，可选;  
shape：常量的形状，可选;  
name：常量的名字，可选;  
verify_shape：常量的形状是否可以被更改，默认不可更改;

2. `tf.assign`   
为变量赋值。

3. `tf.variable()`  
定义变量节点。  

4. `np.random.seed()`  

5. `np.random.randint()`
### 1.2.3 算法实现及优化  
+ **优化**
 1. `tf.train.GradientDescentOptimizer(学习率)`  
   
 2. `tf.train.MomentumOptimizer(学习率，动量系数)`   


## <div id="12-参考">1.2 参考</div>  
[1] 大腿君.[谷歌大神带你十分钟看懂TensorFlow](https://zhuanlan.zhihu.com/p/32225723)  
[2] Geron.[机器学习实战](https://book.douban.com/subject/30317874/)
## <div id="13-文件目录">1.3 文件目录</div>
1. **[Docker](Docker)**-------在docker中使用tensorflow
2. **[Anaconda](Anaconda)**-------在Anaconda中使用tensorflow
