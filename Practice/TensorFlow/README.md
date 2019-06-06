# Tensorflow目录
学习使用 Tensorflow
## <div id="11-Tensorflow基础概念">1.1 Tensorflow基础概念</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tensorflow是一个用于数值计算的强大开源软件库，非常适合大型机器学习，是世界上最受欢迎的开源机器学习框架。Tensorflow能把代码转换成操作图。而真正运行的正是这种图。将模型以图的形式展现出来后，工作者可以推迟或者删除不必要的操作。甚至重用部分结果，可以减少代码量。  

## 1.2 tensorflow常用包及函数介绍
### 1.2.1 数据相关  
1. `tf.constant=(value, dtype=None, shape=None, name="Const", verify_shape=False)`  
**作用**：  
将数据转换为常量，即不可更改的常量。  
**参数解释**：  
value: 符合tf中定义的数据类型的常数值或者常数列表;  
dtype：数据类型，可选;  
shape：常量的形状，可选;  
name：常量的名字，可选;  
verify_shape：常量的形状是否可以被更改，默认不可更改;  
2. `tf.matmul(x,y)`  
**作用**：  
表示矩阵x与y相乘,注意与`tf.multiply()`区分  
**参数解释**：  
注意矩阵x与y的类型相同且尺寸符合乘法规则。 


## <div id="12-参考">1.2 参考</div>  
[1] 大腿君.[谷歌大神带你十分钟看懂TensorFlow](https://zhuanlan.zhihu.com/p/32225723)  
[2] Geron.[机器学习实战](https://book.douban.com/subject/30317874/)
## <div id="13-文件目录">1.3 文件目录</div>
1. **[Docker](Docker)**-------在docker中使用tensorflow
2. **[Anaconda](Anaconda)**-------在Anaconda中使用tensorflow

```mermaid
graph TD;
  A-->B;
  A-->C;
  B-->D;
  C-->D;
  ```
