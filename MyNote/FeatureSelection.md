# <center>第3章 特征选择</center>
特征选择的方法详解
-------------------------
## 3.1 特征选择简介
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据预处理完成后，接下来需要从给定的特征集合中筛选出对当前学习任务有用的特征，这个过程称为特征选择（feature selection）。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;特征选择的两个标准：

 + 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
 + 特征与目标的相关性：优先选择与目标相关性高的特征。
   
常见的特征选择方法有三种：过滤法（Filter）、包裹法（Wrapper）、嵌入法（Embedding）。
	
## 3.2 过滤式（Filter）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**先进行特征选择，再对学习器进行训练，
设定阈值或特征个数，对特征进行“过滤”。**
### 3.2.1 方差过滤法  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算每个特征的**方差**，然后根据阈值删除取值小于阈值的特征。使用feature_selection库的VarianceThreshold类来选择特征的代码如下：

``` python
from sklearn.feature_selection import VarianceThreshold

#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)
```
### 3.2.2 皮尔森相关系数
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该方法衡量的是变量之间的线性相关性，结果的取值区间为[-1，1]，-1表示完全的负相关(这个变量下降，那个就会上升)，+1表示完全的正相关，0表示没有线性相关。用feature_selection库的SelectKBest类结合Pearson 相关系数来选择特征的代码如下：

``` python
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```
### 3.2.3 互信息和最大信息系数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;互信息（Mutual information）用于评价离散特征对离散目标变量的相关性，互信息计算公式如下：  
$$
I(X;Y)=\sum_{x \in X}\sum_{y \in Y}p(x,y)\log \frac{p(x,y)}{p(x)p(y)}.
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;互信息选择特征有如下缺点：
  
+ 它不属于度量方式，也没有办法归一化，在不同数据集上的结果无法做比较；  
+ 对于连续变量的计算不是很方便（X和Y都是集合，x，y都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最大信息系数（Maximal Information Coefficient， MIC）解决了这两个问题。minepy提供了MIC功能，代码如下：
  
``` python
from minepy import MINE

m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
printm.mic()
```

### 3.2.4 信息增益
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参考我的决策树算法笔记 [DecisionTree](https://blog.csdn.net/lrglgy/article/details/87733853)。

## 3.3 包裹式（Wrapper）  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**针对特定的学习器选择出最有利于其性能的特征。**
#### 3.3.1 递归特征消除
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;递归训练学习器，每次运行都选出其中最好的特征并应用于下一次训练中。feature_selection库的RFE类来选择特征的代码如下：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```
#### 3.3.2 Las Vegas Wrapper
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LVW是典型的包裹式特征选择方法，该算法将最终要使用的学习器的性能作为特征子集的评价标准，然后针对特征空间中的不同子集，计算每个子集的预测效果，效果最好的，即作为最终被挑选出来的特征子集。算法流程如下图所示： 

![Alt text](https://img-blog.csdn.net/20170412143312730?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDA4OTQ0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 3.4 嵌入式（Embedding）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**特征选择过程与学习器的训练过程融为一体。**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;嵌入式的方法有L1正则化和随机森林。

## 3.5 参考
[数据预处理与特征选择](https://blog.csdn.net/u010089444/article/details/70053104)  
[机器学习中的特征——特征选择的方法以及注意点](https://blog.csdn.net/google19890102/article/details/40019271)