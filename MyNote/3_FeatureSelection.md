Table of Contents
=================

   * [第1章 决策树](1_DecisionTree.md#决策树)
      * [1 基本流程](1_DecisionTree.md#1-基本流程)
      * [2 流程核心（划分选择）](1_DecisionTree.md#2-流程核心划分选择)
         * [2.1 信息增益](1_DecisionTree.md#21-信息增益)
            * [2.1.1 概念](1_DecisionTree.md#211-概念)
            * [2.1.2 劣势](1_DecisionTree.md#212-劣势)
         * [2.2 增益率](1_DecisionTree.md#22-增益率)
            * [2.2.1 概念](1_DecisionTree.md#221-概念)
            * [2.2.2 劣势](1_DecisionTree.md#222-劣势)
         * [2.3 基尼指数](1_DecisionTree.md#23-基尼指数)
            * [2.3.1 概念](1_DecisionTree.md#231-概念)
            * [2.3.2 使用](1_DecisionTree.md#232-使用)
      * [3 优化（过拟合 属性过多）](1_DecisionTree.md#3-优化过拟合属性过多)
         * [3.1 预剪枝](1_DecisionTree.md#31-预剪枝)
         * [3.2 后剪枝](1_DecisionTree.md#32-后剪枝)
         * [3.3 多变量决策树](1_DecisionTree.md#33-多变量决策树)
      * [4 特殊值（连续与缺失值）](1_DecisionTree.md#4-特殊值连续与缺失值)
         * [4.1 连续值](1_DecisionTree.md#41-连续值)
         * [4.2 缺失值](1_DecisionTree.md#42-缺失值)
      * [5 参考](1_DecisionTree.md#5-参考)

* [第2章 数据预处理](2_DataPreprocess.md#第2章-数据预处理)
      * [数据预处理分类及方法详解](2_DataPreprocess.md#数据预处理分类及方法详解)
      * [2.1 归一化](2_DataPreprocess.md#21-归一化)
      * [2.2 特征二值化](2_DataPreprocess.md#22-特征二值化)
      * [2.3 独热编码](2_DataPreprocess.md#23-独热编码)
      * [2.4 缺失值计算](2_DataPreprocess.md#24-缺失值计算)
      * [2.5 数据变换](2_DataPreprocess.md#25-数据变换)
      * [2.6 样本不均衡](2_DataPreprocess.md#26-样本不均衡)
      * [2.7 参考](2_DataPreprocess.md#27-参考)  

* [第3章 特征选择](3_FeatureSelection.md#第3章-特征选择)
      * [3.1 特征选择简介](3_FeatureSelection.md#31-特征选择简介)
      * [3.2 过滤式（Filter）](3_FeatureSelection.md#32-过滤式（Filter）)
         * [3.2.1 方差过滤法](3_FeatureSelection.md#321-方差过滤法)
         * [3.2.2 皮尔森相关系数](3_FeatureSelection.md#322-皮尔森相关系数)
         * [3.2.3 互信息和最大信息系数](3_FeatureSelection.md#323-互信息和最大信息系数)
         * [3.2.4 信息增益](3_FeatureSelection.md#324-信息增益)
      * [3.3 包裹式（Wrapper）](3_FeatureSelection.md#33-包裹式（Wrapper）)
           * [3.3.1 递归特征消除](3_FeatureSelection.md#331-递归特征消除)
           * [3.3.2 Las Vegas Wrapper](#332-Las_Vegas_Wrapper)
      * [3.4 嵌入式（Embedding）](3_FeatureSelection.md#34-嵌入式（Embedding）)
      * [3.5 参考](3_FeatureSelection.md#35-参考)

* [第4章 Optimise](4_Optimise.md) 
 * [4.1 优化原理](4_Optimise.md#41-优化原理)
 		* [4.1.1 费马定理](4_Optimise.md#411-费马定理)
 		* [4.1.2 一阶优化](4_Optimise.md#412-一阶优化)
   		* [4.1.3 二阶优化](4_Optimise.md#413-二阶优化)
 * [4.2 常用优化算法](4_Optimise.md#42-常用优化算法)
       * [4.2.1 梯度下降算法](4_Optimise.md#421-梯度下降算法)
       * [4.2.2 梯度下降算法的优化](4_Optimise.md#422-梯度下降算法的优化)
       * [4.2.3 牛顿法](4_Optimise.md#423-牛顿法)  
       * [4.2.4 拟牛顿法](4_Optimise.md#424-拟牛顿法)
 * [4.4 参考](4_Optimise.md#44-参考) 

* [第5章 Questions\_in\_Newton](5_QuestionInNetwon.md)  
 * [5.1 收敛性](5_QuestionInNetwon.md#51-收敛性)  
 * [5.2 正定性](5_QuestionInNetwon.md#52-Positive)
 * [5.3 参考](5_QuestionInNetwon.md#53-参考)  

* [第6章 拟牛顿法详解](6_Quasi_Newton.md#第6章-拟牛顿法详解)
      * [6.1 DFP算法](6_Quasi_Newton.md#61-DFP算法)
      * [6.2 BFGS算法](6_Quasi_Newton.md#62-BFGS算法)
      * [6.3 参考](6_Quasi_Newton.md#63-参考) 



# <div id="第3章-特征选择"><center>第3章 特征选择</center></div>
特征选择的方法详解
-------------------------
## <div id="31-特征选择简介">3.1 特征选择简介</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据预处理完成后，接下来需要从给定的特征集合中筛选出对当前学习任务有用的特征，这个过程称为特征选择（feature selection）。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;特征选择的两个标准：

 + 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
 + 特征与目标的相关性：优先选择与目标相关性高的特征。
   
常见的特征选择方法有三种：过滤法（Filter）、包裹法（Wrapper）、嵌入法（Embedding）。
	
## <div id="32-过滤式（Filter）">3.2 过滤式（Filter）</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**先进行特征选择，再对学习器进行训练，
设定阈值或特征个数，对特征进行“过滤”。**
### <div id="321-方差过滤法">3.2.1 方差过滤法</div>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算每个特征的**方差**，然后根据阈值删除取值小于阈值的特征。使用feature_selection库的VarianceThreshold类来选择特征的代码如下：

``` python
from sklearn.feature_selection import VarianceThreshold

#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)
```
### <div id="322-皮尔森相关系数">3.2.2 皮尔森相关系数</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该方法衡量的是变量之间的线性相关性，结果的取值区间为[-1，1]，-1表示完全的负相关(这个变量下降，那个就会上升)，+1表示完全的正相关，0表示没有线性相关。用feature_selection库的SelectKBest类结合Pearson 相关系数来选择特征的代码如下：

``` python
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```
### <div id="323-互信息和最大信息系数">3.2.3 互信息和最大信息系数</div>  
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

### <div id="324-信息增益">3.2.4 信息增益</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参考我的决策树算法笔记 [DecisionTree](https://blog.csdn.net/lrglgy/article/details/87733853)。

## <div id="33-包裹式（Wrapper）">3.3 包裹式（Wrapper）</div>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**针对特定的学习器选择出最有利于其性能的特征。**
#### <div id="331-递归特征消除">3.3.1 递归特征消除</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;递归训练学习器，每次运行都选出其中最好的特征并应用于下一次训练中。feature_selection库的RFE类来选择特征的代码如下：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```
#### <div id="332-Las_Vegas_Wrapper">3.3.2 Las Vegas Wrapper</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LVW是典型的包裹式特征选择方法，该算法将最终要使用的学习器的性能作为特征子集的评价标准，然后针对特征空间中的不同子集，计算每个子集的预测效果，效果最好的，即作为最终被挑选出来的特征子集。算法流程如下图所示： 

![Alt text](https://img-blog.csdn.net/20170412143312730?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDA4OTQ0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## <div id="34-嵌入式（Embedding）">3.4 嵌入式（Embedding）</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**特征选择过程与学习器的训练过程融为一体。**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;嵌入式的方法有L1正则化和随机森林。

## <div id="35-参考">3.5 参考</div>
[数据预处理与特征选择](https://blog.csdn.net/u010089444/article/details/70053104)  
[机器学习中的特征——特征选择的方法以及注意点](https://blog.csdn.net/google19890102/article/details/40019271)