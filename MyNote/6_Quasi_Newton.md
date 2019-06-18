Table of Contents
=================

* **[第1章 决策树](1_DecisionTree.md#决策树)**
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
* **[第2章 数据预处理](2_DataPreprocess.md#第2章-数据预处理)**
      * [数据预处理分类及方法详解](2_DataPreprocess.md#数据预处理分类及方法详解)
      * [2.1 归一化](2_DataPreprocess.md#21-归一化)
      * [2.2 特征二值化](2_DataPreprocess.md#22-特征二值化)
      * [2.3 独热编码](2_DataPreprocess.md#23-独热编码)
      * [2.4 缺失值计算](2_DataPreprocess.md#24-缺失值计算)
      * [2.5 数据变换](2_DataPreprocess.md#25-数据变换)
      * [2.6 样本不均衡](2_DataPreprocess.md#26-样本不均衡)
      * [2.7 参考](2_DataPreprocess.md#27-参考)  

* **[第3章 特征选择](3_FeatureSelection.md#第3章-特征选择)**
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

* **[第4章 Optimise](4_Optimise.md)** 
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

* **[第5章 Questions\_in\_Newton](5_QuestionInNetwon.md)**  
 * [5.1 收敛性](5_QuestionInNetwon.md#51-收敛性)  
 * [5.2 正定性](5_QuestionInNetwon.md#52-Positive)
 * [5.3 参考](5_QuestionInNetwon.md#53-参考)  

* **[第6章 拟牛顿法详解](6_Quasi_Newton.md#第6章-拟牛顿法详解)**
      * [6.1 DFP算法](6_Quasi_Newton.md#61-DFP算法)
      * [6.2 BFGS算法](6_Quasi_Newton.md#62-BFGS算法)
      * [6.3 参考](6_Quasi_Newton.md#63-参考) 

# <center>第6章 拟牛顿法详解</center>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;拟牛顿算法可以利用多种方式对矩阵$G_k$进行更新($\Delta G_k$)，常用算法有DFP，BFGS，Broyden，SR1等。本章将主要介绍DFP与BFGS算法。
拟牛顿算法进行梯度更新的基本公式如下：
$$
\begin{gather}
G_{k+1}y_k=s_k \tag{6.1} \\  
G_{k+1}=G_k+\Delta G_k\tag{6.2} \\
\Delta G_k=\alpha uu^T+\beta vv^T \tag{6.3}
\end{gather}
$$  
其中$y_k=g_{k+1}-g_k$，$s_k=x_{k+1}-x_k$，且$\Delta G_k$进行上述假设的原因是矩阵加法必须满足两个矩阵行数列数相等。由式子$s_k=G_{k+1} y_k$易知$G_k$一定是一个n阶方阵（$s_k$和$y_k$分别是n维行向量和列向量），通过上面的假设可以很容易保证$G_k$是n阶矩阵并且还是对称矩阵，在计算求解方面更加方便。 
## <div id="61-DFP算法">6.1 DFP算法</div>      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在DFP算法中，直接将假设条件（6.3）代入基本公式（6.2）中进行求解，  
$$
\begin{gather}
(G_k+\Delta G_k)y_k=s_k \tag{6.4}\\
(G_k+\alpha uu^T+\beta vv^T)y_k=s_k \tag{6.5}\\
G_k y_k+(\alpha u^T y_k)u+(\beta v^T y_k)v=s_k \tag{6.6}
\end{gather}
$$
易知 $\alpha u^T y_k$ 和 $\beta v^T y_k$ 为常数，不妨假设 $\alpha u^T y_k=1$且 $\beta v^T y_k=-1$，进而可得：  
$$
G_k y_k+u-v=s_k \tag{6.7}
$$
再假设$u=s_k$且$v=G_k y_k$，进一步求得：
$$
\begin{gather}
\alpha =\frac{1}{s_k^T y_k} \tag{6.8}\\
\beta = \frac{-1}{y_k^T G_k y_k} \tag{6.9}
\end{gather}
$$  
最后可求出$\Delta G_k$：  
$$
\Delta G_k = \frac{s_k s_k^T}{s_k^T y_k}-\frac{G_k y_k y_k^T G_k}{y_k^T G_k y_k} \tag{6.10}
$$

>**算法6.1（DFP算法）**  
>输入：目标函数$f(x)$，梯度$g(x)=\nabla f(x)$，精度要求$\epsilon$；  
>输出：$f(x)$的极小值点$x^*$.  
>(1) 取初始点 $x_k$，取$G_0$为正定对称矩阵，置 $k=0$；  
>(2) 计算 $g_k=g(x_k)$，若 $||g_k||<\epsilon$，停止计算，返回近似解 $x^*=x_k$，否则转（3）；  
>(3) 计算 $p_k=-G_k g_k$：  
>(4) 一维搜索，求 $\lambda_k$ 使得：
> $f(x_k+\lambda_k p_k)=\min \limits_{\lambda>=0} f(x_k+\lambda p_k)$  
>(5) 置$x_{k+1}=x_k+p_k$；  
>(6) 计算 $g_{k+1}=g(x_{k+1})$，若 $||g_{k+1}||<\epsilon$，停止计算，得近似解 $x^*=x_{k+1}$，否则按式（6.10）和（6.2）算出$G_{k+1}$；  
>(7) 置$k=k+1$，返回(3)。

## <div id="62-BFGS算法">6.2 BFGS算法</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在BFGS算法中，将基本公式（6.1）稍稍作了改动如下：
$$
y_k=G_{k+1}s_k \tag{6.11}
$$
继续将假设条件（6.3）代入公式（6.11）得：
$$
\begin{align}
u &= G_k s_k \tag{6.12}\\
v &= y_k \tag{6.13}\\
\alpha &=\frac{1}{y_k^T s_k} \tag{6.14}\\
\beta &= \frac{-1}{s_k^T G_k s_k} \tag{6.15}
\end{align}
$$
最后求得，  
$$
\Delta G_k = \frac{y_k y_k^T}{y_k^T s_k}-\frac{G_k s_k s_k^T G_k}{s_k^T G_k s_k} \tag{6.16}
$$

>**算法6.2（BFGP算法）**  
>输入：目标函数$f(x)$，梯度$g(x)=\nabla f(x)$，精度要求$\epsilon$；  
>输出：$f(x)$的极小值点$x^*$.  
>(1) 取初始点 $x_k$，取$G_0$为正定对称矩阵，置 $k=0$；  
>(2) 计算 $g_k=g(x_k)$，若 $||g_k||<\epsilon$，停止计算，返回近似解 $x^*=x_k$，否则转（3）；  
>(3) 计算 $G_k p_k=-g_k$，得到 $p_k$；  
>(4) 一维搜索，求 $\lambda_k$ 使得：
> $f(x_k+\lambda_k p_k)=\min \limits_{\lambda>=0} f(x_k+\lambda p_k)$  
>(5) 置$x_{k+1}=x_k+p_k$；  
>(6) 计算 $g_{k+1}=g(x_{k+1})$，若 $||g_{k+1}||<\epsilon$，停止计算，得近似解 $x^*=x_{k+1}$，否则按式（6.16）和（6.2）算出$G_{k+1}$；  
>(7) 置$k=k+1$，返回(3)。
   
## <div id='63-参考'>6.3 参考</a>
[1] 李航.[统计学习方法](http://www.dgt-factory.com/uploads/2018/07/0725/统计学习方法.pdf)   
 



