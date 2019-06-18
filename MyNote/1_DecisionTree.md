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

 

# <div id="#决策树"><center>决策树</ceter></div>
决策树算法<b>过程</b>及**重点**
## <div id="1-基本流程">1 基本流程</div>
>
<br><b>输入：</b>训练集 $D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\};$
<br>属性集 $A=\{a_1,a_2,\dots,a_d\}$
<br><b>过程：</b>函数TreeGrenerate(D,A)  
1. 生成节点node;  
2. **if** D中样本全属于同一个类别C **then**.  
3. &nbsp;&nbsp;&nbsp;将node标记为C类叶节点; *「递归返回情形（1）」*   
4.  &nbsp;&nbsp;&nbsp;**return**.  
4. **end if**  
5. **if** A=$\emptyset$ **or** D中样本在A上取值相同 **then**  
6. &nbsp;&nbsp;&nbsp;将node标记为叶节点，其类别标记为D中样本数最多的类； *「递归返回情形（2）」*   
7. &nbsp;&nbsp;&nbsp;**return**  
7. **end if**  
8. 从A中选择最优划分属性$a_*$;  
9. **for** $a_*$ 的每一个值 $a_*^v$ **do**.  
10. 为node生成一个分支；令 $D_v$ 表示 $D$ 中在 $a_*$ 上取值为 $a_*^v$ 的样本集;  
11. **if** $D_v$ 为空 **then**  
12. &nbsp;&nbsp;&nbsp;将分支节点标记为叶节点，其类别标记为D中样本最多的类；*「递归返回情形（3）」*   
13. &nbsp;&nbsp;&nbsp;**return**  
13. **else**  
14. &nbsp;&nbsp;&nbsp;以TreeGeneration($D_v$, $A$ \ \{$a_*$\})为分支节点  
15. **end if**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>注意:</b>递归返回（2）和（3）都是面临节点无法被划分的情形，**但**（2）是利用当前节点的后验分布进行类别判断，（3）是利用父节点的样本分布作为当前节点的先验分布进行类别判断。
## <div id="2-流程核心划分选择">2 流程核心（划分选择）</div>
选择最优划分属性（**信息增益**，**信息增益率**，**基尼指数**）
### <div id="21-信息增益">2.1 信息增益</div>
#### <div id="概念">2.1.1 概念</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>信息熵</b> $Ent(D)$ : $Ent(D)=-\sum_{k=1}^{|y|}p_k\log_2^{p_k}$.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$Ent(D)$越小，D的纯度越高。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>信息增益</b>$Gain(D,a)$:$Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{D}Ent(D^v)$.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;信息增益越大，则意味着使用属性$a$来划分所获得的“纯度提升”越大，$ID3$决策树学习算法以其为准则划分属性。
#### <div id="212-劣势">2.1.2 劣势</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对可取值数目较多的属性有所偏好，极端情况以编号为属性时，该属性划分并无意义。
### <div id="22-增益率">2.2 增益率</div>
#### <div id="221-概念">2.2.1 概念</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>增益率</b> $Gain\_ratio(D,a)$ :$$Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)},$$
其中
$$IV(a)=-\sum_{v=1}^V\frac{|D^v|}{|D|}\log_2^{\frac{|D^v|}{|D|}}.$$
$C4.5$决策树算法以其为标准划分属性。
#### <div id="222-劣势">2.2.2 劣势</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对取值数目较少的属性有所偏好，不直接用来选划分属性，而是从候选划分属性中找出信息增益高于平均水平的属性，再从中选出增益率最高的。
### <div id="23-基尼指数">2.3 基尼指数</div>
#### <div id="231-概念">2.3.1 概念</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>基尼值</b> $Gini(D)$ :
$$Gini(D)=\sum_{k=1}^y \sum_{k'\not=k}p_kp_{k'}$$
$$=1-\sum_{k=1}^{|y|}p_k^2$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;属性$a$的**基尼指数**定义为:
$$Gini\_index(D,a)=\sum_{v=1}^VGini(D^v).$$
#### <div id="232-使用">2.3.2 使用</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;选择那个使得划分后基尼指数最小的属性为最优划分属性。
## <div id="3-优化过拟合属性过多">3 优化（过拟合+属性过多）</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>剪枝</b>是决策树学习算法对付“过拟合”的主要手段，采用**留出法**。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color=#7CFC00 face="黑体">**联想**</font>$\longrightarrow$<font color=#FF0000face="黑体">算法的优化</font>（Note named Optimise）
### <div id="31-预剪枝">3.1 预剪枝</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随机划分出训练集与验证集，比较划分前后验证集精读，决定是否对该节点划分。存在**欠拟合**的风险。
### <div id="32-后剪枝">3.2 后剪枝</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;先从训练集形成一棵完整决策树，从叶节点开始剪枝，比较剪枝前后验证集精度，决定是否剪枝。训练**时间开销**比较大。
### <div id="33-多变量决策树">3.3 多变量决策树</div>
当**属性过多**时，一个个训练时间消耗过大，可以采用**属性的线性组合**进行划分。
## <div id="4-特殊值连续与缺失值">4 特殊值（连续与缺失值）</div>
### <div id="41-连续值">4.1 连续值</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将连续属性$a$在样本集$D$上出现的$n$个不同取值排序组成集合$\{a^1,a^2,\dots,a^n\}$，选取**中位点**为划分点。
### <div id="42-缺失值">4.2 缺失值</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为每一个样本$x$赋予一个权重$w_x$,并定义
$$
\begin{split}
\rho &= \frac{\sum_{x\in \tilde{D}}w_x}{\sum_{x\in D}w_x} \\
p_k &= \frac{\sum_{x\in \tilde{D_k}}w_x}{\sum_{x\in \tilde{D}}w_x} \\
\tilde{\gamma}_v &= \frac{\sum_{x\in \tilde{D_v}}w_x}{\sum_{x\in \tilde{D}}w_x}
\end{split}
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;信息增益的计算式可推广为：
$$
\begin{split}
Gain(D,a) &= \rho\times Gain(\tilde{D},a)\\
		   &= \rho\times (Ent(\tilde{D})-\sum_{v=1}^V\tilde{\gamma_v}Ent(\tilde{D_v}))\\
Ent\tilde{(D)} &= -\sum_{k=1}^{|y|}\tilde{p_k}\log_2^{\tilde{p_k}}
\end{split}
$$
属性值未知样本划分入所有子节点，对应权重改变为$\tilde{\gamma_v}\cdot w_x$；直观地看，就是让同一个样本以不同概率划入到不同子节点中。
## <div id="5-参考">5.参考</div>
[1]周志华.机器学习[M].清华大学出版社,2016:425.
