# Pycharm使用教程
## 1 常见问题（初级）
一. 使用Pycharm建立运行一个python程序  

1. No interpreter (解释器丢失)
	* please specify a different SDK name  
	[pycharm 报错：pycharm please specify a different SDK name](https://blog.csdn.net/lancegentry/article/details/79381047)  
2. Select run/debug configuration (选择run/debug的配置信息)
	* Edit Configuration  
	[图三](3.png)
	* Create a new configuration  
	[图四](4.png)
	* Script path <font color=#FF0000 face="黑体">**(Script path作用是什么?)**</font>  
	[图五](5.png)
3. Test and Success (测试运行，试着打印一句话)  
	[图六](6.png)

二. Script Path的作用是什么？  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Script即为脚本，类似于演戏时用到的脚本，script 其实就是一系列指令——演员看了指令就知道自己该表演什么，说什么台词；计算机看了指令就知道自己该做什么事情。所以 script 其实就是短小的、用来让计算机自动化完成一系列工作的程序，这类程序可以用文本编辑器修改，不需要编译，通常是解释运行的。而Python是解释性语言，也就是脚本，所以这里的```Script Path```（脚本路径）就是Python文件的路径。

[如何用通俗易懂的语言解释脚本（script）是什么？](https://www.zhihu.com/question/19901542)  
[将pycharm的运行配置脚本路径修改为当前选中文件](https://www.zoulei.net/2015/03/18/change-pycharm-run-configurations-script-path-to-current-file/) 
 

三.解释器作用？  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;解释器是一种计算机程序，能够把高端编程语言一行一行解释运行。解释器像是一位“中间人”，每次运行程序时都要先转成另一种语言再作运行，因此解释器的程序运行速度比较缓慢。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;解释器的好处是它消除了编译整个程序的负担，程序可以拆分成多个部分来模块化，但这会让运行时的效率打了折扣。相对地，编译器已一次将所有源代码翻译成另一种语言，如机器代码，运行时便无需再依赖编译器或额外的程序，故而其运行速度比较快。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用解释器来运行程序会比直接运行编译过的机器代码来得慢，但是相对的这个解释的行为会比编译再运行来得快。这在程序开发的雏型化阶段和只是撰写试验性的代码时尤其来得重要，因为这个“编辑-解释-调试”的循环通常比“编辑-编译-运行-调试”的循环来得省时许多。  
参考：[你们要的 PyCharm 快速上手指南](https://zhuanlan.zhihu.com/p/26066151)

## 2 流程核心（划分选择）
选择最优划分属性（**信息增益**，**信息增益率**，**基尼指数**）
### 2.1 信息增益
#### 2.1.1 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>信息熵</b> $Ent(D)$ : $Ent(D)=-\sum_{k=1}^{|y|}p_k\log_2^{p_k}$.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$Ent(D)$越小，D的纯度越高。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>信息增益</b>$Gain(D,a)$:$Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{D}Ent(D^v)$.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;信息增益越大，则意味着使用属性$a$来划分所获得的“纯度提升”越大，$ID3$决策树学习算法以其为准则划分属性。
#### 2.1.2 劣势
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对可取值数目较多的属性有所偏好，极端情况以编号为属性时，该属性划分并无意义。
### 2.2 增益率
#### 2.2.1 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>增益率</b> $Gain\_ratio(D,a)$ :$$Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)},$$
其中
$$IV(a)=-\sum_{v=1}^V\frac{|D^v|}{|D|}\log_2^{\frac{|D^v|}{|D|}}.$$
$C4.5$决策树算法以其为标准划分属性。
#### 2.2.2 劣势
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对取值数目较少的属性有所偏好，不直接用来选划分属性，而是从候选划分属性中找出信息增益高于平均水平的属性，再从中选出增益率最高的。
### 2.3 基尼指数
#### 2.3.1 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>基尼值</b> $Gini(D)$ :
$$Gini(D)=\sum_{k=1}^y \sum_{k'\not=k}p_kp_{k'}$$
$$=1-\sum_{k=1}^{|y|}p_k^2$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;属性$a$的**基尼指数**定义为:
$$Gini\_index(D,a)=\sum_{v=1}^VGini(D^v).$$
#### 2.3.2 使用
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;选择那个使得划分后基尼指数最小的属性为最优划分属性。
## 3 优化（过拟合+属性过多）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>剪枝</b>是决策树学习算法对付“过拟合”的主要手段，采用**留出法**。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color=#7CFC00 face="黑体">**联想**</font>$\longrightarrow$<font color=#FF0000face="黑体">算法的优化</font>（Note named Optimise）
### 3.1 预剪枝
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随机划分出训练集与验证集，比较划分前后验证集精读，决定是否对该节点划分。存在**欠拟合**的风险。
### 3.2 后剪枝
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;先从训练集形成一棵完整决策树，从叶节点开始剪枝，比较剪枝前后验证集精度，决定是否剪枝。训练**时间开销**比较大。
### 3.3 多变量决策树
当**属性过多**时，一个个训练时间消耗过大，可以采用**属性的线性组合**进行划分。
## 4 特殊值（连续与缺失值）
### 4.1 连续值
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将连续属性$a$在样本集$D$上出现的$n$个不同取值排序组成集合$\{a^1,a^2,\dots,a^n\}$，选取**中位点**为划分点。
### 4.2 缺失值
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
## 参考
[1]周志华.机器学习[M].清华大学出版社,2016:425.
