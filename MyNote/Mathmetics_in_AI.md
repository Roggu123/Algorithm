Table of Contents
=================

* **[算法中数学](#算法中的数学)**
   * [第一章 矩阵和向量](#1-矩阵和向量)  
       * [1.1 基本概念](#11-基本概念) 
           * [1.1.1 矩阵定义](#111-矩阵定义)
           * [1.1.2 向量定义](#112-向量定义)  
           * [1.1.3 矩阵分类](#113-矩阵分类)
           * [1.1.4 矩阵方程](#114-矩阵方程) 
       * [1.2 基本运算](#12-基本运算)
           * [1.2.1 矩阵乘法](#121-矩阵乘法)  
           * [1.2.3 矩阵微积分](#123-矩阵求导)
           * [1.2.4 参数标准方程推导](#124-参数标准方程推导) 
       * [1.3 参考](#13-矩阵参考)
   * [第二章 误差](#2-误差)
       * [2.1 标准差](#21-标准差)
       * [2.2 均方误差](#22-均方误差)
       * [2.3 均方根误差](#23-均方根误差)
       * [2.4 交叉熵](#24-交叉熵)
       * [2,5 平均绝对误差](#25-平均绝对误差)
       * [2.6 参考](#26-误差参考)

   * [第三章 随便记记](#3-随便记记)  
       * [3.1 线性相关](#31-线性相关)  
       * [3.2 线性变换](#32-线性变换)
       * [3.3 行列式与特征向量辨析](#33-行列式与特征向量辨析)
       * [3.4 参数标准方程推导](#34-参数标准方程推导)    
       * [3.8 参考](#33-参考)         
 
#<center><div id='算法中的数学'>算法中的数学</div></center>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在学习算法的过程中发现很多地方都用到了数学知识，特地在此结合一些业界广泛认可的书对算法中用到的主要的数学知识进行总结及进一步的学习研究。
## <div id="1-矩阵和向量">第一章 矩阵和向量</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵是算法当中出现频率很高的词，算法中的很多计算都是矩阵的计算。本章的主要参考资料为DC.Lay编写的《线性代数几其应用》。
### <div id="11-基本概念">1.1 基本概念</div>  
#### <div id="111-矩阵定义">1.1.1 矩阵定义</div>  
**线性方程组角度**：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵是一个包含线性方程组主要信息的紧凑的矩形阵列。若m,n是正整数，一个$m \times n$矩阵是一个有m行n列的数的矩形阵列（注意：行数写在前面）。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把方程组中的每一个变量的系数写在对齐一列中的矩阵称为方程组的系数矩阵。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在系数矩阵右边增加由方程组右边常数组成的一列所得到的矩阵称为方程组的增广矩阵。

#### <div id="112-向量定义">1.1.2 向量定义</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;仅含一列的矩阵称为列向量，或简称向量(Vector)。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所有两个元素的向量集记为$\mathbb{R}^2$，$\mathbb{R}$表示向量中的元素是实数，而2表示每个向量包含两个元素。一般称$\mathbb{R}^2$中的向量是实数的有序对。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一般将列向量$\left[\begin{matrix}2\\3\end{matrix}\right]$表示为 $(2,3)$ 的形式，而将一行矩阵就表示为$\left[\begin{matrix}2&3\end{matrix}\right]$。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个$n\times m$的矩阵可以理解为 $m-1$ 个向量的线性组合。也可以理解为 $n\times(m-1)$ 矩阵与一个向量的乘积。向量方程可以写成等价的形式为$Ax=b$的矩阵方程。
#### <div id="113-矩阵分类">1.1.3 矩阵分类</div>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵可分为对称矩阵，可逆矩阵，分块矩阵， 方阵......  

#### <div id="114-矩阵方程">1.1.4 矩阵方程</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵方程与线性方程组，向量方程的关系密切，且在解决实际问题时它们经常会发生相互转化，因此学习矩阵方程时最好结合线性方程组，向量方程一起考虑。

+ 线性方程组
$$
\begin{aligned}
a_1 x_1+a_2 x_2+a_3 x_3=b_1\\
a_4 x_1+a_5 x_2+a_6 x_3=b_2\\
a_7 x_1+a_8 x_2+a_9 x_3=b_3
\end{aligned}\tag{1.1}
$$    
  1. 定义   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;形如 $a_1 x_1+a_2 x_2+\dots+a_n x_n=b$ 的方程称为线性方程，其中 $b$ 与 $a_1,a_2,\dots,a_n$ 是实数或复数，通常为已知数。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;线性方程组由一个或几个包含相同变量的线性方程组成的。  

  2. 求解方法  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用行化简算法，先将线性方程组变为增广矩阵，然后利用行初等变换将其转化为简化阶梯形，进而求解变量。  

  3. 重要定理  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当线性方程组的增广矩阵的最右列非主元列时，线性方程组有解即线性方程组相容。  

+ 向量方程  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将线性方程组（1.1）中的系数用向量表示为如下形式：  
$$
\mathbf{a_1}x_1+\mathbf{a_2}x_2+\mathbf{a_3}x_3=\mathbf{b}\\
\mathbf{a_1}=(a_1,a_4,a_7)\\
\mathbf{a_2}=(a_2,a_5,a_8)\\
\mathbf{a_3}=(a_3,a_6,a_9)\\
\mathbf{b}=(b_1,b_2,b_3)
$$  
  1. 重要定理  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;仅含一列的矩阵称为列向量，或简称向量。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;向量方程 $x_1 \mathbf{a_1}+x_2 \mathbf{a_2}+x~_3 \mathbf{a_3}=\mathbf{b}$ 与增广矩阵为
$\left[
\begin{matrix}
   \mathbf{a_1} &\mathbf{a_2} &\mathbf{a_3} &\mathbf{b}
  \end{matrix} 
\right]$
的线性方程组有相同的结集；  
  
  2. 求解方法  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;转换为增广矩阵，同样利用行化简算法求解。  
  
+ 矩阵方程  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;向量的线性组合可视为矩阵与向量的乘积，线性方程组（1.1）用矩阵表示为如下形式：  
$$
A\mathbf{x}=\mathbf{b}\\
\mathbf{x}=(x_1,x_2,x_3)\\
\mathbf{b}=(b_1,b_2,b_3)\\
A=\left[\begin{matrix} a_1 &a_2 &a_3\\ a_4 &a_5 &a_6\\ a_7 &a_8 &a_9 \end{matrix}\right]
$$  
  1. 定义  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;形如$A\mathbf{x}=\mathbf{b}$ 的方程为矩阵方程。其中 $A$ 为 $m\times n$ 的矩阵，而 $\mathbf{x}$ 和 $\mathbf{b}$ 为 $n$ 维向量。  
  
  2. 求解方法  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同上$\dots$
  
  3. 性质  
$$
A(\mathbf{u}+\mathbf{v})=A\mathbf{u}+A\mathbf{v}\\
A(c\mathbf{u})=cA\mathbf{u}
$$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中 $c$ 是标量，$A$ 是 $m\times n$ 矩阵，$\mathbf{u}$ 和 $\mathbf{v}$ 是 $\mathbb{R}^n$ 中的向量。  

### <div id="12-基本运算">1.2 基本运算</div>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵的基本运算包括加减乘除，逆.....  
#### <div id="121-矩阵乘法">1.2.1 矩阵乘法</div>  
+ 一般矩阵乘积  
$$\mathbf{A}\mathbf{B}$$  
+ 点积
$$\mathbf{A}\cdot\mathbf{B}=\mathbf{A}^T\mathbf{B}$$  
+ 阿达马乘积  
$$\mathbf{A}\circ\mathbf{B}$$  
+ 克罗内克乘积  
$$\mathbf{A}\otimes\mathbf{B}$$  

[1] wiki.[矩陣乘法](https://zh.wikipedia.org/wiki/矩陣乘法)  
[2] wiki.[点积](https://zh.wikipedia.org/wiki/点积)  

  
### <div id="123-矩阵求导">1.2.3 矩阵微积分</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵微积分就是用向量和矩阵来表示因变量中每个分量对自变量中每个分量的导数。根据变量格式可以将矩阵微分简单概括为表3-1所示的9种形式：
<div align="center">

|             |标量$y$      |向量$\mathbf{y}$    |矩阵$\mathbf{Y}$  |  
|----            |:---------:|:----: |:----:|  
|标量 $x$          |$\frac{\partial y}{\partial x}$|$\frac{\partial\mathbf{y}}{\partial x}$|$\frac{\partial\mathbf{Y}}{\partial x}$|   
|向量$\mathbf{x}$ |$\frac{\partial y}{\partial\mathbf{x}}$|$\frac{\partial\mathbf{y}}{\partial\mathbf{x}}$||
|矩阵$\mathbf{X}$ |$\frac{\partial y}{\partial\mathbf{X}}$||| 
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;易知，表格中有三项为空，分别为矩阵$\mathbf{Y}$对向量$\mathbf{x}$的导数，矩阵$\mathbf{Y}$对矩阵$\mathbf{X}$的导数和向量$\mathbf{x}$对矩阵$\mathbf{Y}$的导数，这是由于矩阵$\mathbf{Y}$对向量$\mathbf{x}$的导数可视为矩阵$\mathbf{Y}$对多个标量 $x_i$ 求导，其结果是秩超过2的张量，无法用一个二维矩阵表示。其它的类似。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵求导有两种布局，即两种求导规则。这两种求导规则没有优劣之分，但需要保持在一个问题当中布局使用的一致性，否则容易导致混乱。
  
**符号**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了更加方便高效地说明矩阵微积分的具体形式，首先对矩阵微积分中的符号制定统一的规则。在该规则中，我们通过字体来区分标量，向量和矩阵。  

  1. 用粗体大写字母表示矩阵，如 $\mathbf{X}$,$\mathbf{Y}$ 等，其中 M(n,m) 表示 n 行 m 列的矩阵。
  2. 用粗体小写字母表示向量，如 $\mathbf{x}$,$\mathbf{y}$ 等，其中 M(n,1) 表示包含 n 个元素的列向量，根据《线性代数及其应用》中的定义向量其实就是指一列的矩阵。
  3. 用斜体小写字母表示标量，如 *x*,*y*等。
  4. $\mathbf{X}^T$表示矩阵转置， tr(X)是轨迹，det(X) 或 |X| 是行列式。
  5. 除非有特别说明。字母表中前半部分的普通的字母(a, b, c, …)用来表示常量,后半部分字母(t, x, y, …)用来表示变量。  

**分子布局**  

1. 对标量 x 求导  
分子为标量 y ：$\frac{dy}{dx}$；  
分子为向量 $\mathbf{y} = (y_1,y_2,\dots,y_n)$ ： $\frac{\partial\mathbf{y}}{\partial x} = (\frac{\partial y_1}{\partial x},\frac{\partial y_2}{\partial x},\dots,\frac{\partial y_n}{\partial x})$，这是列向量且为正切向量；  
分子为矩阵 $\mathbf{Y} = \left[
 \begin{matrix}
   y_{11} & y_{12} & \dots & y_{1m}\\
   y_{21} & y_{22} & \dots & y_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   y_{n1} & y_{n2} & \cdots & y_{nm} 
  \end{matrix} 
\right]
 $ ： $\frac{\partial\mathbf{Y}}{\partial x} = \left[
 \begin{matrix}
   \frac{y_{11}}{\partial x} & \frac{y_{12}}{\partial x} & \dots & \frac{y_{1m}}{\partial x}\\
   \frac{y_{21}}{\partial x} & \frac{y_{22}}{\partial x} & \dots & \frac{y_{2m}}{\partial x} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{y_{n1}}{\partial x} & \frac{y_{n2}}{\partial x} & \cdots & \frac{y_{nm}}{\partial x} 
  \end{matrix} 
\right]
$。  
2. 对向量 $\mathbf{x} = (x_1,x_2,\dots,x_m)$求导  
分子为标量 y ：$\frac{dy}{d\mathbf{x}} =\left[ \begin{matrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \cdots &\frac{\partial y}{\partial x_m}\end{matrix} \right]$，该矩阵为 y 在空间 $\mathbf{R}^n$ 的梯度，该空间以 $\mathbf{x}$ 为基；  
分子为向量 $\mathbf{y} = (y_1,y_2,\dots,y_n)$ ： $\frac{\partial\mathbf{y}}{\partial x} = \left[
 \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_m}\\
   \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_m}\\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial y_n}{\partial x_1} & \frac{\partial y_n}{\partial x_2} & \cdots & \frac{\partial y_n}{\partial x_m} 
 \end{matrix} 
\right]$，该矩阵被称为雅可比矩阵（Jacobian）；  
3. 对矩阵 $\mathbf{X} = \left[
 \begin{matrix}
   x_{11} & x_{12} & \dots & x_{1m}\\
   x_{21} & x_{22} & \dots & x_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   x_{n1} & x_{n2} & \cdots & x_{nm} 
  \end{matrix} 
\right]$求导  
分子为标量 y ： $\frac{\partial y}{\partial\mathbf{X}} = \left[
 \begin{matrix}
   \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \dots & \frac{\partial y}{\partial x_{n1}}\\
   \frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \dots & \frac{\partial y}{\partial x_{n2}} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial y}{\partial x_{1m}} & \frac{\partial y}{\partial x_{2m}} & \cdots & \frac{y}{\partial x_{nm}} 
  \end{matrix} 
\right]
$；  

**分母布局**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其实分母布局的结果就是对应分子布局结果的转置，为了方便阅读及对比，还是将分母布局的相关结果在下面表示出来。  

1. 对标量 x 求导  
分子为标量 y ：$\frac{dy}{dx}$；  
分子为向量 $\mathbf{y} = (y_1,y_2,\dots,y_n)$ ： $\frac{\partial\mathbf{y}}{\partial x} = \left[
 \begin{matrix}
   \frac{\partial y_1}{\partial x} & \frac{\partial y_2}{\partial x} & \dots & \frac{\partial y_m}{\partial x}
  \end{matrix} 
\right]
 $，这是行向量；  
分子为矩阵 $\mathbf{Y} = \left[
 \begin{matrix}
   y_{11} & y_{12} & \dots & y_{1m}\\
   y_{21} & y_{22} & \dots & y_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   y_{n1} & y_{n2} & \cdots & y_{nm} 
  \end{matrix} 
\right]
 $ ： $\frac{\partial\mathbf{Y}}{\partial x} = \left[
 \begin{matrix}
   \frac{y_{11}}{\partial x} & \frac{y_{21}}{\partial x} & \dots & \frac{y_{n1}}{\partial x}\\
   \frac{y_{12}}{\partial x} & \frac{y_{22}}{\partial x} & \dots & \frac{y_{n2}}{\partial x} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{y_{1m}}{\partial x} & \frac{y_{2m}}{\partial x} & \cdots & \frac{y_{nm}}{\partial x} 
  \end{matrix} 
\right]
$。  
2. 对向量 $\mathbf{x} = (x_1,x_2,\dots,x_m)$求导  
分子为标量 y ：$\frac{dy}{d\mathbf{x}} =(\frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2},\cdots,\frac{\partial y}{\partial x_m})$；  
分子为向量 $\mathbf{y} = (y_1,y_2,\dots,y_n)$ ： $\frac{\partial\mathbf{y}}{\partial \mathbf{x}} = \left[
 \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_1}\\
   \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_n}{\partial x_2}\\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial y_1}{\partial x_m} & \frac{\partial y_2}{\partial x_m} & \cdots & \frac{\partial y_n}{\partial x_m} 
 \end{matrix} 
\right]$；  
3. 对矩阵 $\mathbf{X} = \left[
 \begin{matrix}
   x_{11} & x_{12} & \dots & x_{1m}\\
   x_{21} & x_{22} & \dots & x_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   x_{n1} & x_{n2} & \cdots & x_{nm} 
  \end{matrix} 
\right]$求导  
分子为标量 y ： $\frac{\partial y}{\partial\mathbf{X}} = \left[
 \begin{matrix}
   \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{12}} & \dots & \frac{\partial y}{\partial x_{1m}}\\
   \frac{\partial y}{\partial x_{21}} & \frac{\partial y}{\partial x_{22}} & \dots & \frac{\partial y}{\partial x_{2m}} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial y}{\partial x_{n1}} & \frac{\partial y}{\partial x_{n2}} & \cdots & \frac{y}{\partial x_{nm}} 
  \end{matrix} 
\right]
$；

**常用的基础求导规则**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有时可能并不需要将矩阵微积分的具体值求解出来，只是为了推导简化式子从而发现不同变量间的关系可以通过一些常用导规则将矩阵微积分进一步化简，便于计算及推导。下面表格中统一用小写加粗字母如$\mathbf{u}$和$\mathbf{v}$表示向量，大写加粗字母如$\mathbf{A}$表示矩阵，小写未加粗字母如 $u$ 和 $v$ 表示标量。  

+ 对标量求导时的求导规则：  
  <div align="center">
  <table>
     <tr>
        <td>条件</td>
        <td>表达式</td>
        <td>分子布局</td>
        <td>分母布局</td>
     </tr>
     <tr>
        <td colspan="4" align="center"><b>向量对标量求导</b></td>
   </tr>
     <tr>
        <td><b>a</b>不是x的函数</td>
        <td>$$\frac{\partial\mathbf{a}}{\partial x}=$$</td>
        <td colspan="2">$$\mathbf{0}$$</td>
   </tr>
     <tr>
        <td>a不是x的函数,$$\mathbf{u}=\mathbf{u}(x)$$</td>
        <td>$$\frac{\partial a\mathbf{u}}{\partial x}=$$</td>
        <td colspan="2">$$a\frac{\partial\mathbf{u}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$\mathbf{A}$不是x的函数,$$\mathbf{u}=\mathbf{u}(x)$$</td>
        <td>$$\frac{\partial\mathbf{Au}}{\partial x}=$$</td>
        <td>$$A\frac{\partial\mathbf{u}}{\partial x}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial x}A^T$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(x)$$</td>
        <td>$$\frac{\partial\mathbf{u}^T}{\partial x}=$$</td>
        <td colspan="2">$$(\frac{\partial\mathbf{u}}{\partial x})^T$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(x),\mathbf{v}=\mathbf{v}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{u+v})}{\partial x}=$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{u}}{\partial x}+\frac{\partial\mathbf{v}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(x),\mathbf{v}=\mathbf{v}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{u}^T\times\mathbf{v})}{\partial x}=$$</td>
        <td>$$(\frac{\partial\mathbf{u}}{\partial x})^T\mathbf{v}+\mathbf{u}^T\frac{\partial\mathbf{v}}{\partial x}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial x}\mathbf{v}+\mathbf{u}^T(\frac{\partial\mathbf{v}}{\partial x})^T$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(x)$$</td>
        <td>$$\frac{\partial\mathbf{g}(\mathbf{u})}{\partial x}=$$</td>
        <td>$$\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{}u}\frac{\partial\mathbf{u}}{\partial x}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial x}\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(x)$$</td>
        <td>$$\frac{\partial\mathbf{f}(\mathbf{g}(\mathbf{u}))}{\partial x}=$$</td>
        <td>$$\frac{\partial\mathbf{f}(\mathbf{g})}{\partial\mathbf{g}}\frac{\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}\frac{\partial\mathbf{u}(x)}{\partial x}$$</td>
        <td>$$\frac{\partial\mathbf{u}(x)}{\partial x}\frac{\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}\frac{\partial\mathbf{f}(\mathbf{g})}{\partial\mathbf{g}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x),\mathbf{v}=\mathbf{v}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{U}\times\mathbf{v})}{\partial x}$$</td>
        <td>$$\frac{\partial\mathbf{U}}{\partial x}\times\mathbf{v}+\mathbf{U}\times\frac{\partial\mathbf{v}}{\partial x}$$</td>
        <td>$$\mathbf{v}^T\times\frac{\partial\mathbf{U}}{\partial x}+\frac{\partial\mathbf{v}}{\partial x}\times\mathbf{U}^T$$</td>
     </tr>
     <tr>
        <td colspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由上述公式可知，在分子布局下向量对标量的求导规则与函数对变量的求导规则一样，将向量与矩阵视为对标量的函数；而分母布局下只需导数视为分子布局下导数的转置，然后在利用转置的相关公式进行变化即可。</td>
     </tr>
     <tr>
        <td colspan="4" align="center">**矩阵对标量求导**</td>
     </tr>
     <tr>
        <td colspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为书写简便，矩阵对标量求导时只考虑分子布局，对于分母布局只有将对应的导数式子变为转置即可。</td>
     </tr>  
     <tr>
        <td>$$\mathbf{A}=\mathbf{U}(x)$$</td>
        <td>$$\frac{\partial a\mathbf{U}(x)}{\partial x}=$$</td>
        <td colspan="2">$$a\frac{\partial\mathbf{U}}{\partial x}$$</td>
   </tr>
     <tr>
        <td>$$\mathbf{A},\mathbf{B}不是x的函数\\ \mathbf{U}=\mathbf{U}(x)$$</td>
        <td>$$\frac{\partial\mathbf{AUB}}{\partial x}=$$</td>
        <td colspan="2">$$\mathbf{A}\frac{\partial\mathbf{U}}{\partial x}\mathbf{B}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x),\mathbf{V}=\mathbf{V}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{U+V})}{\partial x}=$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{U}}{\partial x}+\frac{\partial\mathbf{V}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x),\mathbf{V}=\mathbf{V}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{U}\mathbf{V})}{\partial x}=$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{U}}{\partial x}\mathbf{V}+\mathbf{U}\frac{\partial\mathbf{V}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x),\mathbf{V}=\mathbf{V}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{U}\oplus\mathbf{V})}{\partial x}=\\(异或运算)$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{U}}{\partial x}\oplus\mathbf{V}+\mathbf{U}\oplus\frac{\partial\mathbf{V}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x),\mathbf{V}=\mathbf{V}(x)$$</td>
        <td>$$\frac{\partial(\mathbf{U}\circ\mathbf{V})}{\partial x}=\\(复合运算)$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{U}}{\partial x}\circ\mathbf{V}+\mathbf{U}\circ\frac{\partial\mathbf{V}}{\partial x}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{U}=\mathbf{U}(x)$$</td>
        <td>$$\frac{\partial\mathbf{U}^{-1}}{\partial x}$$</td>
        <td colspan="2">$$-\mathbf{U}^{-1}\frac{\partial\mathbf{U}}{\partial x}\mathbf{U}^{-1}$$</td>
     </tr>
     <tr>     
        <td>$$\mathbf{U}=\mathbf{U}(x,y)$$</td>  
        <td>$$\frac{\partial^2\mathbf{U}^{-1}}{\partial x\partial y}=$$</td>
        <td colspan="2">$$\mathbf{U}^{-1}(\frac{\partial\mathbf{U}}{\partial x}\mathbf{U}^{-1}\frac{\partial\mathbf{U}}{\partial y})\mathbf{U}^{-1}$$</td>  
     </tr>
     <tr>
        <td>$\mathbf{A}$不是x的函数，$\mathbf{g}(\mathbf{X})$是具有标量系数的任何多项式，或由无穷多项式系列定义的任何矩阵函数（例如$e^{\mathbf{X}}，sin(\mathbf{X})，cos(\mathbf{X})，ln(\mathbf{X})等$）; $\mathbf{g}(x)$是等价的标量函数，$\mathbf{g}'(x)$是它的导数，$\mathbf{g} '(\mathbf{X})$是相应的矩阵函数导数</td>
        <td>$$\frac{\partial\mathbf{g}(x\mathbf{A})}{\partial x}=$$</td>
        <td colspan="2">$$\mathbf{A}\mathbf{g}'(x\mathbf{A})=\mathbf{g}'(x\mathbf{A})\mathbf{A}$$</td>
     </tr>
     <tr>
        <td>$\mathbf{A}$不是$x$的函数</td> 
        <td>$$\frac{\partial e^{\mathbf{A}x}}{\partial x}=$$</td>
        <td colspan="2">$$\mathbf{A}e^{\mathbf{A}x}=e^{\mathbf{A}x}\mathbf{A}$$</td>
     </tr>
  </table>
</div>

+ 对向量求导时的求导规则：  
  <div align="center">
  <table>
     <tr>
        <td>条件</td>
        <td>表达式</td>
        <td>分子布局</td>
        <td>分母布局</td>
     </tr>
     <tr>
        <td colspan="4" align="center">**标量对向量求导**</td>
     </tr>
     <tr>
        <td>a不是$\mathbf{x}$的函数</td>
        <td>$$\frac{\partial a}{\partial\mathbf{x}}=$$</td>
        <td>$$\mathbf{0}^T$$</td>  
        <td>$$\mathbf{0}$$</td>
   </tr>
     <tr>
        <td>a不是$\mathbf{x}$的函数,$$u=u(\mathbf{x})$$</td>
        <td>$$\frac{\partial au}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$a\frac{\partial u}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{x}),v=v(\mathbf{x})$$</td>
        <td>$$\frac{\partial(u+v)}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\frac{\partial u}{\partial\mathbf{x}}+\frac{\partial v}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{x}),v=v(\mathbf{x})$$</td>
        <td>$$\frac{\partial (u\times v)}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$v\frac{\partial u}{\partial\mathbf{x}}+u\frac{\partial v}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{x})$$</td>
        <td>$$\frac{\partial g(u)}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\frac{\partial g(u)}{\partial u}\frac{\partial u}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{x})$$</td>
        <td>$$\frac{\partial f(g(u))}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\frac{\partial f(g)}{\partial g}\frac{g(u)}{\partial u}\frac{\partial u(\mathbf{x})}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(\mathbf{x}),\mathbf{v}=\mathbf{v}(\mathbf{x})$$</td>
        <td>$$\frac{\partial(\mathbf{u}\times\mathbf{v})}{\partial\mathbf{x}}=\frac{\partial\mathbf{u}^T\mathbf{v}}{\partial\mathbf{x}}$$</td>
        <td>$$\mathbf{v}^T\frac{\partial\mathbf{u}}{\partial\mathbf{x}}+\mathbf{u}^T\frac{\partial\mathbf{v}}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}\mathbf{v}+\frac{\partial\mathbf{v}}{\partial\mathbf{x}}\mathbf{u}$$</td>
     </tr>
     <tr>
        <td>$\mathbf{u}=\mathbf{u}(\mathbf{x}),\mathbf{v}=\mathbf{v}(\mathbf{x})$，$\mathbf{A}$不是$\mathbf{x}$的函数</td>
        <td>$$\frac{\partial(\mathbf{u}\mathbf{A}\mathbf{v})}{\partial\mathbf{x}}=\frac{\partial\mathbf{u}^T\mathbf{A}\mathbf{v}}{\partial\mathbf{x}}$$</td>
        <td>$$\mathbf{v}^T\mathbf{A}^T\frac{\partial\mathbf{u}}{\partial\mathbf{x}}+\mathbf{u}^TA\frac{\partial\mathbf{v}}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}\mathbf{A}\mathbf{v}+\frac{\partial\mathbf{v}}{\partial\mathbf{x}}\mathbf{A}^T\mathbf{u}$$</td>
     </tr>   
     <tr>
        <td colspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在分子布局下标量对向量量的求导规则与函数对变量的求导规则一样，尽管分子与分母布局的表达式一样但它们将使用不同的求导规则求解。而最后两种情况下，分子布局与分母布局的求导表达式不相同是由于两种求导规则下得到的矩阵形状各不相同，所以要调整相应乘数的形状与位置。</td>
     </tr>
     <tr>
        <td colspan="4" align="center">**向量对向量求导**</td>
     </tr>  
     <tr>
        <td>$\mathbf{a}$不是$\mathbf{x}$的函数</td>
        <td>$$\frac{\partial\mathbf{a}}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\mathbf{0}$$</td>
     </tr>
     <tr>
        <td></td>
        <td>$$\frac{\partial\mathbf{x}}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\mathbf{I}$$</td>
     </tr>
     <tr>
        <td>$\mathbf{A}$不是$\mathbf{x}$的函数</td>
        <td>$$\frac{\partial\mathbf{Ax}}{\partial\mathbf{x}}=$$</td>
        <td>$$\mathbf{A}$$</td>
        <td>$$\mathbf{A}^T$$</td>
     </tr>
     <tr>
        <td>$\mathbf{A}$不是$\mathbf{x}$的函数</td>
        <td>$$\frac{\partial\mathbf{x^TA}}{\partial\mathbf{x}}=$$</td>
        <td>$$\mathbf{A}^T$$</td>
        <td>$$\mathbf{A}$$</td>
     </tr>
     <tr>
        <td>$a$不是$\mathbf{x}$的函数，$$\mathbf{u}=\mathbf{u}(\mathbf{x})$$</td>
        <td>$$\frac{\partial a\mathbf{u}}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$a\frac{\partial\mathbf{u}}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(\mathbf{x}),v=v(\mathbf{x})$$</td>
        <td>$$\frac{\partial v\mathbf{u}}{\partial\mathbf{x}}=$$</td>
        <td>$$v\frac{\partial\mathbf{u}}{\partial\mathbf{x}}+\mathbf{u}\frac{\partial v}{\partial\mathbf{x}}$$</td>
        <td>$$v\frac{\partial\mathbf{u}}{\partial\mathbf{x}}+\frac{\partial v}{\partial\mathbf{x}}\mathbf{u}^T$$</td>
     </tr>
     <tr>
        <td>$\mathbf{A}$不是$\mathbf{x}$的函数，$$\mathbf{u}=\mathbf{u}(\mathbf{x})$$</td>
        <td>$$\frac{\partial\mathbf{A}\mathbf{u}}{\partial\mathbf{x}}=$$</td>
        <td>$$\mathbf{A}\frac{\partial\mathbf{u}}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}\mathbf{A}^T$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(\mathbf{x}),\mathbf{v}=\mathbf{v}(\mathbf{x})$$</td>
        <td>$$\frac{\partial(\mathbf{v}+\mathbf{u})}{\partial\mathbf{x}}=$$</td>
        <td colspan="2">$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}+\frac{\partial\mathbf{v}}{\partial\mathbf{x}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(\mathbf{x})$$</td>
        <td>$$\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{x}}=$$</td>
        <td>$$\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}\frac{\partial\mathbf{u}}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}$$</td>
     </tr>
     <tr>
        <td>$$\mathbf{u}=\mathbf{u}(\mathbf{x})$$</td>
        <td>$$\frac{\partial\mathbf{f}(\mathbf{g}(\mathbf{u}))}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{f}(\mathbf{g})}{\partial\mathbf{g}}\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}\frac{\partial\mathbf{u}}{\partial\mathbf{x}}$$</td>
        <td>$$\frac{\partial\mathbf{u}}{\partial\mathbf{x}}\frac{\partial\mathbf{g}(\mathbf{u})}{\partial\mathbf{u}}\frac{\partial\mathbf{f}(\mathbf{g})}{\partial\mathbf{g}}$$</td>
     </tr>
     <tr>
        <td colspan="4" align="center">向量对向量的导数可以划分为五类，（1）向量之和对向量的导数，（2）向量标量乘积对向量的导数，（3）向量矩阵乘积对向量的导数，（4）向量复合函数对向量的导数，根据普通函数对变量的求导规则求导，然后根据导数表达式将对应乘数（向量或矩阵）进行形状和位置的变化。</td>
     </tr>
  </table>
</div>  

+ 对矩阵求导时的求导规则：
  <div align="center">
  <table>
     <tr>
        <td>条件</td>
        <td>表达式</td>
        <td>分子布局</td>
        <td>分母布局</td>
     </tr>
     <tr>
        <td colspan="4" align="center">**标量对矩阵求导**</td>
     </tr>
     <tr>
        <td>$a$不是$\mathbf{X}$的函数</td>
        <td>$$\frac{\partial a}{\partial\mathbf{X}}=$$</td>
        <td>$$\mathbf{0}^T$$</td>  
        <td>$$\mathbf{0}$$</td>
     </tr>
     <tr>
        <td>a不是$\mathbf{X}$的函数,$$u=u(\mathbf{X})$$</td>
        <td>$$\frac{\partial au}{\partial\mathbf{X}}=$$</td>
        <td colspan="2">$$a\frac{\partial u}{\partial\mathbf{X}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{X}),v=v(\mathbf{X})$$</td>
        <td>$$\frac{\partial uv}{\partial\mathbf{X}}=$$</td>
        <td colspan="2">$$u\frac{\partial v}{\partial\mathbf{X}}+v\frac{\partial u}{\partial\mathbf{X}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{X}),v=v(\mathbf{X})$$</td>
        <td>$$\frac{\partial(u+v)}{\partial\mathbf{X}}=$$</td>
        <td colspan="2">$$\frac{\partial u}{\partial\mathbf{X}}+\frac{\partial v}{\partial\mathbf{X}}$$</td>
     </tr>
     <tr>
        <td>$$u=u(\mathbf{X})$$</td>
        <td>$$\frac{\partial g(u)}{\partial\mathbf{X}}=$$</td>
        <td colspan="2">$$\frac{\partial g(u)}{\partial u}\frac{\partial u}{\partial\mathbf{X}}$$</td>
        <tr>
        <td>$$u=u(\mathbf{X})$$</td>
        <td>$$\frac{\partial f(g(u))}{\partial\mathbf{X}}=$$</td>
        <td colspan="2">$$\frac{\partial f(g)}{\partial g}\frac{\partial g(u)}{\partial u}\frac{\partial u}{\partial\mathbf{X}}$$</td>
     </tr>
     </tr>
   </table>
   </div>  
  

当向量或矩阵乘积对向量或标量或矩阵求导时，分子分母布局不同，否则相同。  

 
### <div id="124-参数标准方程推导">1.2.4 参数标准方程推导</div>  
   
   
**参考**：  
[1] Vinicier.[机器学习中的线性代数之矩阵求导](https://blog.csdn.net/u010976453/article/details/54381248)  
[2] Echo.[矩阵求导 -- 机器学习常用](https://www.cnblogs.com/echo-coding/p/8629197.html)  
[3] Veröffentlicht am.[矩阵求导笔记](https://yushroom.github.io/2016/08/12/Matrix-Calculus/)  
[4] 仙守.[数学-矩阵计算（4）两种布局](https://blog.csdn.net/shouhuxianjian/article/details/46669365)  
[5] 维基.[Matrix_calculus](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)  
[6]  lx青萍之末.[矩阵求导、几种重要的矩阵及常用的矩阵求导公式](https://blog.csdn.net/daaikuaichuan/article/details/80620518)  
   

### <div id='13-矩阵参考'>1.3 参考</div> 
矩阵方程  
[1] Lay.[线性代数及其应用](https://github.com/Roggu123/Algorithm/blob/master/Book/《线性代数及其应用》中文PDF_第4版_英文PDF_第5版_习题指导.rar) 

## <div id="2-误差">第二章 误差</div>  
### <div id="21-标准差">2.1 标准差</div>  

### <div id="22-均方误差">2.2 均方误差</div>
应用于梯度下降算法中均方误差代价函数  

### <div id="23-均方根误差">2.3 均方根误差</div>  

### <div id="24-交叉熵">2.4 交叉熵</div>  
  
### <div id="25-平均绝对误差">2.5 平均绝对误差</div>  

### <div id="26-误差参考">2.4 参考</div>  
[1] -牧野-.[交叉熵损失函数和均方误差损失函数](https://blog.csdn.net/dcrmg/article/details/80010342)  
[2] Geron.[机器学习实战]() (P43-P44)  

## <div id="3-学习笔记">第三章 线性代数学习笔记</div>
### <div id="31-线性相关">3.1 线性相关</div>  
+ 线性相关定义：  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若$\mathbb{R}^n$中一组向量$\{v_1,v_2,\dots,v_p\}$为线性无关，若向量方程：$$x_1v_1+x_2v_2+\dots+x_pv_p=0$$  
  仅有平凡解。向量组$\{v_1,v_2,\dots,v_p\}$为线性相关的，若存在不全为零的权 $c_1,c_2,\dots,c_3$，使$$c_1v_1+c_2v_2+\dots+c_pv_p=0$$
   
+ 线性相关的判定：  
  1. 将齐次方程转化为增广矩阵，若有自由变量则线性相关；
  2. 对于两个向量，若一个向量是另一个向量的倍数则它们线性有关；
  3. 观察向量方程，若向量个数超过方程个数则线性相关；  
  
> 齐次方程： 形如 $\mathbf{A}x=\mathbf{0}$ 的线性方程组，其中$\mathbf{A}$ 为 $m\times n$ 的矩阵，$\mathbf{0}$ 为 $\mathbb{R}^m$ 中的零向量。
  

### <div id="32-线性变换">3.2 线性变换</div>  
+ 定义：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;变换（或映射）$T$ 称为线性的，若  
（1）对于 $T$ 定义域中一切 $\mathbf{u}$ 和 $\mathbf{v}$，$T(\mathbf{v} + \mathbf{u}) = T(\mathbf{u})+T(\mathbf{v})$。  
（2）对一切 $\mathbf{u}$ 和标量 c，$T(c\mathbf{u}) = cT(\mathbf{u})$。  
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;映射$T:\mathbb{R}^n\rightarrow\mathbb{R}^m$为到 $\mathbb{R}^m$ 的映射，若 $\mathbb{R}^m$ 中的任意 $\mathbf{b}$ 都至少有一个 $\mathbb{R}^n$ 中的 $\mathbf{x}$ 与之对应（也称为满射）。  
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;映射 $T:\mathbb{R}^n\rightarrow\mathbb{R}^m$ 称为一对一映射，若 $\mathbb{R}^m$ 中每一 $\mathbf{b}$ 都至多有一个 $\mathbb{R}^n$ 至多一个 $x$ 与之对应。 
+ 性质：  
若 $T$ 是线性变换，则  
（1）$T(\mathbf{0}) = \mathbf{0}$；  
（2）$T(c\mathbf{u}+d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$；  
（3）$T(c_1\mathbf{v}_1+c_2\mathbf{v}_2+\dots+c_p\mathbf{v}_p)=c_1T(\mathbf{v}_1)+c_2T(\mathbf{v}_2)+\dots+c_p\mathbf{v}_p$。  
+ 重要定理：  
（1）设 $T:\mathbb{R}^n\rightarrow\mathbb{R}^m$ 为线性变换，则存在唯一的矩阵 $\mathbf{A}$ 使$$T(x)=\mathbf{A}x，对 \mathbf{R}^n 中的一切x$$其中，$\mathbf{A}$是$m\times n$矩阵，它的第 j 列是向量 $T(\mathbf{e}_j)$，  $\mathbf{e}_j$ 是单位矩阵 $\mathbf{I}_n$ 的第 j 列：$$\mathbf{A}=[T(\mathbf{e}_1)\dots T(\mathbf{e}_n)]$$矩阵 $\mathbf{A}$ 称为线性变换的标准矩阵。  
  
  （2）存在与唯一性问题可以将线性变换与前面的线性方程，线性无关相互联系起来：  
  设 $T:\mathbb{R}^n\rightarrow\mathbb{R}^m$ 是线性变换，设 $A$ 是 $T$ 的标准矩阵，则  
  a. $T$ 为到 $\mathbb{R}^m$ 上的映射，当且仅当 $A$ 的各列生成 $\mathbf{R}^m$，即方程 $Ax=b$ 有解；  
  b. $T$ 是一对一映射，当且仅当 $A$ 的列线性无关，即方程 $Ax=0$ 仅有平凡解。
    
### <div id="33-行列式与特征向量辨析">3.3 行列式与特征向量辨析</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文将从定义，运算定理，作用三个角度对行列式，特征向量和特征值进行辨析。  

+ **定义**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当$n \geq2$时，$n\times n$矩阵$A$的**行列式**为形如$\pm a_{1j}detA_{1j}$的$n$项的和，其中$a_{11},a_{12},\dots,a_{1n}$来自矩阵$A$的第一行，其中加减号交替出现，即  
$$
\begin{align}
det A &= a_{11}\cdot det A_{11}-a_{12}\cdot det A_{12}+ \dots+(-1)^{i+j}a_{1n} det A_{1n}\\
&= \sum_{j=1}^{n}(-1)^{1+j}a_{1j} det A_{1j}
\end{align}
$$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有$n\times n$矩阵$A$，$x$是非零向量，若存在数$\lambda$使$Ax=\lambda x$成立，那么称$\lambda$为矩阵$A$的**特征值**，向量$x$为对应$\lambda$的**特征向量**。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由如上定义可知，行列式和特征值都表示一个值，而特征向量是一个向量，因此特征向量和行列式关系不大。特征值和行列式都是方阵的值，这是它们的相似之处。 然而，一个方阵只有一个行列式，但却有多个特征值。

### <div id="34-参数标准方程推导">3.4 参数标准方程推导</div>  
**参考**：  
[1] 木兄.[正规方程求解特征参数的推导过程](https://blog.csdn.net/chenlin41204050/article/details/78220280)
### <div id="38-参考">3.8 参考</div>  
  
 

 