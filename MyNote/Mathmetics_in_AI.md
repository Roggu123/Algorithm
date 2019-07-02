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
           * [1.2.3 矩阵求导](#123-矩阵求导)
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
x_1 \mathbf{a_1}+x_2 \mathbf{a_2}+x_3 \mathbf{a_3}=\mathbf{b}\\
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
### <div id="123-矩阵求导">1.2.3 矩阵求导</div>  
**思路**：  
  
 1. 矩阵求导有两种布局：分子布局，分母布局  
 2. 每种布局又有五种形式，标量对向量，向量对标量，矩阵对标量，标量对矩阵，向量对向量。  
 3. 先学会如何求，然后探究为什么有两种布局。  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵求导包含两种布局
 
### <div id="124-参数标准方程推导">1.2.4 参数标准方程推导</div>  
   
   
**参考**：  
[1] Vinicier.[机器学习中的线性代数之矩阵求导](https://blog.csdn.net/u010976453/article/details/54381248)  
[2] Echo.[矩阵求导 -- 机器学习常用](https://www.cnblogs.com/echo-coding/p/8629197.html)  
[3] Veröffentlicht am.[矩阵求导笔记](https://yushroom.github.io/2016/08/12/Matrix-Calculus/)
   

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
    
### <div di="33-行列式与特征向量辨析">3.3 行列式与特征向量辨析</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文将从定义，运算定理，作用三个角度对行列式，特征向量和特征值进行辨析。  

+ **定义**  
行列式：当$n \geq2$时，$n\times n$矩阵$A$的行列式为$\pm a_{1j}detA_{1j}$，其中$a_{1j}$表示矩阵$A$的第一行，其中加减号交替出现。  
特征向量：有$n\times n$矩阵$A$，向量$x$是非零向量，若$Ax=\lambda x$，那么$\lambda$为矩阵$A$的特征值，向量$x$为对应特征值的特征向量。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由如上定义可知，行列式和特征值都表示一个值，而特征向量是一个向量，因此特征向量和行列式关系不大。特征值和行列式都是方阵的值，这是它们的相似之处。 然而，一个方阵只有一个行列式，但却有多个特征值。

### <div id="34-参数标准方程推导">3.4 参数标准方程推导</div>  
**参考**：  
[1] 木兄.[正规方程求解特征参数的推导过程](https://blog.csdn.net/chenlin41204050/article/details/78220280)
### <div id="38-参考">3.8 参考</div>  
  
 

