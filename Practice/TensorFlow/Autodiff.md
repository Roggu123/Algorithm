 
#<center><div id='算法中的数学'>自动微分</div></center>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在进行问题求解和优化时要不可避免的使用梯度。手工计算梯度过于复杂且容易出错，而各种编程环境提供了自动微分即自动计算梯度的方法。数值微分，符号微分，前向自动微分和后向自动微分是四种主要的自动微分方法。
## <div id="11-数值微分">1.1 数值微分</div>  
+ 定义  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;用数字计算偏导的近似值，如函数 $h(x)$ 的导数 $h'(x_0)$ 为函数在点 $x_0$ 处的斜率，或用如下方程计算：$$h'(x_0)=\lim_{\epsilon \to 0} \frac{h(x_0+\epsilon)-h(x_0)}{\epsilon}\tag{1.1}$$  

+ 代码  

  ```python    
  def f(x,y):
        return(x**2*y+y+2)  
        
  def derivative(f, x, y, x_eps, y_eps)
  ```
  
+ 优缺点


## <div id="112-符号微分">1.2 符号wei fen</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;仅含一列的矩阵称为列向量，或简称向量(Vector)。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所有两个元素的向量集记为$\mathbb{R}^2$，$\mathbb{R}$表示向量中的元素是实数，而2表示每个向量包含两个元素。一般称$\mathbb{R}^2$中的向量是实数的有序对。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一般将列向量$\left[\begin{matrix}2\\3\end{matrix}\right]$表示为 $(2,3)$ 的形式，而将一行矩阵就表示为$\left[\begin{matrix}2&3\end{matrix}\right]$。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个$n\times m$的矩阵可以理解为 $m-1$ 个向量的线性组合。也可以理解为 $n\times(m-1)$ 矩阵与一个向量的乘积。向量方程可以写成等价的形式为$Ax=b$的矩阵方程。
#### <div id="113-矩阵分类">1.1.3 矩阵分类</div>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;矩阵可分为对称矩阵，可逆矩阵，分块矩阵， 方阵......  

#### <div id="114-矩阵方程">1.1.4 矩阵方程</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对比矩阵方程，线性方程组，向量方程，通过逐步推导辨析三者区别，关系。探究他们有解的条件。
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

## <div id="3-学习笔记">第三章 学习笔记</div>
### <div id="31-线性相关">3.1 线性相关</div>  
+ 线性相关定义：  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若$\mathbb{R}^n$中一组向量$\{v_1,v_2,\dots,v_p\}$为线性无关，若向量方程：$$x_1v_1+x_2v_2+\dots+x_pv_p=0$$  
  仅有平凡解。向量组$\{v_1,v_2,\dots,v_p\}$为线性相关的，若存在不全为零的权 $c_1,c_2,\dots,c_3$，使$$c_1v_1+c_2v_2+\dots+c_pv_p=0$$
   
+ 线性相关的判定：  
  1. 将齐次方程转化为增广矩阵，若有自由变量则线性相关；
  2. 对于两个向量，若一个向量是另一个向量的倍数则它们线性有关；
  3. 观察向量方程，若向量个数超过方程个数则线性相关；  

### <div id="32-线性变换">3.2 线性变换</div>  

### <div id="38-参考">3.8 参考</div>  
  
 

