Table of Contents
=================

   * [第6章 拟牛顿法详解](#第6章-拟牛顿法详解)
      * [6.1 DFP算法](#61-DFP算法)
      * [6.2 BFGS算法](#62-BFGS算法)
      * [6.3 参考](#63-参考)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc) 
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
 



