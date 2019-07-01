Table of Contents
=================
* [我的Python参考手册](#我的Python参考手册)
  * [第一章 Python基础概念](#1-Python基础概念)
     * [1.1 包，模块，类](#11-包，模块，类)  
     * [1.2 数据类型](#12-数据类型) 
  * [第二章 Python模块及函数](#2-Python模块及函数)  
     * [2.1 datetime日期时间模块](#21-datetime)
     * [2.2 ...](#22-...)
     * [2.3 ...](#23-....)
  * [第三章 Python其它知识](#3-Python其它知识)  
  * [参考](#参考)  
  
        
     
# <center><div id="我的Python手册">我的Python手册</div></center>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结合[《Python 3.7.4rc1 文档》](https://docs.python.org/zh-cn/3/)对Python中基础概念以及一些常用模块和函数（方法）进行学习总结。

# <center><div id="1-Python基础概念">第一章 Python基础概念</div></center>  

## <div id="11-包，模块，类">1.1 包，模块，类</div>
+ 包  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一种通过用“带点号的模块名”来构造 Python 模块命名空间的方法。使用加点的模块名可以使不同模块软件包的作者不必担心彼此的模块名称一样。包也可以理解为是一个模块集合。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;必须要有 \__init\__.py 文件才能让 Python 将包含该文件的目录当作包。  
例子：  

  ```python
  from package import item
  ```  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中`package`为包名称，而`item`可以是包的子模块（或子包），也可以是包中定义的其他名称，如函数，类或变量。 import 语句首先测试是否在包中定义了item；如果没有，它假定它是一个模块并尝试加载它。如果找不到它，则引发 ImportError 异常。
    
+ 模块  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个包含Python定义和语句的文件。文件名就是模块名后跟文件后缀 .py。模块能定义函数，类和变量，模块里也能包含可执行的代码。  
例子：  
  
  ```python
  import item
  ```
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;导入一个名为`item`的模块，解释器首先寻找具有该名称的内置模块。如果没有找到，然后解释器从 sys.path 变量给出的目录列表里寻找名为`item.py`的文件。  
  
+ 类  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;是模块的组成部分，用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。  
例子：  
  
  ```python
  from datetime import timedelta
  from datetime import date
  ```  
  从模块`datetime`中导入`timedelta`和`date`两个类，其中`timedelta`表示表示两个 date 或者 time 的时间间隔。而`date`表示一个日期，包含有年，月，日。

# <center><div id="2-Python函数">第二章 Python模块及函数</div></center>  
## <div id="21-datetime">2.1 datetime日期时间模块</div>    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模块`datetime`提供了可以通过多种方式操作日期和时间的类。在支持日期时间数学运算的同时，实现的关注点更着重于如何能够更有效地解析其属性用于格式化输出和数据操作。  

+ 常用函数
 1. `datetime.utcnow()`  
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;返回当前的UTC日期及时间，即一个当前UTC时间的datetime对象。
 2. `datetiem.now(tz=None)`  
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;返回当前datetime对象，如果参数tz未指明，返回本地当前datetime对象，若指明则返回tz参数所代表时区的当前datetime对象。  
 3. `待补充`  

## <div id="22-...">2.2 ....</div>   

## <div id="23-...">2.3 ....</div>
np.arrange()用法  
range()用法  
%time用法   
# <center><div id="3-Python其它知识">第三章 Python其它知识</div></center>  
 

#<center><div id="参考">参考</div></center>  
[1] .[Python 3.7.4rc1 文档（Web）](https://docs.python.org/zh-cn/3/)  
[2] .[Python 3.7.4rc1 文档（PDF）](https://github.com/Roggu123/Algorithm/tree/master/Practice/Python_Manual)  
[3] 菜鸟教程.[Python 面向对象](http://www.runoob.com/python/python-object.html)