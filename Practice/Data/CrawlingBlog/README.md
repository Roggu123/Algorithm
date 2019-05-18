# <center>注意事项（Notes）</center>
关于本文件夹中各文件的作用及验证结果
## 1.1 程序文件
### 1.1.1 CrawlingBlog_AddRead.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没啥用，但不敢删，试验当中的烂尾楼
### 1.1.2 CrawlingBlog.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;26日前参考博客[Python爬取博客网站所有页面文章内容](https://www.jianshu.com/p/b1721a4be55b) ，并进行了适合自己情况的修改，但其中的乱码问题一直未得到有效解决，中途放弃。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;26日后参考博客[python3爬虫 爬取图片，爬取新闻网站文章并保存到数据库](https://blog.csdn.net/qiushi_1990/article/details/78041347)，并根据自己的需求进行了一定的修改，结果基本符合自己的要求。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该代码为本次学习实践的**核心**。
### 1.1.3 StudyTest.py
+ 根据参考代码进行适合自己情况的修改并在该文件中试验
+ 对每一阶段运行成功的代码在此文件中进行备份
+ 对实践过程中的一些编程问题进行积累，学习，实践

## 1.2 结果文件
### 1.2.1 interest.csv
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;程序
``StudyTest.py``中``探究如何把数据存入表格``的方法一产生的结果。
### 1.2.2 record.csv
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;程序``CrawlingBlog.py``26日前的运行结果保存。
### 1.2.3 resluts.csv
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;程序``CrawlingBlog.py``26日后的运行结果保存，有乱码。
### 1.2.4 Table.csv
程序``CrawlingBlog.py``26日后的运行产生的结果，基本符合要求。

## 1.3 记录文档
### 1.3.1 Unicode.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;网页信息爬取时，网页及程序编码问题的积累。
### 1.3.2 README.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;说有用其实没什么用，说没用但是真有点用。
### 1.3.3 Process.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;忠实地记录了实践过程并进行了一定的总结。
### 1.3.4 ***.png
+ Confifuration.png :有关pycharm配置的截图；
+ Title_Tag.png: 查找博客标题位置的截图；
+ Author_Tag.png: 查找博客作者位置的截图；
+ ReadNum_Tag.png: 查找博客阅读量的截图；

## 1.4 配置文档
### 1.4.1 \_pycache_
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可能是python程序的配置文件吧，不要轻易删除。





## 1.10 参考
[Python爬取博客网站所有页面文章内容](https://www.jianshu.com/p/b1721a4be55b)   
[python3爬虫 爬取图片，爬取新闻网站文章并保存到数据库](https://blog.csdn.net/qiushi_1990/article/details/78041347)

