# Introducation
Docker简介
## 组成
+ **Docker Client** : Docker提供给用户的客户端。Docker Client提供给用户一个终端，用户输入Docker提供的命令来管理本地或者远程的服务器。
+ **Docker Daemon** : Docker服务的守护进程。每台服务器（物理机或虚机）上只要安装了Docker的环境，基本上就跑了一个后台程序Docker Daemon，Docker Daemon会接收Docker Client发过来的指令,并对服务器的进行具体操作。
+ **Docker Images** : 俗称Docker的镜像，这个可难懂了。你暂时可以认为这个就像我们要给电脑装系统用的系统CD盘，里面有操作系统的程序，并且还有一些CD盘在系统的基础上安装了必要的软件，做成的一张 “只读” 的CD。
+ **Docker Registry** : 这个可认为是Docker Images的仓库，就像git的仓库一样，用来管理Docker镜像的，提供了Docker镜像的上传、下载和浏览等功能，并且提供安全的账号管理可以管理只有自己可见的私人image。就像git的仓库一样，docker也提供了官方的Registry，叫做[Dock Hub](http://hub.Docker.com)
+ **Docker Container** : 俗称Docker的容器，这个是最关键的东西了。Docker Container是真正跑项目程序、消耗机器资源、提供服务的地方，Docker Container通过Docker Images启动，在Docker Images的基础上运行你需要的代码。 你可以认为Docker Container提供了系统硬件环境，然后使用了Docker Images这些制作好的系统盘，再加上你的项目代码，跑起来就可以提供服务了。 听到这里，可能你会觉得是不是有点像一个VM利用保存的备份或者快照跑起来环境一样，其实是挺像的，但是实际上是有本质的区别。

## 具体使用
1. 安装**docker客户端**，[下载链接](https://www.docker.com/products/docker-desktop "With a Title")；
2. 安装完毕后，我们在终端中可以使用docker命令，如利用```docker version```查看**docker版本**（见[图片1](1.png)）；
3. 查看**docker镜像**```docker image```（见[图片2](2.png))  ： 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由图可知，docker镜像有```pycharm_helpers```(猜想:大概是将pycharm应用于docker中),```tensorflow/tensorflow```（猜想：docker中使用tensorflow）,和```busybox```(一个linux下的大工具箱，它集成压缩了 Linux 的许多工具和命令)。
4. 查看在运行的**docker容器**```docker ps```,查看所有的**docker容器**```docker ps -a```(见[图片3](3.png)和[图片4](4.png))：  
	由图可知，docker容器命令可在终端和jupter notebook中输入，但在获取最新 TensorFlow CPU 版本的 docker 映像后，再输入在jupyter notebooke中运行docker的命令后都只在终端中运行docker.(疑问？？？？？？)获取tensorflow的docker镜像后，如何在jupyter notebook中运行docker,1.修改docker配置. 
	
	[参考一](http://www.voidcn.com/article/p-gignqddb-bnx.html)  
	[参考二](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html)  
	[终端有关命令解释](https://www.jianshu.com/p/21d5afc1c079)
