---
title: 解决cuda程序执行出错
date: 2018-04-25 14:01:38
tags:
- linux
- CUDA
categories:
- solutions
- CUDA
---

`error while loading shared libraries: libcudart.so.8.0: cannot open shared object file: No such file or directory`
<!-- more -->


# 问题描述

> error while loading shared libraries: libcudart.so.8.0: cannot open shared object file: No such file or directory

# 解决办法：

首先确认 `/etc/profile` 或者 当前用户的BASH配置文件 `~/.bashrc`中包含了 `cuda8.0` 的安装路径及相应的库文件。

```
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-8.0/lib64
```
使配置文件生效
```
source /etc/profile
```
再次执行。

若仍提示相同的错误。这是由于新安装的共享动态链接库为系统所共享，需要手动激活。[敲黑板]

1. 往/lib 和 /usr/lib中添加库文件后，不用修改/etc/ld.so.conf的，但是完了之后要调一下ldconfig，不然这个library会找不到。 
2. 想往上面两个目录以外加东西的时候，一定要修改/etc/ld.so.conf，然后再调用ldconfig，不然也会找不到。 
3. 比如安装了一个mysql到/usr/local/mysql，mysql有一大堆library在/usr/local/mysql/lib下面，这时就需要在/etc/ld.so.conf下面加一行/usr/local/mysql/lib，保存过后ldconfig一下，新的library才能在程序运行时被找到。 
4. 如果想在这两个目录以外放lib，但是又不想在/etc/ld.so.conf中加东西（或者是没有权限加东西）。那也可以，就是export一个全局变量LD_LIBRARY_PATH，然后运行程序的时候就会去这个目录中找library。一般来讲这只是一种临时的解决方案，在没有权限或临时需要的时候使用。 
5. ldconfig做的这些东西都与运行程序时有关，跟编译时一点关系都没有。编译的时候还是该加-L就得加，不要混淆了。 
6. 总之，就是不管做了什么关于library的变动后，最好都ldconfig一下，不然会出现一些意想不到的结果。不会花太多的时间，但是会省很多的事。


ldconfig命令的用途主要是在默认搜寻目录/lib和/usr/lib以及动态库配置文件/etc/ld.so.conf内所列的目录下，搜索出可共享的动态链接库（格式如lib*.so*）,进而创建出动态装入程序(ld.so)所需的连接和缓存文件。

`ldconfig`通常在系统启动时运行，而当用户安装了一个新的动态链接库时，就需要手工运行这个命令。

来自: <http://man.linuxde.net/ldconfig>
```
sudo ldconfig /usr/local/cuda/lib64
```

OK。