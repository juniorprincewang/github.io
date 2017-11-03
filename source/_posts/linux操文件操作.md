---
title: linux操作
date: 2017-08-13 09:48:19
tags:
---

对linux文件操作，网络操作拾遗

<!-- more -->

# 网络操作

# 配置静态ip

操作网络服务
```
/etc/init.d/networking stop #停止
/etc/init.d/networking start #开启
/etc/init.d/networking restart #重启
``
配置静态IP
```
sudo gedit /etc/network/interfaces # 打开配置文件
#在打开的文件中输入以下
auto lo
iface lo inet loopback
auto eth0
iface eth0 inet static
address 192.168.1.188
netmask 255.255.255.0
gateway 192.168.1.1
dns-nameserver 8.8.8.8
```
需要说明的是：
1. `auto eth0`表示让网卡`eth0`开机自动挂载`eth0`。
2. `eth0`是通过`ifconfig`得出的网卡名称。
3. 将`eth0`的IP分配方式修改为静态分配(static)后，为其制定IP、网关、子网掩码、DNS等信息。

设置DNS。
```
sudo gedit /etc/resolv.conf
#添加以下记录
nameserver dns的ip地址,如8.8.8.8
#重启服务
sudo /etc/init.d/networking restart
```

# 文件操作
## 压缩解压缩


```
tar xf archive.tar.xz
tar xf archive.tar.gz
tar xf archive.tar

```

##  压缩

```
tar xvfJ filename.tar.xz
```

## ulimit

ulimit命令用来限制系统用户对shell资源的访问。

ulimit 用于限制 shell 启动进程所占用的资源，支持以下各种类型的限制：所创建的内核文件的大小、进程数据块的大小、Shell 进程创建文件的大小、内存锁住的大小、常驻内存集的大小、打开文件描述符的数量、分配堆栈的最大大小、CPU 时间、单个用户的最大线程数、Shell 进程所能使用的最大虚拟内存。同时，它支持硬资源和软资源的限制。

作为临时限制，ulimit 可以作用于通过使用其命令登录的 shell 会话，在会话终止时便结束限制，并不影响于其他 shell 会话。而对于长期的固定限制，ulimit 命令语句又可以被添加到由登录 shell 读取的文件中，作用于特定的 shell 用户。

```
-a：显示目前资源限制的设定； 
-c ：设定core文件的最大值，单位为区块； 
-d <数据节区大小>：程序数据节区的最大值，单位为KB； 
-f <文件大小>：shell所能建立的最大文件，单位为区块； 
-H：设定资源的硬性限制，也就是管理员所设下的限制； 
-m <内存大小>：指定可使用内存的上限，单位为KB； 
-n <文件数目>：指定同一时间最多可开启的文件数； 
-p <缓冲区大小>：指定管道缓冲区的大小，单位512字节； 
-s <堆叠大小>：指定堆叠的上限，单位为KB； 
-S：设定资源的弹性限制； 
-t ：指定CPU使用时间的上限，单位为秒； 
-u <程序数目>：用户最多可开启的程序数目； 
-v <虚拟内存大小>：指定可使用的虚拟内存上限，单位为KB。
```

[1] (ulimit命令) [http://man.linuxde.net/ulimit]
