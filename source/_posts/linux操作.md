---
title: linux操作
date: 2017-08-13 09:48:19
tags:
- linux
- ssh
- scp
- ip
---

对linux文件操作，网络操作，进程管理拾遗

<!-- more -->

# man

有时候查阅函数时，比如`write`系统调用，通过`man write`查出的结果不对。

    WRITE(1)
    NAME
         write — send a message to another user

    SYNOPSIS
         write user [tty]

原因是`man`是按照手册的章节号的顺序进行搜索的，比如：
使用`man -k write`命令可以查阅所有关于`write`的内容。

    ...
    write (1)            - send a message to another user
    ===> write (2)            - write to a file descriptor
    writev (2)           - read or write data into multiple buffers
    ...

`man write`是`write`命令手册，我们可以定位到`write (2)`是我们想要的库函数。可以在命令行输入`2`来查阅第二个`write`。
```
    man 2 write
```


# 网络操作
## 配置静态ip

操作网络服务
```
/etc/init.d/networking stop #停止
/etc/init.d/networking start #开启
/etc/init.d/networking restart #重启
```

配置静态IP

```
sudo gedit /etc/network/interfaces # 打开配置文件
#在打开的文件中输入以下
auto lo
iface lo inet loopback
auto eth0
# 设置IPv4
iface eth0 inet static
    address 192.168.1.188
    netmask 255.255.255.0
    gateway 192.168.1.1
    dns-nameserver 8.8.8.8
# 设置IPv6
iface eth0 inet6 static
    address 2a04:f80:0754:168:225:218:171:0/112
    gateway 2a04:f80:0754::1
    dns-nameservers 2001:4860:4860::8888

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

## 通过IPv6访问Google等

在有IPv6支持的网络中，设置自己的静态IPv6。
如果没有IPv6网络支持，可以使用miredo网络工具。这是一款主要用于BSD和Linux的IPV6 Teredo隧道链接，可以转换不支持IPV6的网络连接IPV6，内核中需要有IPV6和TUN隧道支持。
安装也很简单：
```
sudo apt install miredo
```
通过`ifconfig`可以查看到多了一个`teredo`网卡。
接下来就可以`ping`通google的IPv6地址了。
```
ping6 ipv6.google.com
```
然后再在`/etc/hosts`中追加访问地址。幸好github上有开源项目：<https://github.com/lennylxx/ipv6-hosts>

```
sudo su
curl https://github.com/lennylxx/ipv6-hosts/raw/master/hosts -L >> /etc/hosts
/etc/init.d/networking restart
ping6 ipv6.google.com
```

参考： 
[1] [ubuntu16.04使用ipv6](http://blog.csdn.net/scylhy/article/details/72699166)
[2] [ubuntu 使用 ipv6 隧道](http://blog.letow.top/2017/11/05/ubuntu-%E5%BC%80%E5%90%AF-ipv6/)
[3] [IPv4下使用IPv6](https://newdee.cf/posts/8544442c/)

## 网桥bridge

安装两个配置网络所需软件包：
```
apt-get install bridge-utils        # 虚拟网桥工具
apt-get install uml-utilities       # UML（User-mode linux）工具
```
### 添加网卡/网桥

网桥是一个虚拟的交换机，而tap接口就是在这个虚拟交换机上用来和虚拟机连接的那个口。虚拟机就通过这么一个连接的方式和主机连接。

创建网桥，名字是virbr0
```
sudo brctl added virbr0
sudo ifconfig virbr0 192.168.122.1 net mask 255.255.255.0 up
```

创建tap接口，名字为tap0，并添加到网桥
```
sudo tunctl -t tap0
sudo ifconfig tap0 0.0.0.0 up
sudo brctl addif virbr0 tap0
```

### 删除网卡/网桥

刪除虚拟网卡
```
tunctl -d <虚拟网卡名>
```
刪除虚拟网桥
```
ifconfig <网桥名> down
brctl delbr <网桥名>
```
将网卡tap0, eth0 移出bridge(br0)
```
brctl delif br0 tap0
brctl delif br0 eth0
```


## scp

`scp`是`secure copy`的缩写，用于远程文件的安全拷贝，使用ssh传输，认证。
主机上的文件前可能需要用户名和主机指定并通过`:`连接文件。本地文件可通过相对或绝对路径表示。
### 表示格式为
`scp [参数] [原路径] [目标路径]`

### 命令参数
命令参数：

    -1  强制scp命令使用协议ssh1  
    -2  强制scp命令使用协议ssh2  
    -4  强制scp命令只使用IPv4寻址  
    -6  强制scp命令只使用IPv6寻址  
    -B  使用批处理模式（传输过程中不询问传输口令或短语）  
    -C  允许压缩。（将-C标志传递给ssh，从而打开压缩功能）  
    -p 保留原文件的修改时间，访问时间和访问权限。  
    -q  不显示传输进度条。  
    **-r**  递归复制整个目录。  
    -v 详细方式显示输出。scp和ssh(1)会显示出整个过程的调试信息。这些信息用于调试连接，验证和配置问题。   
    -c cipher  以cipher将数据传输进行加密，这个选项将直接传递给ssh。   
    -F ssh_config  指定一个替代的ssh配置文件，此参数直接传递给ssh。  
    -i identity_file  从指定文件中读取传输时使用的密钥文件，此参数直接传递给ssh。    
    -l limit  限定用户所能使用的带宽，以Kbit/s为单位。     
    -o ssh_option  如果习惯于使用ssh_config(5)中的参数传递方式，   
    **-P** port  注意是大写的P, port是指定数据传输用到的端口号   
    -S program  指定加密传输时所使用的程序。此程序必须能够理解ssh(1)的选项。

需要关注的是文件夹`-r`和端口`-P`参数。

### 实例

复制本地文件到远端服务器。
```
scp -P 2222 local_file remote_username@remote_ip:remote_folder  
```
复制本地文件夹到远端服务器。
```
scp -P 2222 -r local_folder remote_username@remote_ip:remote_folder  
```

从远端服务器到本地，只需要把参数颠倒即可。

[1] [每天一个linux命令（60）：scp命令](http://www.cnblogs.com/peida/archive/2013/03/15/2960802.html)





# 文件操作


## tar

tar命令可以为linux的文件和目录创建档案。利用tar命令，可以把一大堆的文件和目录全部打包成一个文件。
首先要把打包和压缩两个概念搞清楚，打包是指将一大堆文件或目录变成一个总的文件；压缩则是将一个大的文件通过一些压缩算法变成一个小文件。
为什么要区分这两个概念呢？这源于Linux中很多压缩程序只能针对一个文件进行压缩，这样当你想要压缩一大堆文件时，你得先将这一大堆文件先打成一个包（tar命令），然后再用压缩程序进行压缩（gzip bzip2命令）。

### tar用法

```
tar <选项> (打包的文件或目录列表)
```

选项为：

    -A或--catenate：新增文件到以存在的备份文件；
    -B：设置区块大小；
    -c或--create：建立新的备份文件；
    -C <目录>：这个选项用在解压缩，若要在特定目录解压缩，可以使用这个选项。
    -d：记录文件的差别；
    -x或--extract或--get：从备份文件中还原文件；
    -t或--list：列出备份文件的内容；
    -z或--gzip或--ungzip：通过gzip指令处理备份文件；
    -Z或--compress或--uncompress：通过compress指令处理备份文件；
    -f<备份文件>或--file=<备份文件>：指定备份文件；
    -v或--verbose：显示指令执行过程；
    -r：添加文件到已经压缩的文件；
    -u：添加改变了和现有的文件到已经存在的压缩文件；
    -j：支持bzip2解压文件；
    -v：显示操作过程；
    -l：文件系统边界设置；
    -k：保留原有文件不覆盖；
    -m：保留文件不被覆盖；
    -w：确认压缩文件的正确性；
    -p或--same-permissions：用原来的文件权限还原文件；
    -P或--absolute-names：文件名使用绝对名称，不移除文件名称前的“/”号；
    -N <日期格式> 或 --newer=<日期时间>：只将较指定日期更新的文件保存到备份文件里；
    --exclude=<范本样式>：排除符合范本样式的文件。

### tar实例

将文件全部打包成tar包：
```
tar -cvf log.tar log2012.log    #仅打包，不压缩！ 
tar -zcvf log.tar.gz log2012.log   #打包后，以 gzip 压缩 
tar -jcvf log.tar.bz2 log2012.log  #打包后，以 bzip2 压缩 
```
在选项f之后的文件档名是自己取的，我们习惯上都用 .tar 来作为辨识。 如果加z选项，则以.tar.gz或.tgz来代表gzip压缩过的tar包；如果加j选项，则以.tar.bz2来作为tar包名。

将tar包解压缩：
```
tar -zxvf /opt/soft/test/log.tar.gz
```

其实最简单的使用 tar 就只要记住底下的方式即可：
```
压　缩：tar -jcv -f filename.tar.bz2 要被压缩的文件或目录名称
查　询：tar -jtv -f filename.tar.bz2
解压缩：tar -jxv -f filename.tar.bz2 -C 欲解压缩的目录
```

解压缩xz压缩包：
```
tar xJvf filename.tar.xz
```

解压缩zip包：
```
unzip filename.zip
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

## find

`find`命令用于查找指定文件夹下的文件。

```
find [dir path] [params]
```

### params选项

```
-atime<24小时数>：查找在指定时间曾被存取过的文件或目录，单位以24小时计算；
-ctime<24小时数>：查找在指定时间之时被更改的文件或目录，单位以24小时计算；
-mtime<24小时数>：查找在指定时间曾被更改过的文件或目录，单位以24小时计算；
-exec<执行指令>：假设find指令的回传值为True，就执行该指令；
-name<范本样式>：指定字符串作为寻找文件或目录的范本样式；
-iname<范本样式>：此参数的效果和指定“-name”参数类似，但忽略字符大小写的差别；
-gid<群组识别码>：查找符合指定之群组识别码的文件或目录；
-path<范本样式>：指定字符串作为寻找目录的范本样式；
-perm<权限数值>：查找符合指定的权限数值的文件或目录；
-print：假设find指令的回传值为Ture，就将文件或目录名称列出到标准输出。格式为每列一个名称，每个名称前皆有"./"字符串；
-size<文件大小>：查找符合指定的文件大小的文件；
-uid<用户识别码>：查找符合指定的用户识别码的文件或目录；
-user<拥有者名称>：查找符和指定的拥有者名称的文件或目录；

```


### 实例

可以根据文件或正则表达式匹配。

在当前目录查找后缀名为`txt`的文件。
```
find . -name "*.txt"
```

找出后缀名不是`txt`的文件。
```
find . ! -name "*.txt"
```



详情参考：(find命令)[http://man.linuxde.net/find]


# 进程管理

## 查看进程和端口

查看进程
```
ps -ef | grep 进程名
```

查看端口占用情况

```
netstat -nap | grep 端口号
```

## 让进程后台运行

之前通过ssh登陆Linux服务器，在服务器上运行程序遇到这样一个问题，就是ssh连接中断或退出后，运行的程序会中止。因此，怎样才能让进程在后台长时间的运行呢？

经过查询[Linux 技巧：让进程在后台可靠运行的几种方法](https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/),总算找到了解决办法。

### 解决办法

当用户注销（logout）或者网络断开时，终端会收到 HUP（hangup）信号从而关闭其所有子进程。因此，我们的解决办法就有两种途径：要么让进程忽略 HUP 信号，要么让进程运行在新的会话里从而成为不属于此终端的子进程。

### nohup

nohup 的用途就是让提交的命令忽略 hangup 信号。需在要处理的命令前加上 nohup即可，标准输出和标准错误缺省会被重定向到nohup.out文件中。一般我们可在结尾加上"&"来将命令同时放入后台运行，也可用">filename 2>&1"来更改缺省的重定向文件名。


### setsid

nohup 无疑能通过忽略 HUP 信号来使我们的进程避免中途被中断，但如果进程不属于接受 HUP 信号的终端的子进程，那么NOHUP 指令将无效。幸运的是setsid 就能帮助我们做到这一点。仅在命令前加setsid即可。

### &

将一个或多个命名包含在“()”中就能让这些命令在子 shell 中运行中，当我们将"&"也放入"()"内之后，就会发现所提交的作业并不在作业列表中，也就是说，是无法通过jobs来查看的。

### 跟后台进程有关的操作

Linux Jobs等前后台运行命令解

1. command & ：让进程在后台运行
2. ctrl + z ：可以将一个正在前台执行的命令放到后台，并且暂停
3. jobs ：查看后台运行的进程
4. fg %number ：让后台运行的序号为number（不是pid）的进程到前台来
5. bg %number ：让进程号为number的进程到后台去。

## 注销

ubuntu 11.10及其以上版本，注销的命令行为：
```
gnome-session-quit
```



## LD_PRELOAD

LD_PRELOAD是linux的环境变量，用于动态库的加载，动态库加载的优先级最高。加载顺序为LD_PRELOAD>LD_LIBRARY_PATH>/etc/ld.so.cache>/lib>/usr/lib。

举例为：
1. 首先通过gcc把源文件打包成动态库。
```
gcc -shared -fpic -o libpreload.so preload.c
```
2. 使用LD_PRELOAD加载\*.so文件。
```
LD_PRELOAD=./libpreload.so ./test
```

# 信息查看

## CPU

可以直接得到CPU详细的信息

```
sudo cat /proc/cpuinfo
```

查看物理CPU的个数

```
sudo cat /proc/cpuinfo |grep "physical id" | sort | uniq |wc -l

```

查看逻辑CPU的个数

```
cat /proc/cpuinfo |grep "processor"|wc -l
```

查看CPU是几核
```
cat /proc/cpuinfo |grep "cores"|uniq
```
 
查看CPU的主频
```
cat /proc/cpuinfo |grep MHz|uniq
```

## 当前操作系统内核信息

```
uname -a
```

## 当前操作系统发行版本
```
cat /etc/issue

```
得到更详细的信息
```
sudo lsb_release -a
```

## 当前CPU运行模式

```
getconf LONG_BIT
```
如果是32，说明当前CPU运行在32bit模式下, 但不代表CPU不支持64bit。
如果是64，说明当前CPU支持64bit。

参考

[linux 下查看机器是cpu是几核的](https://www.cnblogs.com/xd502djj/archive/2011/02/28/1967350.html)