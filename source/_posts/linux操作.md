---
title: linux操作
date: 2017-08-13 09:48:19
tags:
- linux
- ssh
- scp
- ip
- zsh
categories:
- [linux]
---

对linux网络操作，文件操作，进程管理拾遗。针对发行版本为Ubuntu 16.04，间或有 Ubuntu 18.04。

<!-- more -->

# Linux 终端命令行快捷键

+ `Ctrl R` : 再按历史命令中出现过的字符串：按字符串寻找历史命令（重度推荐）
+ `Tab` : 自动补齐（重度推荐）
移动
+ `Ctrl A` ： 移动光标到命令行首
+ `Ctrl E` : 移动光标到命令行尾
+ `Ctrl B` : 光标后退一个`字符`
+ `Ctrl F` : 光标前进一个`字符`
+ `Alt F` : 光标前进一个`单词`
+ `Alt B` : 光标后退一个`单词`
编辑
+ `Ctrl H` : 删除光标的前一个字符
+ `Ctrl D` : 删除当前光标所在字符
+ `Alt D` : 删除光标之后的一个 `单词`
+ `Ctrl W` : 删除光标前的 `单词` (Word, 不包含空格的字符串)
+ `Ctrl K` ：删除光标之后所有字符
+ `Ctrl U` : 清空当前键入的命令
+ `Ctrl Y` : 粘贴 `Ctrl W` 或 `Ctrl K` 删除的内容

盗图一张...**Ctrl U 命令应当是清空所有输入命令，而非光标之前所有字符！**
![bash命令行命令操作](../linux操作/bash命令行命令.jpg)

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

其中查询出得每条结果后面数字代表得意义：
```
       1       可执行程序或 shell 命令
       2       系统调用(内核提供的函数)
       3       库调用(程序库中的函数)
       4       特殊文件(通常位于 /dev)
       5       文件格式和规范，如 /etc/passwd
       6       游戏
       7       杂项(包括宏包和规范，如 man(7)，groff(7))
       8       系统管理命令(通常只针对 root 用户)
       9       内核例程 [非标准
```

`man write`是`write`命令手册，我们可以定位到`write (2)`是我们想要的库函数。可以在命令行输入`2`来查阅第二个`write`。
```
    man 2 write
```

# 查找软件源

## 更新软件源

`ubuntu 14.04` 的软件源配置文件是 `/etc/apt/sources.list` 。将系统自带的该文件做个备份，将该文件替换为下面内容。

- 更新为清华源

```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse
```

- 更新软件包

```
sudo apt update
sudo apt upgrade
```

`ubuntu` 其他系列如 `ubuntu16.04` 需要的软件源参考<https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/>。

## 查找不确定的软件包

比如查找 支持 `gd` 的 `php` 软件包。
```
apt-cache search gd | grep php
```

# 网络操作
## 配置静态ip

操作网络服务
```
/etc/init.d/networking stop #停止
/etc/init.d/networking start #开启
/etc/init.d/networking restart #重启
```

+ 配置静态IP  

**Ubuntu16.04使用**

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

**注意**：*dns-nameservers* 设置不起作用，我设置了两个dns，不知道为啥。

设置DNS。
```
sudo gedit /etc/resolv.conf
#添加以下记录
nameserver dns的ip地址,如8.8.8.8
#重启服务
sudo /etc/init.d/networking restart
```

**注意** ：上面设置的文件重启后会覆盖，如果要持久的保存，需要修改：*/etc/resolvconf/resolv.conf.d/base* 。

```
vim /etc/resolvconf/resolv.conf.d/base
nameserver 172.16.3.4 
nameserver 172.16.3.3 
```

使DNS生效  
```
sudo /etc/init.d/resolvconf restart 
```


** Ubuntu18.04 **

在18.04上在 */etc/network/interfaces文件* 配置ip地址也是可以用的，`但是要重启才能生效`。通过 `service networking restart` 无效。  


18.04上新采用的netplan命令。网卡信息配置在 */etc/netplan/01-network-manager-all.yaml* 文件，需做如下配置: 

    # Let NetworkManager manage all devices on this system
    network:
      version: 2
      # renderer: NetworkManager
      ethernets:
              enp4s0:
                    dhcp4: no
                    addresses: [192.168.0.123/24]
                    gateway4: 192.168.0.1
                    nameservers:
                            addresses: [8.8.8.8, 8.8.4.4]

然后使用以下命令使配置立即生效。  
```
sudo netplan apply
```

查看IP是否设置成功。  

```
ip a
```

这里有几点需要注意： 
1. 将renderer: NetworkManager注释，否则netplan命令无法生效； 
2. ip配置信息要按如上格式，使用yaml语法格式，每个配置项使用 **空格缩进表示层级**；  
3. 对应配置项后跟着 **冒号，之后要接个空格** ，否则netplan命令也会报错。

参考：  
[Ubuntu 18.04 LTS设置固定ip](https://blog.csdn.net/u010039418/article/details/80934346)

** Ubuntu18.04 DNS**

在**Ubuntu 18.04**中通过 */etc/resolv.conf* 设置DNS后，重启不起作用。这是一个自动生成文件，来自于进程 *systemd-resolved*。  

> This file is managed by man:systemd-resolved(8). Do not edit.

通过修改配置文件 */etc/systemd/resolved.conf* 可以修改 DNS。  

```
sudo vim /etc/systemd/resolved.conf
```

> [Resolve]
> DNS=8.8.8.8

重启服务：
```
sudo systemctl restart systemd-resolved
```
查看DNS配置信息
```
sudo systemd-resolve --status
```

> Global
> DNS Servers: 8.8.8.8

参考 [Ubuntu-18.04 DNS 配置](https://vqiu.cn/ubuntu-18-04-dns-config/)  

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

若hosts中出现 *^M 字符*，是由于基于 DOS/Windows 的文本文件在每一行末尾有一个 CR（回车）和 LF（换行），而 UNIX 文本只有一个换行,即win每行结尾为\r\n，而linux只有一个\n 。
那么在VIM中替换掉即可。  

```
:%s/\r//g
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
sudo brctl addbr virbr0
sudo ifconfig virbr0 192.168.122.1 net mask 255.255.255.0 up
```

创建一张虚拟TUN网卡，名字为tap0。  
```
sudo tunctl -t tap0 -u <username>
```
将网卡设置为任何人都有权限使用  
```
sudo chmod 0666 /dev/net/tun
```

为tap0网卡设置一个IP地址，不要与真实的IP地址在同一个网段。  
```
sudo ifconfig tap0 <0.0.0.0> up
```
添加到网桥。  
```
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

### netplan 配置网桥

```
sudo vi /etc/netplan/50-cloud-init.yaml
```

在里面添加如下内容：

    network:
      version: 2
      ethernets:
        ens33:
          dhcp4: no
          dhcp6: no

      bridges:
        br0:
          interfaces: [ens33]
          dhcp4: no
          addresses: [192.168.0.51/24]
          gateway4: 192.168.0.1
          nameservers:
            addresses: [192.168.0.1]


```
sudo netplan apply
```

确认网络桥接状态：
```
sudo networkctl status -a
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
解压缩tar包：
```
tar -xvf file.tar
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

[1] [ulimit命令](http://man.linuxde.net/ulimit)

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



详情参考：[find命令](http://man.linuxde.net/find)

## 查看文件系统大小

+ 产看文件大小  

```
ls -sh filename
```
其中
> -s 表示查看文件的size 
> -h 表示以 `human readable` 的格式显示

+ 查看文件夹大小

返回该目录的大小  

```
du -sh
```
参数 `-sh` 同上。  

+  查看磁盘空间大小和剩余大小  

```
df -hl
```

得到的结果为：

    Filesystem      Size  Used Avail Use% Mounted on
    udev            7.8G     0  7.8G   0% /dev
    tmpfs           1.6G  1.7M  1.6G   1% /run
    /dev/sda2       164G   34G  123G  22% /
    tmpfs           7.9G  216K  7.9G   1% /dev/shm
    tmpfs           5.0M  4.0K  5.0M   1% /run/lock
    tmpfs           7.9G     0  7.9G   0% /sys/fs/cgroup
    ...

`/dev/sda2       164G   34G  123G  22% /` 表示： SATA硬盘接口的第一个硬盘（`a`），第二个分区（`2`），容量是`164G`，用了`34G`，可用是`123G`，因此利用率是`22%`， 被挂载到根分区目录上（`/`）。


[玩转Linux之硬盘分区格式化挂载与合并](https://o-my-chenjian.com/2017/05/10/Play-Disk-On-Linux/)  

+ 查看硬盘的分区  

```
fdisk -l
```

## 比较文件夹

linux中比较文件的命令为 `diff` ，`diff` 命令也可以[比较两个文件夹](https://blog.csdn.net/fengxianger/article/details/52936773)。
```
diff -urNa dir1 dir2
-a  Treat  all  files  as text and compare them     
    line-by-line, even if they do not seem to be text.

-N, --new-file
    In  directory  comparison, if a file is found in
    only one directory, treat it as present but empty
    in the other directory.

-r  When comparing directories, recursively compare
    any subdirectories found.
-u  Use the unified output format.
```

这里结果的格式采用的是 `合并格式的diff` ，这里可以[读懂diff](http://www.ruanyifeng.com/blog/2012/08/how_to_read_diff.html) 。
显示结果如下：
    

    　　--- f1    2012-08-29 16:45:41.000000000 +0800
    　　+++ f2    2012-08-29 16:45:51.000000000 +0800
    　　@@ -1,7 +1,7 @@
    　　 a
    　　 a
    　　 a
    　　-a
    　　+b
    　　 a
    　　 a
    　　 a

它的第一部分，也是文件的基本信息。

    　　--- f1    2012-08-29 16:45:41.000000000 +0800
    　　+++ f2    2012-08-29 16:45:51.000000000 +0800

"---"表示变动前的文件，"+++"表示变动后的文件。

第二部分，变动的位置用两个@作为起首和结束。

    　　@@ -1,7 +1,7 @@

前面的"-1,7"分成三个部分：减号表示第一个文件（即f1），"1"表示第1行，"7"表示连续7行。合在一起，就表示下面是第一个文件从第1行开始的连续7行。同样的，"+1,7"表示变动后，成为第二个文件从第1行开始的连续7行。

第三部分是变动的具体内容。

    　　 a
    　　 a
    　　 a
    　　-a
    　　+b
    　　 a
    　　 a
    　　 a

除了有变动的那些行以外，也是上下文各显示3行。它将两个文件的上下文，合并显示在一起，所以叫做"合并格式"。每一行最前面的标志位，空表示无变动，减号表示第一个文件删除的行，加号表示第二个文件新增的行。

+ 仅仅输出不同的文件名

```
-q   Report only whether the files differ, not the details of the differences.
-r   When comparing directories, recursively compare any subdirectories found.
```

```
diff -qr dir1 dir2
```
    
    Files dir1/different and dir2/different differ
    Only in dir1: only-1
    Only in dir2: only-2

[diff to output only the file names](https://stackoverflow.com/a/6217722)

## patch打补丁

通常对 `diff` 的结果，可以使用 `patch` 命令来自动补齐变动的内容。 `patch` 通过读入 `patch` 命令文件（可以从标准输入），对目标文件进行修改。

`patch` 的标准格式为

```
patch [options] [originalfile] [patchfile]
```

patch文件的样子：

    diff -Nur linux-2.4.15/Makefile linux/Makefile
    --- linux-2.4.15/Makefile       Thu Nov 22 17:22:58 2001
    +++ linux/Makefile      Sat Nov 24 16:21:53 2001
    @@ -1,7 +1,7 @@
     VERSION = 2
     PATCHLEVEL = 4
    -SUBLEVEL = 15
    -EXTRAVERSION =-greased-turkey
    +SUBLEVEL = 16
    +EXTRAVERSION =
     KERNELRELEASE=$(VERSION).$(PATCHLEVEL).$(SUBLEVEL)$(EXTRAVERSION)


绝大多数情况下，`patch` 都用以下这种简单的方式使用：

```
patch -p[num] <patchfile
```

`-p` 参数决定了是否使用读出的源文件名的前缀目录信息，不提供-p参数，则忽略所有目录信息。  
`-p0`（或者 `-p 0`）表示使用全部的路径信息，`-p1` 将忽略第一个"/"以前的目录，依此类推。 如 */usr/src/linux-2.4.15/Makefile* 这样的文件名，在提供 `-p3` 参数时将使用 *linux-2.4.15/Makefile* 作为所要patch的文件。

使用 `-R` 选项可以取消补丁。  

```
patch -p[num] <patchfile -R
```

需要留意以下几点：

1. 一次打多个patch的话，一般这些patch有先后顺序，得按次序打才行。
2. 在patch之前不要对原文件进行任何修改。
3. 如果patch中记录的原始文件和你得到的原始文件版本不匹配(很容易出现)，那么你可以尝试使用patch, 如果幸运的话，可以成功。大部分情况下，会有不匹配的情况，此时patch会生成rej文件，记录失败的地方，你可以手工修改。

[用Diff和Patch工具维护源码](https://www.ibm.com/developerworks/cn/linux/l-diffp/index.html)  
[diff和patch用法](https://www.cnblogs.com/tobeprogramer/archive/2013/04/28/3049561.html)

# 进程管理

## 查看进程和端口

查看进程
```
ps -ef | grep 进程名
```
结果：
```
UID        PID  PPID  C STIME TTY          TIME CMD

root         1     0  0 Nov02 ?        00:00:00 init [3]       
```
查看端口占用情况

```
netstat -nap | grep 端口号
```

### 列出目前所有的正在内存当中的程序

```
ps aux
```
得到的结果展示
```
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND

root         1  0.0  0.0  10368   676 ?        Ss   Nov02   0:00 init [3]
```
说明：

+ USER：使用者账号名
+ PID ：进程ID号，可以根据此来kill掉进程， `kill -9 PID`
+ %CPU：使用掉的 CPU 资源百分比
+ %MEM：占用的物理内存百分比
+ VSZ ：使用掉的虚拟内存量 (Kbytes)
+ RSS ：占用的固定的内存量 (Kbytes)
+ TTY ：该 process 是在那个终端机上面运作，若与终端机无关，则显示 ?，另外， tty1-tty6 是本机上面的登入者程序，若为 pts/0 等等的，则表示为由网络连接进主机的程序。
+ STAT：该程序目前的状态，主要的状态有
    + R ：该程序目前正在运作，或者是可被运作
    + S ：该程序目前正在睡眠当中 (可说是 idle 状态)，但可被某些讯号 (signal) 唤醒。
    + T ：该程序目前正在侦测或者是停止了
    + Z ：该程序应该已经终止，但是其父程序却无法正常的终止他，造成 zombie (疆尸) 程序的状态
+ START：被触发启动的时间
+ TIME ：实际使用 CPU 运作的时间
+ COMMAND：该程序的实际指令

### netstat

`netstat` 用来打印Linux中网络系统的状态信息，显示各种网络相关信息，如网络连接，接口状态，路由表， (Interface Statistics)，masquerade 连接，多播成员 (Multicast Memberships) 等等。Netstat用于显示与IP、TCP、UDP和ICMP协议相关的统计数据，一般用于检验本机各端口的网络连接情况。。 常用到的选项为：
```
-a或--all：                       显示所有连线中的Socket；
-A<网络类型>或--<网络类型>：        列出该网络类型连线中的相关地址；
-c或--continuous：                持续列出网络状态；
-C或--cache：                     显示路由器配置的快取信息；
-e或--extend：                    显示网络其他相关信息；
-F或--fib：                       显示FIB；
-g或--groups：                    显示多重广播功能群组组员名单；
-h或--help：                      在线帮助；
-i或--interfaces：                显示网络界面信息表单；
-l或--listening：                 显示监控中的服务器的Socket；
-n或--numeric：                   直接使用ip地址，而不通过域名服务器；
-N或--netlink或--symbolic：       显示网络硬件外围设备的符号连接名称；
-o或--timers：                    显示计时器；
-p或--programs：                  显示正在使用Socket的程序识别码和程序名称；
-r或--route：                     显示Routing Table；
-s或--statistice：                显示网络工作信息统计表；
-t或--tcp：                       显示TCP传输协议的连线状况；
-u或--udp：                       显示UDP传输协议的连线状况；
-v或--verbose：                   显示指令执行过程；
-V或--version：                   显示版本信息；
-x或--unix：                      此参数的效果和指定"-A unix"参数相同；
--ip或--inet：                    此参数的效果和指定"-A inet"参数相同。
```
+ 列出所有端口 
```
netstat -anp
```

+ 列出所有 tcp 端口 
```
netstat -antp
```

+ 列出所有 udp 端口 
```
netstat -anup
```
+ 只显示所有监听端口 
```
netstat -lnp
```
+ 只列出所有监听 tcp 端口 
```
netstat -ltnp
```

+ 只列出所有监听 udp 端口 
```
netstat -lunp
```
+ 只列出所有监听 UNIX 端口 
```
netstat -lxnp
```
+ 找出程序运行的端口
```
netstat -anp | grep ssh
```
> NOTE:并不是所有的进程都能找到，没有权限的会不显示，使用 root 权限查看所有的信息。

+ 找出运行在指定端口的进程
```
netstat -anp | grep ':3306'
```
+ 持续输出 netstat 信息(每隔一秒输出网络信息)
```
netstat -cnp
```
+ 显示所有端口的统计信息 
```
netstat -s
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

# 内核操作

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

### 删除老版本的内核

在安装了新内核后，旧版本的内核不会自动的删除。 它们会继续占用磁盘空间，并且占用着 Grub root menu。

对于 Ubuntu 操作系统，可以利用 `purge-old-kernels` bash 脚本来执行，它位于 `byobu` 包中。

```
sudo apt-get install byobu
```

然后执行 `purge-old-kernels` 即可。

它会默认保留两个最近的kernel 版本，当然也可以自己设置保留多少。

```
sudo purge-old-kernels --keep 3 -qy
```

[How to Remove Old Kernels in Debian and Ubuntu](https://www.pontikis.net/blog/remove-old-kernels-debian-ubuntu)

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

## 内核编译选项查看

ubuntu 操作系统下的查看方式为：

### 文件下查看
```
cat /usr/src/linux-headers-$(uname -r)/.config | grep NOUVEAU
```

### 从系统/boot目录下获取
```
cat /boot/config-$(uname -r) | grep NOUVEAU
```



参考

[linux 下查看机器是cpu是几核的](https://www.cnblogs.com/xd502djj/archive/2011/02/28/1967350.html)

# tools

## shell---zsh

[安装步骤](https://github.com/robbyrussell/oh-my-zsh/wiki/Installing-ZSH)：  

```
sudo apt install zsh
```

把默认的Shell改成zsh  
```
chsh -s /bin/zsh
```

logout 再登录。  

安装 zsh 配置工具 [Oh My ZSH](https://github.com/robbyrussell/oh-my-zsh)。 

```
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```


可以按照 [Oh My ZSH](https://github.com/robbyrussell/oh-my-zsh/wiki/Themes) 更改zsh的主题。

参考：  
[Ubuntu 18.04 LTS中安装和美化ZSH Shell](https://novnan.github.io/Linux/install-zsh-shell-ubuntu-18-04/)  

## remove zsh

在删除zsh前，将默认shell恢复成其他shell。
```
chsh -s /bin/bash
```

删除 zsh。
```
sudo apt-get --purge remove zsh
```

[Remove Zsh from Ubuntu 16.04](https://askubuntu.com/a/958124)


## BASH SHELL for Windows10


从 PowerShell 启动 支持Linux的Windows子系统。  
1. 以 administrator 启动 Windows PowerShell。  

2. 将以下命令输入PowerShell中以启动Win10中的Bash。  

```
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
```
    按照提示，重启reboot。  

3. 从 Windows 商店中搜索 linux，在搜索结果中下载不同的发行版本，这里选择 Ubuntu 应用。  
4. 启动Ubuntu即可，这里默认以 root 用户登录。


### 文件系统交互 

Ubuntu默认把磁盘挂载到/mnt目录下，可以直接`cd /mnt/c` 进入C盘，进而操作文件。


参考：
[How to Install Linux Bash Shell on Windows 10](https://itsfoss.com/install-bash-on-windows/)  
