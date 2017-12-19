---
title: rCUDA
date: 2017-11-01 10:37:23
tags:
- GPU
- rCUDA
---

本篇博客讲述rCUDA、rCUDA的安装。

<!-- more -->

# rCUDA简介

[rCUDA](http://rcuda.net/index.php/what-s-rcuda.html)，（remtoe CUDA）是CUDA的远程调用版本，在本地无GPU的主机上远程访问有CUDA环境的GPU主机。

rCUDA是Client-Server架构的服务。下面就讲讲如何安装rCUDA。

## 准备条件

### CUDA8.0
目前的rCUDA是基于CUDA-8.0版本的，所以需要在宿主机和虚拟机上提前安装cuda-8.0，并配置好`PATH`和`LD_LIBRARY_PATH`路径。
安装最好使用

CUDA在server服务器中成功运行。使用CUDA的deviceQuery和bandwidthTest样例来测试。
### 确保client和server正常通信。

    1.  可以选择基于TCP/IP的通信（以太网）。
    2.  也可以选择基于RDMA的通信（InfiniBand或者RoCE）。使用Mellanox OFED的ib_write_bw和ib_read_bw测试IB或RoCE。


### 遇到的问题

#### 循环登录的问题

按照上述方式安装好驱动后，重启，到登录界面一切正常。输入登录密码之后，进入桌面，悲剧发生了：桌面一闪就退回到登录界面了，然后就陷入到了输入密码登录、弹出的循环。
其实简单卸载掉驱动就可以了。卸载方法是，首先在登录界面进入到Linux的shell i.e. tty model，同时按下Ctrl+Alt+F1 （F1~F6其中一个就可以）。(Ctrl+Alt+F7可以回到桌面界面)
然后输入用户名，回车，输入密码，回车，成功进入到shell，开始卸载NVIDIA驱动：
```
sudo apt-get remove --purge nvidia-*
sudo apt-get install ubuntu-desktop
sudo rm /etc/X11/xorg.conf
echo 'nouveau' | sudo tee -a /etc/modules
#重启系统
sudo reboot
```
重启之后就可以登录了。

# 安装rCUDA

去官网下载，需要填写信息。<http://rcuda.net/index.php/software-request-form.html>

我在这里保存了一份[rCUDAv16.11.04.02-CUDA8.0-linux64.tgz](../rCUDA/rCUDAv16.11.04.02-CUDA8.0-linux64.tgz)，我的系统是64位Ubuntu16.04。

在client和server两端都需要rCUDA的这份文件。

## rCUDA server

### 设置环境变量。

server使用的库`LD_LIBRARY_PATH`是`cuda-8.0`的库，而不是自己带的库。

```
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
```
如果是临时设置环境变量，那么就直接在终端里输入命令。如果想要永久设置可以有以下方法。

1. 修改/etc/profile文件
在文件中追加上述命令，此方法对所有用户都有效。
然后刷新。
2. 修改~/.bashrc
在文件中追加上述命令，对当前用户有效。
保护后为了及时生效。
```
source ~/.bashrc
```
验证有没有生效。
```
echo $PATH
```

### 开启rCUDA server

```
# 一定要进入子目录*/bin/中
cd rCUDAv16.11.04.02-CUDA8.0/bin/
./rCUDAd
```
**BUT!**粗问题了！！！

    ./rCUDAd: error while loading shared libraries: libcudnn.so.5: cannot open shared object file: No such file or directory。

搜索了一番发现，cuddn是一个独立于CUDA安装的库。专门用于做深度神经网络的库。The NVIDIA CUDA Deep Neural Network library (cuDNN) 。
OK！去官网搜索，找到了[cuDNN的安装教程](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
教程中给出的下载链接失效了，可以去这里找<https://developer.nvidia.com/rdp/cudnn-archive>。
先解压缩文件，然后将部分文件拷贝出来并修改为读取权限。
```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include 
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
```
教程还提供了验证cuDNN安装成功与否的samples。
单独下载cuDNN的samples文件，解压缩后有`mnistCUDNN`和`RNN`两个文件夹。我们验证仅需要`mnistCUDNN`文件夹。
编译`mnistCUDNN`样例。
```
make clean & make
```
运行样例。
```
./mnistCUDNN
```
在此处，出意外了！

    ./mnistCUDNN: error while loading shared libraries: libcudart.so.8.0 cannot open shared object object file: No such file or directory

怎么办!
原来，新安装的共享动态链接库为系统所共享，需要手动激活。[敲黑板]

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

OK！cudNN samples测试通过！

再次启动server!
```
# 一定要进入子目录*/bin/中
cd rCUDAv16.11.04.02-CUDA8.0/bin/
./rCUDAd
```

可以通过`./rCUDAd -h`查看相关命令。
```
-i : 不以守护进程运行，而是以交互式方式运行。
-l : 本地模式，使用TCP
-n <number>: 并发允许的服务器数量。0代表无限多个，默认值为0。
-p : 指定端口，默认为8308。
-v ：详细模式
-h ：打印帮助信息
```


## KVM

KVM全称是基于内核的虚拟机（Kernel-based Virtual Machine），它是Linux的一个内核模块，该内核模块使得Linux变成了一个Hypervisor。

QEMU是一款开源的模拟器及虚拟机监管器(Virtual Machine Monitor, VMM)。QEMU主要提供两种功能给用户使用。一是作为用户态模拟器，利用动态代码翻译机制来执行不同于主机架构的代码。二是作为虚拟机监管器，模拟全系统，利用其他VMM(Xen, KVM, etc)来使用硬件提供的虚拟化支持，创建接近于主机性能的虚拟机。
QEMU使用了KVM模块的虚拟化功能，来为自己的虚拟机提供硬件虚拟化加速。

libvirt又是一个C语言实现的虚拟机管理工具集，即由它提供的API来实现对qemu和kvm的这些管理过程。

KVM要求CPU支持，比如英特尔的VT或ADM-V，有些主板会在主板中**默认禁用CPU的虚拟化支持**，所以最好先进入BIOS中确认自己的CPU虚拟化功能处于开启状态。

好坑，折腾半天，原来BIOS中禁用了CPU的虚拟化支持。所以一定要先确认主机是否支持硬件虚拟化。不然，KVM无法加速，虚拟机的反应真的让人受不了。

### 验证主机是否支持硬件虚拟化

可以通过以下命令查看，如果不返回内容，说明机器不支持KVM或者BIOS中没有开启CPU硬件虚拟化。
```
egrep '(svm|vmx)' /proc/cpuinfo
```

我建议使用下面一种。安装cpu-checker之后通过运行kvm-ok来验证：
```
sudo apt-get install cpu-checker
```

运行
```
kvm-ok
```
如果出现`/dev/kvm exists`说明机器已经支持kvm；否则需要去BIOS中开启。

    INFO: /dev/kvm exists
    KVM acceleration can be used

### 安装KVM相关以依赖
```
sudo apt-get install kvm qemu-kvm libvirt-bin virtinst bridge-utils
```

它们的作用分别为：

    kvm: KVM的内核，通常linux系统自带
    qemu-kvm: KVM的设备模拟器，实际上kvm只是负责加速，qemu才是虚拟机管理器
    libvirt-bin: libvirt库，虚拟机命令行管理工具，包含很多实用工具，如后面需要大量使用的virsh。（安装之后会生成一个名为virbr0的网桥）
    virtinst: 虚拟机创建（virt-install）和克隆工具（vrit-clone）等
    birdge-utils: 用于桥接网卡的工具，如命令brctl）
        如果有图形化桌面，推荐安装virt-manager，这个工具可以非常方便地图形化管理虚拟机，就像常见的virtualbox/vmware界面那样，可以通过点点鼠标来完成虚拟机的管理。

KVM管理工具的一些注解及一些实用工具

    libvirt：操作和管理KVM虚机的虚拟化API，使用C语言编写，可以由Python,Ruby, Perl, PHP, Java等语言调用。可以操作包括KVM，vmware，XEN，Hyper-v, LXC，virtualbox等 Hypervisor。
    virsh：基于libvirt的命令行工具，后面需要大量使用。
    virt-v2v：虚机格式迁移工具，该工具与virt-sysprep都包含在包libguestfs-tools中，后面布署中会用到
    virt-install：创建KVM虚机的命令行工具
    virt-viewer：连接到虚拟机屏幕的工具，需要主机有桌面环境，该工具需要单独安装sudo apt-get install virt-viewer
    virt-clone：虚机克隆工具
    virt-top：类似于linux系统下的top命令，可以显示所有虚拟机CPU、内存等使用情况，该工具需要单独安装sudo apt-get install virt-top

### 参考
[1] [Ubuntu Server/Debian下的KVM虚拟机创建及管理详解](http://notes.maxwi.com/2016/11/29/kvm-create-and-manage)
[2] [Ubuntu 16.04.3 LTS (Xenial Xerus)下载地址](http://mirror.pnl.gov/releases/xenial/)


## rCUDA client


### 设置环境变量。

client使用的库`LD_LIBRARY_PATH`是**rCUDA**的`lib`库。

```
export PATH=$PATH:/usr/local/cuda-8.0/bin #CUDA的路径
export LD_LIBRARY_PATH=$HOME/rCUDA/lib:$LD_LIBRARY_PATH #rCUDA 库路径
# 配置远端GPU
export RCUDA_DEVICE_COUNT=1 #远端GPU的数量
export RCUDA_DEVICE_0 = 192.168.151.134:0 #第1个GPU

```

设置调用远端GPU的环境变量
```
RCUDA_DEVICE_COUNT=<number_of_GPUs>
export RCUDA_DEVICE_X=<server_name_or_ip_address[@port]>[:GPUnumber]
```

### 编译CUDA程序


```
cd $HOME/NVIDIA_CUDA_Samples/1_Utilities/deviceQuery
make EXTRA_NVCCFLAGS=--cudart=shared
```

如果有InfiniBand网络，rCUDA用户可以通过高通信传输性能的InfiniBand Verbs API替代TCP/IP协议。

### 运行程序

```
./deviceQuery

```

#### 遇到的问题


# 下一步计划

优化虚拟机，TCP/IP速度与virtio。数据代码签名。

# 参考文献
[1] [Ubuntu 16.04 CUDA 8 cuDNN 5.1安装](http://blog.csdn.net/jhszh418762259/article/details/52958287)
[2] [ldconfig命令](http://man.linuxde.net/ldconfig)