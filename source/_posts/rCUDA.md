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

1. CUDA在server服务器中成功运行。使用CUDA的deviceQuery和bandwidthTest样例来测试。
2. 确保client和server正常通信。
    1.  可以选择基于TCP/IP的通信（以太网）。
    2.  也可以选择基于RDMA的通信（InfiniBand或者RoCE）。使用Mellanox OFED的ib_write_bw和ib_read_bw测试IB或RoCE。

## 安装rCUDA

去官网下载，需要填写信息。<http://rcuda.net/index.php/software-request-form.html>

我在这里保存了一份[rCUDAv16.11.04.02-CUDA8.0-linux64.tgz](../rCUDA/rCUDAv16.11.04.02-CUDA8.0-linux64.tgz)，我的系统是64位Ubuntu16.04。

在client和server两端都需要rCUDA的这份文件。

### 开启rCUDA server

1. 首先设置环境变量。

```
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
```
如果是临时设置环境变量，那么就直接在终端里输入命令。如果想要永久设置可以有以下方法。

I. 修改/etc/profile文件
在文件中追加上述命令，此方法对所有用户都有效。
然后刷新。
II. 修改~/.bashrc
在文件中追加上述命令，对当前用户有效。
保护后为了及时生效。
```
source ~/.bashrc
```
验证有没有生效。
```
echo $PATH
```

2. 开启rCUDA server

```
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


# 参考文献
[1] [Ubuntu 16.04 CUDA 8 cuDNN 5.1安装](http://blog.csdn.net/jhszh418762259/article/details/52958287)