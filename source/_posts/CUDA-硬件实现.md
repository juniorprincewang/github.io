---
title: CUDA 硬件实现
date: 2018-01-12 14:22:37
tags:
- CUDA
- GPU
- SIMT
---
本篇博客介绍CUDA的硬件实现，SM与SP。
<!--more-->

# GPU物理层

NVidia GPU的流处理器（Stream Multiprocessors, SM）是GPU种非常重要的部分，GPU的并行性是由SM决定的。
以Fermi架构为例，主要组成部分如下:

+ CUDA cores，执行单元
+ Shared Memory/L1Cache，共享内存和一级Cache
+ Register File
+ Load/Store Units
+ Special Function Units,特殊函数单元（SFU），用以计算log/exp，sin/cos，rcp/rsqrt的单精度近似值；
+ Warp Scheduler,一个线程束调度器，用以协调把指令分发到执行单元；



# CUDA基本概念

## 函数限定符

__device__ ：声明某函数在设备上执行，只能从设备中调用
__global__ ：声明某函数为内核(kernel)函数，在设备上执行，只能从宿主中调用
__host__ ：host声明某函数在宿主上执行，只能从宿主中调用

## 变量类型限定符

__constant__限定符与__device__结合使用，声明变量：
    驻留在常量内存空间中，具有应用程序的生命期，可通过运行时库被网格的所有线程访问，也可被宿主访问。
__shared__限定符可以与__device__结合使用，声明变量：
    驻留在线程块的共享内存空间中，具有块的生命期，仅可被块内的所有线程访问。

# 逻辑层

CUDA为了方便编程，提出了kernel、thread、block、grid、warp概念。
- kernel：kernel是CUDA C扩展C语言函数定义出来的函数，它可以被N个CUDA线程调用N次。
- thread: GPU程序执行的最小单位，每个线程拥有自己的程序计数器和状态寄存器，并且用自己的数据执行指令。
- block：一个block由3维空间的thread组成，同一个block中的thread可以同步，也可以通过shared memory通信。
- grid：一个grid再由3维空间的block组成。
- warp：GPU执行 程序的调度单位，目前cuda的一个warp由32个线程组成。

thread、block、grid、kernel的关系图：
![逻辑关系图](../CUDA-硬件实现/CUDA逻辑图.jpg)


