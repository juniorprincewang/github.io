---
title: CUDA 介绍
date: 2018-01-12 14:22:37
tags:
- CUDA
- GPU
- SIMT
- mmap
categories:
- GPU

---
本篇博客介绍CUDA的实现，包括物理和逻辑，内存结构等。
<!--more-->




# GPU物理层


NVidia GPU的流处理器（Stream Multiprocessors, SM）是GPU种非常重要的部分，GPU的并行性是由SM决定的。
以Fermi架构为例，主要组成部分如下:

+ CUDA cores，执行单元
+ Shared Memory/L1Cache，共享内存和一级Cache
+ Register File
+ Load/Store Units
+ Special Function Units: 特殊函数单元（SFU），用以计算log/exp，sin/cos，rcp/rsqrt的单精度近似值；
+ Warp Scheduler：一个线程束调度器。



# CUDA基本概念

## 函数限定符

`__device__` ：声明某函数在设备上执行，只能从设备中调用
`__global__` ：声明某函数为内核(kernel)函数，在设备上执行，只能从宿主中调用
`__host__` ：host声明某函数在宿主上执行，只能从宿主中调用

## 变量类型限定符

`__constant__` 限定符与 `__device__` 结合使用，声明变量：
    驻留在常量内存空间中，具有应用程序的生命期，可通过运行时库被网格的所有线程访问，也可被宿主访问。
`__shared__` 限定符可以与 `__device__` 结合使用，声明变量：
    驻留在线程块的共享内存空间中，具有块的生命期，仅可被块内的所有线程访问。

# 逻辑层

CUDA为了方便编程，提出了 `kernel` 、 `thread` 、 `block` 、 `grid` 、 `warp` 概念。
- `kernel` : 是CUDA C扩展C语言函数定义出来的函数，它可以被N个CUDA线程调用N次。
- `thread` : GPU程序执行的最小单位，每个线程拥有自己的程序计数器和状态寄存器，并且用自己的数据执行指令。
每个线程可以有自己独立的 `指令寄存器` 、 `寄存器状态` 、 `独立的执行路径` 。

- `block` ：一个block由3维空间的thread组成，同一个block中的thread可以同步，也可以通过shared memory通信。
- `grid` ：一个grid再由3维空间的block组成。
- `warp` ：GPU执行 程序的调度单位，目前cuda的一个warp由32个线程组成。
`warp` 包含32个线程，用以协调把指令分发到执行单元，是调度和运行的基本单位。 `warp` 中的所有 `threads` 并行执行相同的指令。
一个 `warp` 只能分配到一个 `SM` 运行， 一个 `SM` 可以同时允许多个 `warp` 执行。

`thread` 、 `block` 、 `grid` 、 `kernel` 的关系图：
![逻辑关系图](../CUDA-硬件实现/CUDA逻辑图.jpg)



# 内存层次

## global memory

## local memory

## shared memory

`shared memory` 按照线程块（block）划分， 其上的数据可以为同一 `block` 中的所有线程共享。
每个 `warp` 的 `shared memory` 大小是 `64KB` , 这个和 `L1 cache` 共用。、
按照 16KB L1 / 48KB shared 或者 48KB L1 / 16KB shared 划分。
 ([PixelVault])
同一个线程块中的线程可以通过共享内存互相通信，在逻辑上同一个线程块中的所有线程同时执行，但是在物理上，同一个线程块中的所有线程并不是同时执行的，所以同一个线程块中的线程并不是同时执行结束的。
共享内存可能会导致线程之间的竞争：多个线程同时访问某个数据。CUDA提供了线程块内的同步，保证同一个线程块中的线程在下一步执行前都完成了上一步的执行。但是**线程块**之间无法同步。


## register

GPU 寄存器提供了快速存取地址。但是寄存器数量有限

|Compute capability| #registers per thread|
|------------------|----------------------|
|1.x|128|
|2.x|63|
|3.x|63|
|3.5|255|


# 参考

