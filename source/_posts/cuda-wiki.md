---
title: cuda wiki
date: 2019-07-31 19:16:45
tags:
- CUDA
categories:
- [GPU,CUDA]
- [wiki]
---

本片博客收录了cuda学习过程中的遇到的知识点总结。  

<!-- more -->  

主要参考 <https://github.com/yszheda/wiki/wiki/CUDA> 的形式，以链接的形式记录，如果有需要便去链接的网页搜索。  

# CUDA Reading List

跟着 <https://github.com/yszheda/wiki/wiki/CUDA-Reading-List> 学就完了。  

+ <http://docs.nvidia.com/cuda/cuda-c-best-practices-guide>

# CUDA Programming

## modular arithmetic

formule like *(a*b - c*d) mod m or (a*b + c) mod m *.  
use double-precision arithmetic to avoid expensive div and mod operations.  

+ [modular arithmetic on the gpu](https://stackoverflow.com/questions/12252826/modular-arithmetic-on-the-gpu)  
+ [Using the modulo (%) operator in CUDA 65536](https://www.beechwood.eu/using-the-modulo-operator-in-cuda-65536/)  

这里总结了单双精度浮点数的区别：
+ [What's the difference between a single precision and double precision floating point operation?](https://stackoverflow.com/a/801146)  
+ 

## CUDA structure

+ [Why does CUDA CUdeviceptr use unsigned int instead of void?](http://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/)  
+ [Why does CUDA CUdeviceptr use unsigned int instead of void?](https://stackoverflow.com/a/18141906)  
> CUdeviceptr is a handle to an allocation in device memory and not an address in device memory.  

### volatile

编译器会自动优化对global和shared memory的读写，比如将global内存变量缓存到register或者 L1 Cache。  
volatile关键字阻止编译器优化，编译器会认为被volatile声明过的变量可能随时会被其他线程访问或修改。  


+ [When to use volatile with shared CUDA Memory](https://stackoverflow.com/a/15331158)
+ [一个volatile引发的CUDA程序的血案](https://baiweiblog.wordpress.com/2017/12/06/cuda-reduce-unroll-the-last-warp%E7%9A%84%E4%B8%80%E4%B8%AA%E6%98%93%E7%8A%AF%E7%9A%84%E9%94%99%E8%AF%AF/)
+ [Warp Synchrony and The First Law of CUDA Development](http://www.cudahandbook.com/2017/05/warp-synchrony-and-the-first-law-of-cuda-development/)


## warp

+ [How do CUDA blocks/warps/threads map onto CUDA cores?](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)  
+ 

## CUDA runtime API

### cudaMemcpyAsync

+ [Effect of using page-able memory for asynchronous memory copy?](https://stackoverflow.com/a/14094079)  

`cudaMemcpyAsync` 是 `cudaMemcpy` 的异步版本。若满足以下两个条件：
    - 使用non-default stream
    - host memory是pinned allocation。  
GPU会分配一个free DMA copy engine，效果就是拷贝过程可以和其他GPU操作同步，比如kernel执行或者另一个拷贝（假如GPU有多个DMA copy engine的话）。  
如果两个条件不能同时满足的话，GPU上的操作和 `cudaMemcpy` 是一致的，只不过它不会阻塞host。  

也就是说， `cudaMemcpyAsync` 不一定使用创建的流和锁页内存。  

## MPS

### MPS介绍

volta及以后的架构对MPS做了基于硬件加速的实现，并对进程做了地址空间隔离，这样进一步减少kernellaunch带来的延迟。Volta下的MPS服务最多可以允许同时48个Client（客户端）。  
+ [Multi-Process Scheduling翻译文](https://cloud.tencent.com/developer/article/1081424)  

### MPS使用  
+ [How do I use Nvidia Multi-process Service (MPS) to run multiple non-MPI CUDA applications?](https://stackoverflow.com/a/34711344)  

start the MPS server:  

```
#!/bin/bash
# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```

stop the MPS server:  
```
#!/bin/bash
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
```

## IPC

+ [cuda-c-programming-guide-Interprocess Communication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)
+ [CUDA initialization error after fork](https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork)  
+ 

## pthreadMigration

pthread只能绑定一个context，而一个设备，多个pthread内创建的context是一致的。  
pthread不能被MPS加速。  

> Simple sample demonstrating multi-GPU/multithread functionality using
> the CUDA Context Management API.  This API allows the a CUDA context to be 
> associated with a CPU process.  CUDA Contexts have a one-to-one orrespondence
> with host threads.  A host thread may have only one device context current
> at a time.

代码见[threadMigration.cpp](https://github.com/huoyao/cudasdk/blob/master/6_Advanced/threadMigration/threadMigration.cpp)。  


## get SM-ID in cuda thread

```
static __device__ __inline__ uint32_t __mysmid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}
```

> %smid and %warpid are defined as volatile values

+ [How can I find out which thread is getting executed on which core of the GPU?](https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu)
+ [which SM a thread is running?](https://devtalk.nvidia.com/default/topic/481465/cuda-programming-and-performance/any-way-to-know-on-which-sm-a-thread-is-running-/2)
+ [sm-id and warp-id](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#sm-id-and-warp-id)

## Dynamic Parallelism

dynamic kernel creation

+ [find CUDA DYNAMIC PARALLELISM in CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
+ <https://www.cnblogs.com/1024incn/p/4557156.html>
+ [CUDA Dynamic Parallelism, bad performance](https://stackoverflow.com/questions/45201062/cuda-dynamic-parallelism-bad-performance)

# PTX

+ [The reference guide for inlining PTX ](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#abstract)  
+ 


# CUDA Snippets


+ [CUDA Code Snippets](https://github.com/yszheda/wiki/wiki/CUDA-Code-Snippets)
+ [CUDA Samples](https://github.com/huoyao/cudasdk)

# review

+ [Interview questions on CUDA Programming?](https://stackoverflow.com/questions/1958320/interview-questions-on-cuda-programming)  
    + How many different kind of memories are in a GPU ?
    + What means coalesced / uncoalesced?
    + Can you implement a matrix transpose kernel?
    + What is a warp ?
    + How many warps can run simultaneously inside a multiprocessor?
    + What is the difference between a block and a thread ?
    + Can thread communicate between them? and blocks ?
    + Can you describe how works a cache?
    + What is the difference between shared memory and registers?
    + Which algorithms perform better on the gpu? data bound or cpu bound?
    + Which steps will you perform to port of an application to cuda ?
    + What is a barrier ?
    + What is a Stream ?
    + Can you describe what means occupancy of a kernel?
    + What means structure of array vs array of structures?

+ [Nvidia Interview | Set 1](https://www.geeksforgeeks.org/nvidia-interview-set-1/)  
+ 
