---
title: cuda wiki
date: 2019-07-31 19:16:45
tags:
- CUDA
- Memory
categories:
- [GPU,CUDA]
- [wiki]
---

本片博客收录了cuda学习过程中的遇到的知识点总结。  

<!-- more -->  

主要参考 <https://github.com/yszheda/wiki/wiki/CUDA> 的形式，以链接的形式记录，如果有需要便去链接的网页搜索。  

# CUDA Programming

## structure

+ [Why does CUDA CUdeviceptr use unsigned int instead of void?](https://stackoverflow.com/a/18141906)  
> CUdeviceptr is a handle to an allocation in device memory and not an address in device memory.  



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

# PTX

+ [The reference guide for inlining PTX ](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#abstract)  
+ 


# CUDA Snippets


+ [CUDA Code Snippets](https://github.com/yszheda/wiki/wiki/CUDA-Code-Snippets)
+ [CUDA Samples](https://github.com/huoyao/cudasdk)
