---
title: CUDA内存介绍
date: 2018-12-21 18:27:11
tags:
- CUDA
- Memory
categories:
- [GPU,CUDA]
---

CUDA内存使用总结。  
<!-- more -->


# 片上内存

    
![硬件模型](../CUDA内存介绍/hardware-model.png)

[3.3. On-chip Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#on-chip-shared-memory) 介绍SMX片上共享内存包括：  
+ 每个SP上有一组32位的寄存器
+ SPs共享的数据缓存`shared memory`
+ SPs共享的只读内存`constant cache`
+ SPs共享的只读内存`texture cache`



# `__device__ ` 声明的变量无法从host上成功访问    

[Access CUDA global device variable from host](https://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host#)




# 参考
1. [CUDA学习笔记九](https://blog.csdn.net/langb2014/article/details/51348616)
2. [CUDA Tutorial](https://jhui.github.io/2017/03/06/CUDA/)
3. [Access CUDA global device variable from host](https://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host#)
