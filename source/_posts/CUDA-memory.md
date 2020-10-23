---
title: CUDA内存介绍
date: 2018-12-21 18:27:11
tags:
- CUDA
- Memory
categories:
- [GPU,CUDA]
---

CUDA内存分类和变量类型限定符总结。  
<!-- more -->

# 内存类型

+ Global memory
device memory  
slow  
+ Texture memory (read only)  
device memory  
cache in *texture cache*  
通用计算没有用到Texture内存。  
+ Constant memory
device memroy,用于存储  constants 和 kernel arguments。   
slow, cached in *constant cache* 
+ Shared memory
on-chip memory，用于block中的thread交换数据。  
fast, 但是需要处理 bank conflicts  
+ Local memory  
device memory  
slow  
在计算能力3.x的GPU上，local memory cached in *L1* 和 *L2*。  
在计算能力5.x 和 6.x的GPU上，local memory cached in *L2*。  
+ Registers  
on-chip memory  
fast  



# 片上内存

    
![硬件模型](../CUDA-memory/hardware-model.png)

[3.3. On-chip Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#on-chip-shared-memory) 介绍SMX片上共享内存包括：  
+ 每个SP上有一组32位的寄存器
+ SPs共享的数据缓存`shared memory`
+ SPs共享的只读内存`constant cache`
+ SPs共享的只读内存`texture cache`



# 类型限定符标  


|变量声明| Memory | Scope | Lifetime| Performance Penalty| 
|-|-|-|-| - |
| `int LocalVar;`          | register  | thread | thread|  1x |
| `int LocalArray[10];`    | local     | thread | thread|   100x |
| `[__device__] __shared__ int SharedVar;` | shared | block | block| 1x |
| `__device__ int GlobalVar;` | global | grid | application| 100x| 
| `[__device__] __constant__ int ConstantVar;` | constant | grid | application| 1x | 

函数限定符包括 `__global__` `__device__` `__host__`，而变量内存限定符包括： `__device__` `__constant__` `__shared__`。  

`__global__`和 `__constant__` 在 kernel 函数外部声明。  
register、`__shared__` 、local 变量在 kernel 函数内部声明。  


对于变量内存限定符，没有限定符的普通变量（Automatic variables）都在register中，只在当前kernel中的当前thread有效。  
arrays变量在local memory中，或者超过register总数的普通变量存储在local memory，这称为 *register spilling*；再者就是太耗register的结构体或array。local变量也只在当前kernel的当前thread有效。    
这可以对应着ptx文件查看。  
local memory变量使用`.local` 助记符（mnemonic） 声明，使用 `ld.local` 和 `st.local` 助记符操作。  
可以通过 cuobjdump 查看 cubin object或者通过nvcc编译器的 *--ptxas-options=-v* 选项确认每个kernel的local memory使用情况(lmem)。  

`__device__` 声明了在global memory中的变量，在整个CUDA context生命周期都可使用。`__device__`变量能够被grid的所有threads访问，也能被host通过runtime library访问。  

`__constant__`声明了在constant memory中的变量，在整个CUDA context生命周期都可使用。`__constant__`变量能够被grid的所有threads访问，也能被host通过runtime library访问。  

`__shared__`变量在一个block的shared memory中，只在当前kernel的当前block有效，只能被当前block的thread访问。  

指针只能对Global memory使用。  


# 参考
CUDA C PROGRAMMING GUIDE: 5.3.2. Device Memory Accesses   
CUDA C PROGRAMMING GUIDE: Appendix B. C LANGUAGE EXTENSIONS  
[CUDA学习笔记九](https://blog.csdn.net/langb2014/article/details/51348616)
[CUDA Tutorial](https://jhui.github.io/2017/03/06/CUDA/)
[Access CUDA global device variable from host](https://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host#)  
[Constant Memory vs Texture Memory vs Global Memory in CUDA](https://stackoverflow.com/questions/8306967/constant-memory-vs-texture-memory-vs-global-memory-in-cuda)  
[一篇介绍CUDA Memory的好文档](http://www.cvg.ethz.ch/teaching/2011spring/gpgpu/cuda_memory.pdf)  
[Access CUDA global device variable from host](https://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host#)
