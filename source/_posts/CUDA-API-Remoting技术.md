---
title: CUDA API Remoting技术
date: 2018-05-14 16:28:02
tags:
- cuda
categories:
- GPU
- GPU虚拟化
---

CUDA的虚拟化有一项技术为 `API Remoting`， 

<!-- more -->



# GPU虚拟化

# API Remoting

设置 cuda的编译选项，使其调用库为动态调用。
`--cudart=shared`


## cuda kernel

位于 `/usr/local/cuda/include/crt/host_runtime.h`中的
```
__cudaRegisterFunction
__cudaUnregisterFatBinary
__cudaRegisterFatBinary
__cudaInitModule
```
其他函数：
```
cudaMalloc
cudaConfigureCall
cudaLaunch
cudaFree
cudaSetupArgument
cudaMemcpy

```



# 参考
[1] [what are the parameters for __cudaRegisterFatBinary and __cudaRegisterFunction functions?](https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f)
[2] []()