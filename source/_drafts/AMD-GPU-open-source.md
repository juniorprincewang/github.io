---
title: AMD GPU open-source资料整理
date: 2019-08-17 18:53:51
tags:
- GPU
- amd
- linux
categories:
- [GPU,amd]
---
AMD显卡逐渐拥抱开源，包括图形和通用计算。现集中整理。  
<!-- more -->


# specifications

xorg 保存了若干AMD开源的GPU设计白皮书。 <https://www.x.org/docs/AMD/old/>  

其中[Radeon R5xx Acceleration](https://www.x.org/docs/AMD/old/R5xx_Acceleration_v1.5.pdf)  给出了Command Processor的详细设计，而且Command Processor是可编程的。包括 Ring Buffer 的使用。    



# opensource driver

AMDGPU 在linux内核源码的 drivers/gpu/drm/amd/amdgpu 目录下。  

[AMD Open Source Driver for Vulkan](https://github.com/GPUOpen-Drivers/AMDVLK)  
[ROCm: open source platform for HPC GPU computing](https://rocmdocs.amd.com)

# blogs

关于 此白皮技术书的解读：  
[Linux环境下的图形系统和AMD R600显卡编程(1)——Linux环境下的图形系统简介]https://www.cnblogs.com/shoemaker/p/linux_graphics01.html()  

[AMD GPU任务调度（1）—— 用户态分析](https://blog.csdn.net/huang987246510/article/details/106658889)  
[AMD GPU任务调度（2）—— 内核态分析](https://blog.csdn.net/huang987246510/article/details/106737570)  
[AMD GPU任务调度（3）—— fence机制](https://blog.csdn.net/huang987246510/article/details/106865386)  
[!DRAW_INDEX与图形流水线](https://blog.csdn.net/huang987246510/article/details/107283374)  

