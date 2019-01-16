---
title: nouveau资料整理
date: 2018-07-18 14:18:35
tags:
- gpu
- nouveau
- linux
- CUDA
categories:
- GPU
---
nouveau是LINUX内核中NVIDIA显卡的开源驱动，但是它不对CUDA支持，熟悉nouveau对于加强掌握NVIDIA GPU显卡有极大的帮助。本文整理了阅读到的nouveau资料。
<!-- more -->

首先[nouveau官网](https://nouveau.freedesktop.org/wiki/)介绍它是NVidia显卡的加速开源驱动。

要认真读一遍[nouveau的wikipedia介绍](https://en.wikipedia.org/wiki/Nouveau_(software))，它详细介绍了发展历史和支持的软件。

先来学习下入门阶段的[介绍](https://nouveau.freedesktop.org/wiki/IntroductoryCourse/)


# development

[nouveau开发](https://nouveau.freedesktop.org/wiki/Development/)也提供了不少的有价值的资料。

## CodeNames
[NVIDIA显卡的代号CodeNames](https://nouveau.freedesktop.org/wiki/CodeNames/)
比如我使用的 `GeForce GTX Titan Black` 的 Codename就是 `NVF1 (GK110B)`。  

## [riva128.txt](https://github.com/Emu-Docs/Emu-Docs/blob/master/PC/GPUs/nVidia/Riva%20128/riva128.txt)

比较老的一个介绍旧显卡的文档，但是阐明了内部的运作。  

## ContextSwitching

[ContextSwitching](https://nouveau.freedesktop.org/wiki/ContextSwitching/) 上下文切换的重要性和如何切换。

硬件上下文指的是显卡硬件的当前状态，即GPU寄存器和命令FIFO等。   
NVidia显卡提供多个命令通道（Command Channels），每个通道与给定的硬件环境相关联。 这意味着，在使用所有通道之前，每个图形客户端将在显卡上拥有其自己的通道和硬件上下文。

上下文之间的切换的方式在所有显卡中不总是一样的。 最新的Nvidia卡自动完成，但需要特殊的初始化，而较旧的则需要驱动程序自行处理。  

在NV10之前，上下文切换由驱动程序完成并且是中断驱动的：每当显卡在当前未激活的通道上获取命令时，它将发送PGRAPH中断（PGRAPH中断是由显卡的图形引擎发送的中断） 到驱动程序，驱动必须保存显卡的寄存器，并恢复新的上下文。  

从NV20开始，上下文切换由GPU在硬件上完成，从NV40开始计算，这些卡需要一个特殊的微代码，称为ctxprogs。

上下文切换现在适用于所有卡。 对于需要ctxprog（NV4x +）的卡，我们曾经复制过专有驱动程序发送的ctxprog，但现在已经为它们编写了ctxprog生成器。

[NVC0显卡的上下文切换固件](https://nouveau.freedesktop.org/wiki/NVC0_Firmware/)

## fence

[GL_NV_fence](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_fence.txt) 解释了 `fence` 。  

`fence` 是 DRM的TTM的重要概念，本质上是一种管理CPU和GPU之间并发的机制。   当GPU不再使用缓冲区对象时，`fence` 会跟踪，通常用于通知任何用户空间进程可以访问此缓冲对象。


# 向上支持的接口

nouveau向用户态支持的库包括 图形渲染API： Mesa 3D、 OpenGL； 计算API：OpenCL、 CUDA。

![DRM, KMS driver, libDRM, Mesa 3D等结构图，来自wiki](../nouveau资料整理/Linux_Graphics_Stack_2013.svg)

从图中可以看出，nouveau集成到了DRM的驱动和用户态上。

## CUDA支持

## Coriander

Nouveau本身不支持 CUDA，但是 [Coriander 项目：Build applications written in NVIDIA® CUDA™ code for OpenCL™ 1.2 devices] 在OpenCL 1.2 上支持CUDA，但是需要使用项目提供的编译器。 Coriander一直在维护，github地址在 <https://github.com/hughperkins/coriander>。

## Gdev

发表在顶会 **USENIX ATC'12** 上的项目[Gdev：Open-Source GPGPU Runtime and Driver Software](https://github.com/shinpei0208/gdev) ，为NVIDIA GPGPU提供了驱动和运行时库的开源的支持。此项目可以运行在nouveau上。  
Gdev停止更新在2014年。


# nouveau 代码

nouveau由两个内核模块 DRM和 KMS驱动组成，和调用用户空间的libdrm， Mesa 3D。

## nouveau 代码的地址
+ Linux-4.4 内核代码中 nouveau代码 <https://elixir.bootlin.com/linux/v4.4.169/source/drivers/gpu/drm/nouveau> 
+ nouveau github 持续更新代码: <https://github.com/skeggsb/nouveau>
+ Linux-4.4 nouveau 更新日志： <https://cgit.freedesktop.org/nouveau/linux-2.6/log/?h=linux-4.4>
+ 用户态libdrm中的nouveau代码： <https://github.com/tobiasjakobi/libdrm/tree/exynos/nouveau> ，libdrm版本 <https://dri.freedesktop.org/libdrm/> 。
  

如果要深入阅读代码，需要记住很多结构体和函数，网上没有什么代码解析的博客，官网也没有什么补充材料，只能靠自己阅读。  


补充一些材料。

## NVKM short for NVIDIA Kernel Module

[Nouveau In Linux 3.20 Will Have A Lot Of Code Cleaning](https://www.phoronix.com/scan.php?page=news_item&px=Nouveau-Linux-3.20) 提到 Linux-3.20 中的更新情况，引入了 `NVKM` 明明空间，用 `nvkm_*` 代替 `nouveau_*` 。

> drm/nouveau: finalise nvkm namespace switch (no binary change)linux-3.20
The namespace of NVKM is being changed to nvkm_ instead of nouveau_,
which will be used for the DRM part of the driver.  This is being
done in order to make it very clear as to what part of the driver a
given symbol belongs to, and as a minor step towards splitting the
DRM driver out to be able to stand on its own (for virt).

> Because there's already a large amount of churn here anyway, this is
as good a time as any to also switch to NVIDIA's device and chipset
naming to ease collaboration with them.


## 

# IRC #nouveau

此频道所有的日志文件存放在了 <https://people.freedesktop.org/~cbrill/dri-log/index.php> 中。

<http://webchat.freenode.net/> 频道为 #nouveau



# 参考
1. [Nouveau源码分析(零):前言、目录](https://blog.csdn.net/GoodQt/article/details/40681007)
2. [nvidia gpu open doc](http://download.nvidia.com/open-gpu-doc/)
