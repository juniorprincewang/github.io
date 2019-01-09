---
title: nouveau资料整理
date: 2018-07-18 14:18:35
tags:
- gpu
- nouveau
categories:
- GPU
---
nouveau是LINUX内核中NVIDIA显卡的开源驱动，熟悉nouveau对于加强掌握NVIDIA GPU显卡有极大的帮助。本文整理了阅读到的nouveau资料。
<!-- more -->

首先[nouveau官网](https://nouveau.freedesktop.org/wiki/)介绍它是nVidia显卡的加速开源驱动。

先来学习下入门阶段的[介绍](https://nouveau.freedesktop.org/wiki/IntroductoryCourse/)


# development

[nouveau开发](https://nouveau.freedesktop.org/wiki/Development/)也提供了不少的有价值的资料。

+ CodeNames
[NVIDIA显卡的代号CodeNames](https://nouveau.freedesktop.org/wiki/CodeNames/)
比如我使用的 `GeForce GTX Titan Black` 的 Codename就是 `NVF1 (GK110B)`。  

+ [riva128.txt](https://github.com/Emu-Docs/Emu-Docs/blob/master/PC/GPUs/nVidia/Riva%20128/riva128.txt)

比较老的一个介绍旧显卡的文档，但是阐明了内部的运作。  

+ ContextSwitching

[ContextSwitching](https://nouveau.freedesktop.org/wiki/ContextSwitching/) 上下文切换的重要性和如何切换。

硬件上下文指的是显卡硬件的当前状态，即GPU寄存器和命令FIFO等。   
NVidia显卡提供多个命令通道（Command Channels），每个通道与给定的硬件环境相关联。 这意味着，在使用所有通道之前，每个图形客户端将在显卡上拥有其自己的通道和硬件上下文。

上下文之间的切换的方式在所有显卡中不总是一样的。 最新的Nvidia卡自动完成，但需要特殊的初始化，而较旧的则需要驱动程序自行处理。  

在NV10之前，上下文切换由驱动程序完成并且是中断驱动的：每当显卡在当前未激活的通道上获取命令时，它将发送PGRAPH中断（PGRAPH中断是由显卡的图形引擎发送的中断） 到驱动程序，驱动必须保存显卡的寄存器，并恢复新的上下文。  

从NV20开始，上下文切换由GPU在硬件上完成，从NV40开始计算，这些卡需要一个特殊的微代码，称为ctxprogs。

上下文切换现在适用于所有卡。 对于需要ctxprog（NV4x +）的卡，我们曾经复制过专有驱动程序发送的ctxprog，但现在已经为它们编写了ctxprog生成器。


+ fence

[GL_NV_fence](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_fence.txt) 解释了 `fence` 。  

# IRC #nouveau
此频道所有的日志文件存放在了 <https://people.freedesktop.org/~cbrill/dri-log/index.php> 中。



# 参考
1. [Nouveau源码分析(零):前言、目录](https://blog.csdn.net/GoodQt/article/details/40681007)
2. [nvidia gpu open doc](http://download.nvidia.com/open-gpu-doc/)
