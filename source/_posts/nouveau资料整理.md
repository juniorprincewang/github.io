---
title: nouveau资料整理
date: 2018-07-18 14:18:35
tags:
- GPU
- nouveau
- linux
- CUDA
categories:
- [GPU,nouveau]
---
nouveau是LINUX内核中NVIDIA显卡的开源驱动，但是它不对CUDA支持，熟悉nouveau对于加强掌握NVIDIA GPU显卡有极大的帮助。本文整理了阅读到的nouveau资料。
<!-- more -->

首先[nouveau官网](https://nouveau.freedesktop.org/wiki/)介绍它是NVidia显卡的加速开源驱动。

要认真读一遍[nouveau的wikipedia介绍](https://en.wikipedia.org/wiki/Nouveau_(software))，它详细介绍了发展历史和支持的软件。

先来学习下入门阶段的[介绍](https://nouveau.freedesktop.org/wiki/IntroductoryCourse/)


# development

[nouveau开发](https://nouveau.freedesktop.org/wiki/Development/)也提供了不少的有价值的资料。   
[NVIDIA挤牙膏式的部分开源资料](http://download.nvidia.com/open-gpu-doc/)  
[NVIDIA挤牙膏式的部分开源资料github版](https://github.com/NVIDIA/open-gpu-doc)  

## CodeNames
[NVIDIA显卡的代号CodeNames](https://nouveau.freedesktop.org/wiki/CodeNames/)
比如我使用的 GeForce GTX Titan Black 的 Codename就是 `NVF1 (GK110B)`。  
Tegra X1 是 `NV110 family (Maxwell)` 的 `NV12B (GM20B)`。  
GeForce GTX (1070, 1080) 是 `NV134 (GP104)`。  


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

## Coriander

Nouveau本身不支持 CUDA，但是 [Coriander 项目：Build applications written in NVIDIA® CUDA™ code for OpenCL™ 1.2 devices] 在OpenCL 1.2 上支持CUDA，但是需要使用项目提供的编译器。 Coriander一直在维护，github地址在 <https://github.com/hughperkins/coriander>。

## Gdev

发表在顶会 **USENIX ATC'12** 上的项目[Gdev：Open-Source GPGPU Runtime and Driver Software](https://github.com/shinpei0208/gdev) ，为NVIDIA GPGPU提供了驱动和运行时库的开源的支持。提供了CUDA driver API，此项目可以运行在nouveau上。  
Gdev停止更新在2014年，因此代码支持到彼时的sm 3.5计算能力的Kepler架构的GTX 780。


# nouveau 代码

nouveau由两个内核模块 DRM和 KMS驱动组成，和调用用户空间的libdrm， Mesa 3D。

## nouveau 代码的地址

nouveau upstream repository 一直由 skeggsb 维护。  
+ nouveau 持续更新代码: <https://github.com/skeggsb/nouveau>  
+ Linux kernel 中的最新 nouveau 更新： <https://github.com/skeggsb/linux>  

+ 用户态libdrm中的nouveau代码： <https://github.com/tobiasjakobi/libdrm/tree/exynos/nouveau> ，libdrm版本 <https://dri.freedesktop.org/libdrm/> 。


其他：  
+ Linux-4.4 内核代码中 nouveau代码 <https://elixir.bootlin.com/linux/v4.4.169/source/drivers/gpu/drm/nouveau> 
+ Linux-4.4 nouveau 更新日志： <https://cgit.freedesktop.org/nouveau/linux-2.6/log/?h=linux-4.4>
  

如果要深入阅读代码，需要记住很多结构体和函数，网上没有什么代码解析的博客，官网也没有什么补充材料，只能靠自己阅读。  


补充一些材料。

## nouveau变量命名

[Nouveau In Linux 3.20 Will Have A Lot Of Code Cleaning](https://www.phoronix.com/scan.php?page=news_item&px=Nouveau-Linux-3.20) 提到 Linux-3.20 中的更新情况，引入了 `NVKM` 命名空间，函数用 `nvkm_*` 代替 `nouveau_*` 。
但是没有二进制文件改动，这是将DRM驱动划拨出来方便以后虚拟化。[改动见这里](https://cgit.freedesktop.org/nouveau/linux-2.6/commit/?h=linux-3.20&id=be83cd4ef9a2a56bd35550bf96146b7b837daf02)  
比如 `struct nouveau_mmu` 改动为 `struct nvkm_mmu`  
**nvkm** is short for **NVIDIA Kernel Module**  

> drm/nouveau: finalise nvkm namespace switch (no binary change)linux-3.20
The namespace of NVKM is being changed to nvkm_ instead of nouveau_,
which will be used for the DRM part of the driver.  This is being
done in order to make it very clear as to what part of the driver a
given symbol belongs to, and as a minor step towards splitting the
DRM driver out to be able to stand on its own (for virt).

> Because there's already a large amount of churn here anyway, this is
as good a time as any to also switch to NVIDIA's device and chipset
naming to ease collaboration with them.

而对应的 **nvif** 就是应该是 **NVIDIA InterFace**的缩写。  

GPU设备名称和芯片名称替换Nouveau自己的GPU名字。
如 *gk104* 替换 *nve0* 。  
其他engine或subdev 的命名规则。  

> sw: rename from software  
> msenc: rename from venc  
> gr: rename from graph  
> msppp: rename from ppp   
> ce: rename from copy   
> pm: rename from perfmon  
> sec: separate from cipher (formerly crypt)  
> mmu: rename from vmmgr  
> pmu: rename from pwr (power)  
> clk: rename from clock  

注： CE is DmaCopy
> uevent : user event
    >e.g.:  struct nouveau_event *uevent;


`oclass` : object class  
`sclass` : subclass, 或者称为 child

## libdrm  

内核的drm主要是为了实现图形的DRI硬件加速而服务的，通过提供一系列ioctls的操作，使得应用层的软件可以直接对显卡硬件操作。驱动实际使用drm是经过libdrm封装之后的接口。  
内核drm主要包括：vblank事件处理，内存管理，输出管理，framebuffer管理，命令注册，fence,suspend/resume 支持，dma服务等。  
用户空间程序可以使用DRM API来命令GPU进行硬件加速的3D渲染和视频解码以及GPGPU计算。

## mesa  

[mesa 3D](https://mesa3d.org/) 的介绍就一句话：  
> Open source implementations of OpenGL, OpenGL ES, Vulkan, OpenCL, and more!

其他可以阅读的资料：  

[mesa3D wikipedia](https://en.wikipedia.org/wiki/Mesa_(computer_graphics))：讲了Mesa3D的前生今世。  
[lago Toral 博客](https://blogs.igalia.com/itoral/)：介绍了很多mesa3d原理和linux图形栈的分析。  
[Linux图形系统和AMD显卡编程 系列教程](https://www.cnblogs.com/shoemaker/tag/AMD%E6%98%BE%E5%8D%A1/)    
[Gallium3D](https://gallium.readthedocs.io/en/latest/):Gallium3D是Mesa3D的一个非常重要组成部分  


[mesa3d 文档](https://docs.mesa3d.org/)  

[mesa source code tree](https://docs.mesa3d.org/sourcetree.html)   
[mesa 框架与目录结构](https://winddoing.github.io/post/39ae47e2.html)   


以10.1.4版本的Mesa3D为例，主要由mesa主模块、gallium模块、egl模块、glsl模块和glx等模块组成. 其中最重要的模块就是主模块，它主要包括mesa/和mapi/这两个folder. mapi/负责所有API的初始化工作，为各个API设置分发表等；mesa/则是整个Mesa3D图形库的核心，它负责的有vbo相关工作、非Gallium3D支持的驱动DRI实现以及软件实现渲染管道等等. gallium模块则是Mesa3D整合原开源项目Gallium的结果，主要实现驱动无关的硬件加速. egl模块用来实现EGL库，glsl模块用来实现GLSL编译器，glx模块用来实现GLX库.


[Mesa & Gallium3D 介绍](https://juejin.im/post/5cd40e35f265da039f0f2b3c)  

### OpenGL  

OpenGL API 是定义了一个跨编程语言、跨平台的应用程序接口(API)的规范，它用于生成2D和3D图像，而它仅仅是定义了一种API，并没有任何实现细节。 
而OpenGL API的具体实现有很多，主要分为开源实现和闭源实现， 闭源实现如各大GPU厂商自己实现的闭源OpenGL图形库，例如AMD显卡的Catalyst闭源驱动；而开源实现便是Mesa3D。

### Gallium3D  

[Gallium3D Technical Overview from freedesktop](https://www.freedesktop.org/wiki/Software/gallium/)  
[Gallium3D Documentation](https://dri.freedesktop.org/doxygen/gallium/index.html)  

Mesa的框架决定了它驱动开发的复杂性：每个显卡厂商的3D驱动都有各自不同的应用后端，通过此调用 Mesa 的 API 来实现 3D 加速。
Intel、AMD 和 NVIDIA 这三大厂商的显卡都具备各自不同的应用后端，造成了开发和维护困难。  

Gallium3D 提供一套统一的 API，这套API将标准的硬件特性（而非软件特性）暴露出来（如shader units），也就是说，Gallium3D 直接与统一的硬件级特性打交道，而非充当一个纯软件层。

因此，这些 API 使得 OpenGL 1.x/2.x，OpenGL 3.x，OpenVG，GPGPU 架构甚至 Direct3D 的实现，都只需要通过一个单独的后端即可。而无须各个厂商自行开发各自不同的后端。

这不仅让开发和维护显示驱动带来了极大的方便，而且统一的 API 使得 Mesa的灵活性和扩展性大大增强。  

Gallium3D 的目的：

+ make driver smaller and smaller
+ model modern graphics hardware
+ support multiple graphics API's
+ Support multiple operating systems


# IRC #nouveau

此频道所有的日志文件存放在了 <https://people.freedesktop.org/~cbrill/dri-log/index.php> 中。

<http://webchat.freenode.net/> 频道为 `#nouveau`  

如果要发言，需要提前注册下，[IRC 账号注册](https://freenode.net/kb/answer/registration) 。  

# yuzu emulator  

[yuzu](https://github.com/yuzu-emu/yuzu) 是 Nintendo Switch 的开源模拟器。  
Nintendo Switch console 使用的显卡是 Tegra X1，Maxwell架构，经过开源社区nouveau和yuzu的不懈努力，已经将其成功虚拟化。  
yuzu使用OpenGL 和 Vulkan 两种图形API实现。  
这对于研究maxwell架构的GPU又进一步提供了资料:D。  

# 参考
1. [Nouveau源码分析(零):前言、目录](https://blog.csdn.net/GoodQt/article/details/40681007)
2. [nvidia gpu open doc](http://download.nvidia.com/open-gpu-doc/)
