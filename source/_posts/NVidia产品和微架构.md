---
title: NVidia产品和微架构
date: 2018-01-13 15:32:42
tags:
- GPU
- GK110
categories:
- GPU
---

本篇博客介绍NVidia显卡的产品类型和微架构。尤其NVidia的Tesla系列产品和Tesla微架构区分清楚。
<!--more-->
# NVidia（英伟达）GPU产品

NVidia推出的GPU产品和GPU架构总是搞混，这里列举一下目前产品，早期的就不算进来了。这里参考维基百科。

|产品|系列|作用|
|-----|----|----|
|个人电脑|GeForce系列|分为桌面平台与移动平台，按系列分类，其中GeForce 256与GeForce 3没有推出移动平台产品。桌面平台包括GeForce 2、GeForce 3至GeForce 9；GeForce 100至GeForce 700、GeForce 900、GeForce 10。举个例子，我台式机用的是GeForce GTX Titan Black就是GeForce 700系列产品。移动平台此系列主要应用到笔记本电脑上的显卡，一般后面带个`M`或其他标识，我15年买的Acer笔记本带的显卡是GeForce GTX 960M ，开发代号GM107  。但是GeForce 10系列就没有后缀，比如GeForce GTX 1080。|
|工作站| Quadro系列|分为桌面平台与移动平台，按系列分类。|
|服务器| Tesla系列|利用图形处理器进行高性能运算，部分型号无显示输出接头。|
|手持设备|GoForce与Tegra系列|Tegra(图睿)是系统单片机，替代GoForce系列。应用于智能手机、便携式媒体播放器和平板电脑等。每个 Tegra 内置ARM架构的处理器核心、基于GeForce的图形处理器等。|
|电子游戏机|无|为电子游戏机设计的图形处理器。|

# NVidia GPU微架构

GPU的微架构（micro-architecture）和GPU的计算能力（compute capability）挂钩。参考维基百科[CUDA](https://en.wikipedia.org/wiki/CUDA)


|计算能力|微架构|GPU核代|代表|
|--------|------|-------|----|
|1.0|Tesla|G80|GeForce 8800 Ultra|
|1.1|Tesla|G92, G94, G96, G98, G84, G86|GeForce GTS 250, Quadro FX 4700 X2|
|1.2|Tesla|GT218, GT216, GT215|GeForce GT 340*, GeForce GT 330*,Quadro FX 380 Low Profile|
|1.3|Tesla|GT200, GT200b|GeForce GTX 295, Quadro FX 5800, Tesla C1060|
|2.0|Fermi|GF100, GF110|GeForce GTX 590, GeForce GTX 580,Quadro 6000,Tesla C2075|
|2.1|Fermi|GF104, GF106 GF108, GF114, GF116, GF117, GF119|GeForce GTX 560 Ti, GeForce GTX 550 Ti, Quadro 2000, Quadro 2000D|
|3.0|Kepler|GK104, GK106, GK107|GeForce GTX 770, GeForce GTX 760,Quadro K5000,Tesla K10|
|3.2|Kepler|GK20A|Tegra K1, Jetson TK1|
|3.5|Kepler|GK110, GK208|GeForce GTX Titan Z, GeForce GTX Titan Black, GeForce GTX Titan, GeForce GTX 780 Ti,Quadro K6000, Tesla K40|
|3.7|Kepler|GK210|Tesla K80|
|5.0|Maxwell|GM107, GM108|GeForce GTX 750 Ti, Quadro K1200, Quadro K620, Quadro M2000M, Tesla M10|
|5.2|Maxwell|GM200, GM204, GM206|GeForce GTX Titan X, GeForce GTX 980 Ti, Quadro M3000M, Tesla M4, Tesla M40|
|5.3|Maxwell|GM20B|Tegra X1, Jetson TX1,|
|6.0|Pascal|GP100|Quadro GP100, Tesla P100|
|6.1|Pascal|GP102, GP104, GP106, GP107, GP108| Titan X, GeForce GTX 1080 Ti,Tesla P40, Tesla P6, Tesla P4,Quadro P6000|
|6.2|Pascal|GP10B|Drive PX2 with Tegra X2 |
|7.0|Volta|GV100|NVIDIA TITAN V, Tesla V100|

总体来说，Tesla架构的GPU计算能力为1.\*, Fermi架构的GPU计算能力为2.\*，Kepler架构的GPU计算能力为3.\*，Maxwell架构的GPU的计算能力为5.\*，Pascal架构的GPU计算能力为6.\*，Volta架构的GPU计算能力为7.\*。

大概来说，每个系列的产品都会升级自己的计算能力，而每项计算能力都包括若干不同系列的产品。

更详细的产品，计算能力参见<https://developer.nvidia.com/cuda-gpus>。

# 微架构

## Fermi


[NVIDIA’s Next Generation CUDA Compute Architecture: Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf)

## Kepler

`Kepler GK110` 支持 `Compute Capability 3.5`，有15个 `SMX` 和 6个 64位的内存控制器。

每个 `SMX` 单元有192个单精度核，每个核有单精度和整数逻辑运算单元。


| |FERMI GF100 | FERMI GF104 | KEPLER GK104 | KEPLER GK110|
|-| -----------|-------------|--------------| ------------|
|Compute Capability | 2.0 | 2.1 | 3.0 | 3.5 |
|Threads / Warp | 32 | 32 | 32 | 32|
|Max Warps / Multiprocessor | 48 | 48 | 64 | 64|
|Max Threads / Multiprocessor | 1536 | 1536 | 2048 | 2048|
|Max Thread Blocks / Multiprocessor | 8 | 8 | 16 | 16|
|32‐bit Registers / Multiprocessor | 32768 | 32768 | 65536 | 65536|
|Max Registers / Thread | 63 | 63 | 63 | 255|
|Max Threads / Thread Block | 1024 | 1024 | 1024 | 1024|
|Shared Memory Size Configurations (bytes) | 16K 48K| 16K 48K | 16K 32K 48K | 16K 32K 48K|
|Max X Grid Dimension | 2^16‐1 | 2^16‐1 | 2^32‐1 | 2^32‐1|
|Hyper‐Q  | No | No | No | Yes|
|Dynamic Parallelism | No | No | No | Yes|



[NVIDIA’s Next Generation CUDA Compute Architecture: Kepler GK110](https://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf)


## Maxwell

[NVIDIA GeForce GTX 980](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_980_Whitepaper_FINAL.PDF)

## Pascal

[NVIDIA Tesla P100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)