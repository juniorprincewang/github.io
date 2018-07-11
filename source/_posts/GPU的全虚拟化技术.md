---
title: GPU的全虚拟化技术
date: 2018-06-18 16:56:05
tags:
- cuda
categories:
- GPU
- GPU虚拟化
---

GPU的全虚拟化需要模拟GPU driver，而各大GPU厂商对driver闭源，这只能通过网上公开的逆向工作来解开driver的面具。
<!-- more -->

# nouveau

NVIDIA 显卡的开源驱动

可以通过一下几种方式来逆向NVIDIA显卡。

- `MMIOtrace` 在PCIe 总线上监听传输。
- `valgrind-mmt` 追踪发送到显卡的命令
- `envytools` 来帮助逆向驱动


# gdev


# envytools

