---
title: PCI driver model
date: 2018-07-09 20:12:10
tags:
- pci
categories:
- GPU
---
由于要研究NVIDIA GPU的工作原理，需要对PCIe的原理掌握清楚，在此整理了一些知识点，包括PCI配置空间和访问。
<!-- more -->

Peripheral Component Interconnect Bus(PCI) 是一种总线接口。
每个总线设备有256字节的配置内存，可通过 `CONFIG_ADDRESS` 和 `CONFIG_DATA` 寄存器获取。
设备驱动的开发人员仅需知道设备的基地址 `base address` 和设备使用的 `IRQ line` 。
PCI设备的配置信息是小端存储 `Little Endian` 。

Linux 下可以通过 `lspci` 查看PCI设备。

```
$ lspci
01:00.0 VGA compatible controller: NVIDIA Corporation GK110B [GeForce GTX TITAN Black] (rev a1)
|  |  |_Function Number
|  |_PCI Device Number
|_PCI Bus Number
```
三个数字分别是 01： Bus Number, 00: Device Number, 0: Function Number。

设备驱动的配置信息，可以通过 `lspci`的 选项 `-x`， `-xxx`， `-xxxx` 打印出来，不过要用 `root` 用户执行。


![450px-Pci-config-space.svg](../PCI-driver-model/450px-Pci-config-space.svg.png)

为了确定PCI设备的位置，PCI设备必须能够映射到系统的IO端口地址空间或者内存映射的地址空间。
系统的固件、设备驱动或操作系统编排BARs，通过将配置命令写入到PCI控制器中来通知设备的地址映射。






# 参考
1. [PCI configuration space](https://en.wikipedia.org/wiki/PCI_configuration_space)
2. [Access PCI device registry from Linux user space](https://falsinsoft.blogspot.com/2017/09/access-pci-device-registry-from-linux.html)
3. [lspci(8) - Linux man page](https://linux.die.net/man/8/lspci)
4. [Access physical memory from Linux user space](https://falsinsoft.blogspot.com/2013/10/access-physical-memory-in-linux.html)
5. [The anatomy of a PCI/PCI Express kernel driver](http://haifux.org/lectures/256/haifux-pcie.pdf)
6. [Linux PCI Driver Model](http://linuxkernel51.blogspot.com/2012/08/linux-pci-driver-model.html)
