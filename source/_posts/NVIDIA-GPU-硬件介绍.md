---
title: NVIDIA GPU 硬件介绍
date: 2018-07-14 16:52:26
tags:
- hardware
- gpu
categories:
- GPU
---
本文介绍NVIDIA GPU的硬件组成，为全面了解GPU的架构和逆向GPU结构，全虚拟化GPU提供硬件背景知识。
<!-- more -->


# nVidia GPU Hardware Introduction

## PCIe 应用程序编程

在 PCIe 配置空间里，0x10开始后面有6个32位的BAR寄存器，BAR寄存器中存储的数据是表示PCIe设备在PCIe地址空间中的基地址，注意这里不是表示PCIE设备内存在CPU内存中的映射地址，关于这两者的关系如下。

BAR寄存器存储的总线地址，应用程序是不能直接利用的，应用程序首先要做的就是读出BAR寄存器的值，然后用 `mmap` 函数建立应用程序内存空间和总线地址空间的映射关系。
这样应用程序往 PCIe 设备内存读写数据的时候，直接利用 PCIe 设备映射到应用程序中的内存地址即可。

读写 PCI 设备的具体代码参考[Simple program to read & write to a pci device from userspace](https://github.com/billfarrow/pcimem)。
利用的是 `sysfs` 设备文件 和 `mmap()` 函数。

首先找出PCI 映射的文件，比如在 `/sys/devices/pci0000\：00\ ` 中。

查找 `PCIe` 的设备文件也可以到 `/sys/bus/pci_express/devices/`  中。

查找 NVIDIA 的驱动：
```
ll /sys/devices/pci0000\:00/0000\:00\:01.0/0000\:01\:00.0/
总用量 0
drwxr-xr-x 12 root root         0 7月   5 11:46 ./
drwxr-xr-x  8 root root         0 7月   5 11:46 ../
-r--r--r--  1 root root      4096 7月   3 16:19 boot_vga
-rw-r--r--  1 root root      4096 7月   5 11:55 broken_parity_status
-r--r--r--  1 root root      4096 7月   3 16:19 class
-rw-r--r--  1 root root      4096 7月   3 16:19 config
-r--r--r--  1 root root      4096 7月   5 11:55 consistent_dma_mask_bits
-rw-r--r--  1 root root      4096 7月   5 11:55 d3cold_allowed
-r--r--r--  1 root root      4096 7月   3 16:19 device
-r--r--r--  1 root root      4096 7月   5 11:55 dma_mask_bits
lrwxrwxrwx  1 root root         0 7月   3 16:19 driver -> ../../../../bus/pci/drivers/nvidia/
-rw-r--r--  1 root root      4096 7月   5 11:55 driver_override
drwxr-xr-x  4 root root         0 7月   3 16:19 drm/
-rw-r--r--  1 root root      4096 7月   3 16:30 enable
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-0/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-1/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-2/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-3/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-4/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-5/
drwxr-xr-x  4 root root         0 7月   3 16:19 i2c-6/
-r--r--r--  1 root root      4096 7月   3 16:21 irq
-r--r--r--  1 root root      4096 7月   5 11:55 local_cpulist
-r--r--r--  1 root root      4096 7月   3 16:19 local_cpus
-r--r--r--  1 root root      4096 7月   5 11:49 modalias
-rw-r--r--  1 root root      4096 7月   5 11:55 msi_bus
drwxr-xr-x  2 root root         0 7月   4 08:38 msi_irqs/
-rw-r--r--  1 root root      4096 7月   3 16:19 numa_node
drwxr-xr-x  2 root root         0 7月   5 11:55 power/
--w--w----  1 root root      4096 7月   5 11:55 remove
--w--w----  1 root root      4096 7月   5 11:55 rescan
-r--r--r--  1 root root      4096 7月   3 16:19 resource
-rw-------  1 root root  16777216 7月   3 16:30 resource0
-rw-------  1 root root 134217728 7月   5 11:55 resource1
-rw-------  1 root root 134217728 7月   5 11:55 resource1_wc
-rw-------  1 root root  33554432 7月   5 11:55 resource3
-rw-------  1 root root  33554432 7月   5 11:55 resource3_wc
-rw-------  1 root root       128 7月   5 11:55 resource5
-rw-------  1 root root    524288 7月   3 16:19 rom
lrwxrwxrwx  1 root root         0 7月   3 16:19 subsystem -> ../../../../bus/pci/
-r--r--r--  1 root root      4096 7月   3 16:19 subsystem_device
-r--r--r--  1 root root      4096 7月   3 16:19 subsystem_vendor
-rw-r--r--  1 root root      4096 7月   3 16:19 uevent
-r--r--r--  1 root root      4096 7月   3 16:19 vendor
```

此显卡为 `GeForce GTX TITAN Black`, 设备ID为

```
$ cat device 
0x100c
```
*目前推测*： `BAR0`、`BAR1`、`BAR3`、`BAR5` 分别对应 `resource0`、 `resource1` 、 `resource3` 、 `resource5` 。


不过各个文件的代表意思，可以通过 [Accessing PCI device resources through sysfs](https://www.kernel.org/doc/Documentation/filesystems/sysfs-pci.txt) 了解下。

## PCI/PCIE/AGP bus interface and card management logic

###  PCI BARs and other means of accessing the GPU

####  Nvidia GPU BARs, IO ports, and memory areas

nvidia GPU通过PCI对外暴露了下面区域：

- PCI 配置空间 / PCIe 扩展配置空间
- MMIO 寄存器： BAR0 - 内存范围  0x1000000 字节或更多
通过MMIO寄存器控制所有引擎。
地址通过PCI BAR 0 来设置。 BAR使用32位地址，是非预取内存。

其中寄存器是32位的，读取时需要32位对齐。  在 NV1A+ 系列显卡中，寄存器的字节序列由PMC中的 开关（switch）控制。从显卡内部访问总是小端序列。

PMC是显卡master controller，尤其重要的MMIO空间的子区域，区域范围在 0x000000 到 0x000fff 之间，包括GPU id信息， Big Red Switch, master 中断控制。

- VRAM (on-board 内存)： BAR1 -内存范围 0x1000000 字节或者更多

这是映射了VRAM的预取内存。在PCIe卡上，使用64位地址；而在PCI卡上，使用32位地址。
BAR的大小取决于显卡类型。而且BAR的大小独立于真实的VRAM大小。这意味着NV30+显卡不可能通过BAR映射出所有的显卡内存。
- NV3 非直接内存访问IO端口： BAR2 -  0x100 字节的IO端口空间

这IO端口范围用于非直接访问BAR0 或 BAR1 通过 实模式代码。这在NV3上有。

- RAMIN： BAR2 或 BAR3 - 内存 0x1000000 字节或更多，取决于显卡类型。

RAMIN是在pre-G80显卡上VRAM末端特殊的区域，保存着各种控制结构体。 RAMIN开始于VRAM的末端，地址向相反的方向增长。因此需要特殊的映射访问它。

pre-NV40显卡限制其大小为1MB，为NV3调整了BAR0 或 BAR1 中的映射。 NV40+ 允许更大的 RAMIN 地址。


- BAR 5: G80非直接内存访问



### 查看方法

通过 `lspci` 命令查看本机PCI设备列表。

> 01:00.0 VGA compatible controller: NVIDIA Corporation GK110B [GeForce GTX TITAN Black] (rev a1)

查看详细的信息：

```
sudo lspci -v -s 01:00.0
```

	01:00.0 VGA compatible controller: NVIDIA Corporation GK110B [GeForce GTX TITAN Black] (rev a1) (prog-if 00 [VGA controller])
		Subsystem: NVIDIA Corporation GK110B [GeForce GTX TITAN Black]
		Flags: bus master, fast devsel, latency 0, IRQ 37
		Memory at f2000000 (32-bit, non-prefetchable) [size=16M]
		Memory at e8000000 (64-bit, prefetchable) [size=128M]
		Memory at f0000000 (64-bit, prefetchable) [size=32M]
		I/O ports at e000 [size=128]
		[virtual] Expansion ROM at f3000000 [disabled] [size=512K]
		Capabilities: <access denied>
		Kernel driver in use: nvidia
		Kernel modules: nvidiafb, nouveau, nvidia_384_drm, nvidia_384


BAR0： 0xf2000000 (MMIO registers)
BAR1 and BAR2: 0xe8000000
BAR3 and BAR4: 0xf0000000
BAR5: 0xe000 (I/O port)


参考[PCI Express I/O System](https://insujang.github.io/2017-04-03/pci-express-io-system/)
