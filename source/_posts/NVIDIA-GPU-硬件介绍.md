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


# nVidia GPU Model

![图片来源论文GPUvm：GPU Virtualization at the Hypervisor](../NVIDIA-GPU-硬件介绍/gpu_management_model.png)

从上述图可以看出组成GPU的几个重要组成：
+ MMIO: 
	+ CPU与GPU的交流就是通过MMIO进行的。
	+ DMA传输大量的数据就是通过MMIO进行命令控制的
	+ I/O端口可用于间接访问MMIO区域，像Nouveau等开源软件从来不访问它
+ GPU context
	+ GPU context代表了GPU计算的状态
	+ 在GPU上拥有自己的虚拟地址
	+ 在GPU上可以共存多种context
+ GPU channel
	+ 任何命令都是由CPU发出
	+ 命令流（command stream）被提交到硬件单元，也就是GPU channel
	+ 每个GPU channel关联一个context，而一个GPU context可以有多个GPU channel。
	+ 每个GPU context 包含相关channel的 GPU channel descriptors 。 每个 descriptor 都是 GPU 内存中的一个对象。
	+ 每个 GPU channel descriptor 存储了 channel 的设置，其中就包括 page table 。
	+ 在每个 GPU channel 中，在GPU内存中分配了唯一的命令缓存，这通过MMIO对CPU可见。
	+ GPU context switching 和命令执行都在GPU硬件内部调度。
+ GPU Page Table
	+ GPU context 在虚拟基地空间由页表隔离其他的 context 。
	+ GPU的页表隔离鱼CPU页表，位于GPU内存中。
	+ GPU 页表的物理地址位于 GPU channel descriptor 中。
	+ GPU 页表不仅仅将 GPU虚拟地址转换成GPU内存的物理地址，也可以转换成CPU的物理地址。因此，GPU页表可以将GPU虚拟地址和CPU内存地址统一到GPU统一虚拟地址空间来。
+ PCIe BAR
	+ GPU 设备通过PCIe总线接入到主机上。 base address registers(BARs) 是 MMIO的窗口，在GPU启动时候配置。
	+ GPU的控制寄存器和内存都映射到了BARs中。
	+ GPU设备内存通过映射的MMIO窗口去配置GPU和访问GPU内存。
+ PFIFO Engine
	+ PFIFO是GPU命令提交通过的一个特殊的部件
	+ PFIFO维护了一些独立命令队列，也就是channel
	+ 此命令队列是 ring buffer，有 put 和 get 的指针。
	+ 所有访问 channel 控制区域的执行指令都被 PFIFO 拦截下来。
	+ GPU 驱动使用 channel descriptor 来存储相关的 channel设定。
+ BO
	+ Buffer Object (bo)，内存的一块(block)，能够用于存储 texture, a render target, shader code等等。
	+ nouveau和gdev经常使用BO


参考
[GPU Architecture Overview](https://insujang.github.io/2017-04-27/gpu-architecture-overview/)

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

查看此PCI设备的ID。

```
sudo lspci -s 01:00.0 -n
```
> 01:00.0 0300: 10de:100c (rev a1)
即，设备ID是 `100c`，而厂商ID是 `10de`。
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


参考
1. [PCI Express I/O System](https://insujang.github.io/2017-04-03/pci-express-io-system/)
2. [lspci](http://manpages.ubuntu.com/manpages/xenial/man8/lspci.8.html)


## PFIFO

`PFIFO` 用于收集用户发送的命令并将其传送到执行单元。大致分成三部分：
+ PFIFO cache： 以FIFO的队列形式存储要执行的GPU命令。
+ PFIFO pusher：搜集用户输入的命令并将其存入cache中。
	共有两种模式：PIO和DMA模式。PIO模式中，用户直接通过USER MMIO 区域写入命令。 DMA模式中，PFIFO从内存的buffer中读命令，称为pushbuffer的内存，而USER MMIO 区域仅用于控制pushbuffer 读取。
+ PFIFO puller：从cache中取命令，并将其送往执行单元。

`channel` 是PFIFO最核心的概念，它是单独的命令流。 channel是上下文切换并且独立的。
为了节省PFIFO每个channel上下文，使用了 `RAMFC` 内存结构体。
PFIFO cache 每次只能对单一的channel设置。
从NV50 开始，PFIFO上下文在做切换时候会保存到memory中。

当pusher把命令插入新的channel时，channel会切换。当puller传递命令时，puller会请求channel切换。这意味着PFIFO和执行单元在不同的channel上。

存储在cache中的命令是由subchannel、method、data组成的元祖。每个channel有8个subchannel，并且有对象关联它们。method是介于0和0x1ffc之间能被4整除的数字，并且选择命令来执行。可获得的method集合依赖于关联到给定subchannel的对象。
method number如同内部硬件寄存器地址，因此能被4整除，这都是遗留问题。
大部分method都会直接原始的传送到执行引擎，一些会特殊一点，直接被PFIFO处理。

+ 0x0000： 绑定对象到subchannel
+ 0x0004-0x00fc：被PFIFO保留使用的method，从不传递给执行引擎。
+ 0x0180-0x01fc：传递给执行引擎的method。

数据值按照32位提交，依据method来转换。



















[PFIFO - The command submission engine](https://github.com/pathscale/pscnv/wiki/PFIFO)