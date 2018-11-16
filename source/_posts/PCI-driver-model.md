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

# PCI设备信息查看及寻址方式

## lspci命令
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

## 文件系统
系统中的PCI设备信息可以从 `/proc/bus/pci/device` , `/sys/bus/pci/device` 文件系统中查看；
其中设备信息的表示格式为： `总线域（16位）`：`总线编号（8位）`：`设备编号（5位）`.`功能编号（3位）`
由以上信息可以知道：每个总线域最多有256个PCI总线，每个总线最多有32个PCI设备，每个设备最多有8个功能；

# PCI 配置空间

在驱动已探测到设备后, 它常常需要读或写 3 个地址空间: 内存, 端口, 和配置。 特别地, 存取配置空间对驱动是至关重要的, 因为这是唯一的找到设备被映射到内存和 I/O 空间的位置的方法。


Linux 将配置空间中的 `venderId`, `deviceId`, `classcode`, `subvendorId`, `subdeviceId`, `class_mask`, `driver_data`(kernel_ulong_t，不是必须的)抽象为 `struct pci_device_id` 结构体，PCI驱动程序用该结构体告诉内核，本身支持什么样的PCI设备列表；
```
MODULE_DEVICE_TABLE(pci,ids); 
```
使用该宏将pci驱动程序支持的设备列表导出到用户空间，供热插拔系统为设备查找驱动程序使用；在编译模块的时候编译系统会抽取该宏数据并导出到用户空间。

# Linux PCI设备驱动结构体

```
#include <linux/pci.h>
```
	包含 PCI 寄存器的符号名和几个供应商和设备 ID 值的头文件.

```
struct pci_dev;
```
	表示内核中一个 PCI 设备的结构.
```
struct pci_driver;
```
代表一个 `PCI 驱动` 结构，该结构体是 `PCI设备` 与 `PCI设备驱动程序` 的联系桥梁，通过该结构体可以查询到驱动程序的设备并初始化设备；
比较重要的字段如下：
```
struct pci_driver{
	char *name;
	struct pci_device_id *ids;
	int (*probe)(struct pci_dev *dev,struct pci_device_id *id);
	void (*remove)(struct pci_dev *dev);
	int (*suspend)(struct pci_dev *dev,u32 state);
	int (*resume)(struct pci_dev *dev);
	... ...
};
```
```
struct pci_device_id;
```
	描述这个驱动支持的 PCI 设备类型的结构.

# Linux PCI设备配置空间访问



```
struct pci_dev *pci_find_device(unsigned int vendor, unsigned int device, struct pci_dev *from);
struct pci_dev *pci_find_device_reverse(unsigned int vendor, unsigned int device, const struct pci_dev *from);
struct pci_dev *pci_find_subsys (unsigned int vendor, unsigned int device, unsigned int ss_vendor, unsigned int ss_device, const struct pci_dev *from);
struct pci_dev *pci_find_class(unsigned int class, struct pci_dev *from);
```
在设备列表中搜寻带有一个特定签名的设备, 或者属于一个特定类的. 返回值是 NULL 如果没找到. from 用来继续一个搜索; 在你第一次调用任一个函数时它必须是 NULL, 并且它必须指向刚刚找到的设备如果你寻找更多的设备. 这些函数不推荐使用, 用 pci_get_ 变体来代替.

```
struct pci_dev *pci_get_device(unsigned int vendor, unsigned int device, struct pci_dev *from);
struct pci_dev *pci_get_subsys(unsigned int vendor, unsigned int device, unsigned int ss_vendor, unsigned int ss_device, struct pci_dev *from);
struct pci_dev *pci_get_slot(struct pci_bus *bus, unsigned int devfn);
```
在设备列表中搜索一个特定签名的设备，或者属于一个特定类。如果没找到，返回值是 NULL。 `from` 用来继续搜索； 在第一次调用任一个函数时它必须是 NULL， 并且如果想要搜寻更多的设备，它必须指向刚刚找到的设备。返回的结构使它的引用计数递增，并且在调用者完成它, 函数 pci_dev_put 必须被调用.


```
int pci_read_config_byte(struct pci_dev *dev, int where, u8 *val);
int pci_read_config_word(struct pci_dev *dev, int where, u16 *val);
int pci_read_config_dword(struct pci_dev *dev, int where, u32 *val);
int pci_write_config_byte (struct pci_dev *dev, int where, u8 *val);
int pci_write_config_word (struct pci_dev *dev, int where, u16 *val);
int pci_write_config_dword (struct pci_dev *dev, int where, u32 *val);
```
读或写 PCI 配置寄存器的函数。尽管 Linux 内核负责字节序, 必须小心字节序，尤其从单个字节组合多字节值时. PCI 总线是小端。


```
unsigned long pci_resource_start(struct pci_dev *dev, int bar);
unsigned long pci_resource_end(struct pci_dev *dev, int bar);
unsigned long pci_resource_flags(struct pci_dev *dev, int bar);
```
处理 PCI 设备资源的函数。

# Linux PCI设备驱动程序探测及注册：

```
int pci_register_driver(struct pci_driver *drv);
int pci_module_init(struct pci_driver *drv);
void pci_unregister_driver(struct pci_driver *drv);
```
从内核注册或注销一个 PCI 驱动的函数.
```
int pci_enable_device(struct pci_dev *dev);
```
使能一个 PCI 设备.

```
int pci_probe(struct pci_dev *dev,struct pci_device_id *id){

	struct pci_privdata *data=NULL;

	struct resource *resource=NULL;

	if(pci_device_is_present(dev)){
		//1,不支持pci总线;
	}

                                      
	if(pci_enable_device(dev)){

		//2,使能设备;

	}

	//3,从配置空间获取信息；

	data=kmalloc(sizeof(struct pci_privdata),GFP_KERNEL);

	data->phy_addr=pci_resource_start(dev,1);

	data->size          =pci_resource_len(dev,1);

	data->flags         =pci_resource_flags(dev,1);

	if(data->flags != IORESOURCE_MEM){

		//4,判断是否为内存空间或IO端口空间；

	}

	//resource=request_mem_region(data->phy_addr,data->size,NAME);

	resource=pci_request_region(data->phy_addr,data->size,NAME);

	if(!resource){

		//5,独占设备;

	}

	//6, io内存映射，映射到内核虚拟地址空间;

	data->addr=ioremap(data->phy_addr,data->size);

	//7,设置PCI设备私有数据;

	pci_set_drvdata(dev,data);

	//8,设置PCI设备为DMA主设备;

	pci_set_master(dev);

	return0;

}



int  pci_remove(struct pci_dev *dev){

	//与probe函数相反的顺序将probe函数中的资源都释放掉;

	return 0;

}



//在驱动模块中注册驱动数据结构;

static int __init pci_module_init(void){

	return pci_register_driver(&pci_driver);

}
```

# 参考
1. [PCI configuration space](https://en.wikipedia.org/wiki/PCI_configuration_space)
2. [Access PCI device registry from Linux user space](https://falsinsoft.blogspot.com/2017/09/access-pci-device-registry-from-linux.html)
3. [lspci(8) - Linux man page](https://linux.die.net/man/8/lspci)
4. [Access physical memory from Linux user space](https://falsinsoft.blogspot.com/2013/10/access-physical-memory-in-linux.html)
5. [The anatomy of a PCI/PCI Express kernel driver](http://haifux.org/lectures/256/haifux-pcie.pdf)
6. [Linux PCI Driver Model](http://linuxkernel51.blogspot.com/2012/08/linux-pci-driver-model.html)
7. [PCI](https://wiki.osdev.org/PCI)
8. [PCI Express](https://wiki.osdev.org/PCI_Express)
9. Linux Device Driver, Third Edition
10. [第 12 章 PCI 驱动](http://www.deansys.com/doc/ldd3/ch12.html#ThePCIInterface.sect1)
11. [LDD之PCI设备](https://blog.csdn.net/a372048518/article/details/54143059 )