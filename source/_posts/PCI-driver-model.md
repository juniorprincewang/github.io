---
title: PCI driver model
date: 2018-07-09 20:12:10
tags:
- pci
categories:
- GPU
---
由于要研究NVIDIA GPU的工作原理，需要对PCI(e)的原理掌握清楚，在此整理了一些知识点，包括PCI配置空间和PCI驱动程序。
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

**BDF**：三个数字分别是 01： Bus Number, 00: Device Number, 0: Function Number。

设备驱动的配置信息，可以通过 `lspci`的 选项 `-x`， `-xxx`， `-xxxx` 打印出来，不过要用 `root` 用户执行。

为了确定PCI设备的位置，PCI设备必须能够映射到系统的IO端口地址空间或者内存映射的地址空间。
系统的固件、设备驱动或操作系统编排BARs，通过将配置命令写入到PCI控制器中来通知设备的地址映射。  


## 文件系统
系统中的PCI设备信息可以从 `/proc/bus/pci/device` , `/sys/bus/pci/device` 文件系统中查看；
其中设备信息的表示格式为： `总线域（16位）`：`总线编号（8位）`：`设备编号（5位）`.`功能编号（3位）`，表示为`AAAA:BB:CC:D`。  
由以上信息可以知道：  
+ 每个总线域最多有256个PCI总线，
+ 每个总线最多有32个PCI设备，
+ 每个设备最多有8个功能；

# PCI 配置空间

在驱动已探测到设备后, 它常常需要读或写 3 个地址空间: 内存（Device Memory）, 端口（I/O port，PIO）, 和配置（Configuration Space）。 特别地, 存取配置空间对驱动是至关重要的, 因为这是唯一的找到设备被映射到内存和 I/O 空间的位置的方法。

![PCI设备配置空间](../PCI-driver-model/450px-Pci-config-space.svg.png)

PCI配置空间寄存器定义在 *include/uapi/linux/pci_regs.h* 。  
```c
/*
 * Under PCI, each device has 256 bytes of configuration address space,
 * of which the first 64 bytes are standardized as follows:
 */
#define PCI_STD_HEADER_SIZEOF	64
#define PCI_VENDOR_ID		0x00	/* 16 bits */
#define PCI_DEVICE_ID		0x02	/* 16 bits */
#define PCI_COMMAND		0x04	/* 16 bits */
...
#define PCI_STATUS		0x06	/* 16 bits */
...
#define PCI_CLASS_REVISION	0x08	/* High 24 bits are class, low 8 revision */
#define PCI_REVISION_ID		0x08	/* Revision ID */
#define PCI_CLASS_PROG		0x09	/* Reg. Level Programming Interface */
#define PCI_CLASS_DEVICE	0x0a	/* Device class */

#define PCI_CACHE_LINE_SIZE	0x0c	/* 8 bits */
#define PCI_LATENCY_TIMER	0x0d	/* 8 bits */
#define PCI_HEADER_TYPE		0x0e	/* 8 bits */
#define PCI_BIST		0x0f	/* 8 bits */

/*
 * Base addresses specify locations in memory or I/O space.
 * Decoded size can be determined by writing a value of
 * 0xffffffff to the register, and reading it back.  Only
 * 1 bits are decoded.
 */
#define PCI_BASE_ADDRESS_0	0x10	/* 32 bits */
#define PCI_BASE_ADDRESS_1	0x14	/* 32 bits [htype 0,1 only] */
#define PCI_BASE_ADDRESS_2	0x18	/* 32 bits [htype 0 only] */
#define PCI_BASE_ADDRESS_3	0x1c	/* 32 bits */
#define PCI_BASE_ADDRESS_4	0x20	/* 32 bits */
#define PCI_BASE_ADDRESS_5	0x24	/* 32 bits */
/* Header type 0 (normal devices) */
#define PCI_CARDBUS_CIS		0x28
#define PCI_SUBSYSTEM_VENDOR_ID	0x2c
#define PCI_SUBSYSTEM_ID	0x2e
#define PCI_ROM_ADDRESS		0x30	/* Bits 31..11 are address, 10..1 reserved */
#define PCI_CAPABILITY_LIST	0x34	/* Offset of first capability list entry */
/* 0x35-0x3b are reserved */
#define PCI_INTERRUPT_LINE	0x3c	/* 8 bits */
#define PCI_INTERRUPT_PIN	0x3d	/* 8 bits */
#define PCI_MIN_GNT		0x3e	/* 8 bits */
#define PCI_MAX_LAT		0x3f	/* 8 bits */
```

PCI 设备由`VendorID`, `DeviceIDs`, 和 `Class Codes` 标识，定义在 *include/linux/pci_ids.h* 。  
`Class Code` 指定了设备的功能，在 `0x0b`偏移处标识 base code class, `0xa`偏移处标识了sub-class code结合指定了设备的功能。比如  
```c
/* include/linux/pci_ids.h */
#define PCI_BASE_CLASS_MEMORY       0x05
#define PCI_CLASS_MEMORY_RAM        0x0500

#define PCI_BASE_CLASS_DISPLAY      0x03
#define PCI_CLASS_DISPLAY_VGA       0x0300  
#define PCI_CLASS_DISPLAY_XGA       0x0301
#define PCI_CLASS_DISPLAY_3D		0x0302
```

可以通过PCI的配置空间来查看 VendorID, DeviceID, Class Code, Subsystem VendorID (i.e. SVendor) and SubsystemID(i.e. SDevice)值。  
```
hexdump /sys/devices/pci0000\:00/0000\:01\:00.0/config 
```


# PCI Device Driver Specifics  

根据[Kernel关于PCI Driver文件，How To Write Linux PCI Drivers](https://www.kernel.org/doc/html/latest/PCI/pci.html)的说法，PCI device初始化的流程为：

	Register the device driver and find the device
	Enable the device
	Request MMIO/PIO resources
	Set the DMA mask size (for both coherent and streaming DMA)
	Allocate and initialize shared control data (pci_allocate_coherent())
	Access device configuration space (if needed)
	Register IRQ handler (request_irq())
	Initialize non-PCI (i.e. LAN/SCSI/etc parts of the chip)
	Enable DMA/processing engines


## Driver Registeration

+ device configuration  

Linux 将配置空间中的 `venderId`, `deviceId`, `classcode`, `Subsystem VendorID(subvendorId)`, `SubsystemID(subdeviceId)`, `class_mask`, `driver_data`(kernel_ulong_t，不是必须的)抽象为 `struct pci_device_id` 结构体，PCI驱动程序用该结构体告诉内核，本身支持什么样的PCI设备列表；  
```c
/* /include/linux/mod_devicetable.h */
struct pci_device_id {
	__u32 vendor, device;		/* Vendor and device ID or PCI_ANY_ID*/
	__u32 subvendor, subdevice;	/* Subsystem ID's or PCI_ANY_ID */
	__u32 class, class_mask;	/* (class,subclass,prog-if) triplet */
	kernel_ulong_t driver_data;	/* Data private to the driver */
};
```

这个结构包含不同的成员:`__u32 vendor` ; `__u32 device` 。这些指定一个设备的 PCI 供应商和设备 ID. 如果驱动可处理任何供应商或者设备
ID, 这些成员取值 `PCI_ANY_ID`。  
具体参数见 <https://www.kernel.org/doc/html/latest/PCI/pci.html#c.pci_device_id>。  
`PCI_DEVICE(vendor, device)` 和 `PCI_DEVICE_CLASS(device_class, device_class_mask)` 可用于初始化 `struct pci_device_id` 的不同字段。  
为了将`struct pci_device_id` 导出到用户空间，供热插拔系统为设备查找驱动程序使用，则需要使用宏 `MODULE_DEVICE_TABLE(pci,ids);` ，在编译模块的时候编译系统会抽取该宏数据并导出到用户空间。  

+ register PCI deriver:  

PCI设备的结构体和函数定义在文件 *include/linux/pci.h* 中，需要引入头文件 `#include <linux/pci.h>`。  
其中重要的结构体包括：  
`struct pci_dev` 表示内核中一个 **PCI 设备**的结构。  
关于`struct pci_dev` 可以参见[The Linux Kernel Device Model](https://dri.freedesktop.org/docs/drm/driver-api/driver-model/overview.html)。  
`struct pci_driver` 代表一个 **PCI 驱动** 结构，该结构体是 **PCI设备** 与 **PCI设备驱动** 的联系桥梁，通过该结构体可以查询到驱动程序的设备并初始化设备；
比较重要的字段如下，可参见具体[struct pci_driver](https://www.kernel.org/doc/html/latest/PCI/pci.html#c.pci_driver)的描述。  ：
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

PCI对 `struct pci_driver *` 指向的对象进行PCI驱动的注册。  
```c
int pci_register_driver(struct pci_driver *drv);
```

PCI driver通过 `pci_register_driver()` 发现PCI设备。  
具体地，用`struct pci_driver` 结构的`id_table`和所有PCI设备的`id_table`比较，找到匹配项，然后取出该PCI设备的`struct pci_dev *` 对象，最后调用`struct pci_driver`对象的`probe`回调方法进行初始化。

### 设备初始化步骤  

`probe` 回调函数的步骤。  

+ 使能设备  

在 `probe` 函数中使用，在driver能够访问设备资源（I/O区域或中断）前，使能PCI设备。  
如果设备suspended，wake up device。  

```
int pci_enable_device(struct pci_dev *dev);
```

如果首次调用`pci_enable_device()`，则使能设备，可以包括为设备赋值PIO，内存和中断。若不是首次，则增加设备的usage count。  

对应的disable函数：  
```
void pci_disable_device(struct pci_dev * dev);
```

PCI设备包括三种能够寻址的regions：configuration space,PIO和memory。  
对于configuration space，kernel提供6种helper function。  
读或写 PCI 配置寄存器的函数。
```c
int pci_read_config_byte(struct pci_dev *dev, int where, u8 *val);
int pci_read_config_word(struct pci_dev *dev, int where, u16 *val);
int pci_read_config_dword(struct pci_dev *dev, int where, u32 *val);
int pci_write_config_byte (struct pci_dev *dev, int where, u8 *val);
int pci_write_config_word (struct pci_dev *dev, int where, u16 *val);
int pci_write_config_dword (struct pci_dev *dev, int where, u32 *val);
```
`offset` 表示配置空间的偏移量，可以在 *include/uapi/linux/pci_regs.h*查看PCI配置空间寄存器 。  

尽管 Linux 内核负责字节序, 必须小心字节序，尤其从单个字节组合多字节值时. PCI 总线是小端。  
比如，从PCI配置空间读取 PCI_INTERRUPT_LINE：  
```c
u8 val;
pci_read_config_byte(dev, PCI_INTERRUPT_LINE, &val);
```

+ 请求PIO/MMIO resource  

CPU不能直接从PCI配置空间访问MMIO和PIO 地址，需要映射到CPU的地址空间后才可以。  
PCI有6 个address regions，每个region包括I/O地址或内存地址，包括 BAR0 - BAR5 (the base address register)。  
不像常规的内存，I/O控制寄存器的内存是非预取的（nonprefetchable）。  
region是内存映射或端口映射的I/O地址内存。  
设备驱动需要调用`pci_request_region()` 确保没有其他设备使用此地址region，保留PCI IO和内存resource；再调用`pci_release_region()`释放此PIO，memory resource。 

`pci_request_region()` 将与PCI设备@pdev关联的所有PCI region标记为所有者@res_name保留。  

```
/**
 *	pci_request_region - Reserve PCI I/O and memory resource
 *	@pdev: PCI device whose resources are to be reserved
 *	@bar: BAR to be reserved
 *	@res_name: Name to be associated with resource
 *
 *	Mark the PCI region associated with PCI device @pdev BAR @bar as
 *	being reserved by owner @res_name.  Do not access any
 *	address inside the PCI regions unless this call returns
 *	successfully.
 *
 *	Returns 0 on success, or %EBUSY on error.  A warning
 *	message is also printed on failure.
 */
int pci_request_region(struct pci_dev *pdev, int bar, const char *res_name)

/**
 *	pci_release_region - Release a PCI bar
 *	@pdev: PCI device whose resources were previously reserved by pci_request_region
 *	@bar: BAR to release
 *
 *	Releases the PCI I/O and memory resources previously reserved by a
 *	successful call to pci_request_region.  Call this function only
 *	after all use of the PCI regions has ceased.
 */
void pci_release_region(struct pci_dev *pdev, int bar)
```

`pci_request_region()`的通用类型是`request_mem_region()`（用于MMIO范围）和`request_region()`（用于IO端口范围）。  

内核还提供了相关的函数：  
```c
/*drivers/pci/pci.c*/
/**
 *	pci_request_regions - Reserved PCI I/O and memory resources
 *	@pdev: PCI device whose resources are to be reserved
 *	@res_name: Name to be associated with resource.
 *
 *	Mark all PCI regions associated with PCI device @pdev as
 *	being reserved by owner @res_name.  Do not access any
 *	address inside the PCI regions unless this call returns
 *	successfully.
 *
 *	Returns 0 on success, or %EBUSY on error.  A warning
 *	message is also printed on failure.
 */
int pci_request_regions(struct pci_dev *pdev, const char *res_name)
{
	return pci_request_selected_regions(pdev, ((1 << 6) - 1), res_name);
}
/**
 *	pci_release_regions - Release reserved PCI I/O and memory resources
 *	@pdev: PCI device whose resources were previously reserved by pci_request_regions
 *
 *	Releases all PCI I/O and memory resources previously reserved by a
 *	successful call to pci_request_regions.  Call this function only
 *	after all use of the PCI regions has ceased.
 */

void pci_release_regions(struct pci_dev *pdev)
{
	pci_release_selected_regions(pdev, (1 << 6) - 1);
}
...
```


为操作PIO和memory regions，内核同样提供了helper functions：  
```c
unsigned long pci_resource_[start|len|end|flags] (struct pci_dev *pdev, int bar);
```

```c
/*include/linux/pci.h*/
// Returns bus start address for a given PCI region
resource_size_t start = pci_resource_start(dev, bar);
// Returns the byte length of a PCI region
resource_size_t len = pci_resource_len(dev, bar)
// Returns bus end address for a given PCI region
resource_size_t end = pci_resource_end(dev, bar);
// Returns the flags associated with this resource.
unsigned long flags = pci_resource_flags(dev, bar);
```
由于这些是宏，参数`dev` 类型为 `struct pci_dev *`，`bar`类型为 `int`，表示BAR index。 返回类型`resource_size_t`为 `typedef phys_addr_t resource_size_t;` 。  
resource flag 最重要的两个为 `IORESOURCE_IO` `IORESOURCE_MEM`。  


**PIO region**  

为操作PIO region，比如device control registers，需要进行以下三个步骤：  
1. 从configuration address的BAR中获得IO base address。
```c
unsigned long io_base = pci_resource_start(pdev, bar);
```

2. 对此region标记占有者owner。
```c
request_region(io_base, length, "my_driver");
```
`length` 表示控制寄存器空间大小；`"my_driver"`表示此region的拥有者，而此entry可在*/proc/ioports*查看。  
当然也可以使用上面提到的 `pci_request_region()` 。  

3. 对相应偏移位置的寄存器进行读写操作  
```c
/* Read */
register_data = inl(io_base + REGISTER_OFFSET);

/* Write */
outl(register_data, iobase + REGISTER_OFFSET);
```

`inX()` 和 `outX()` 可以被通用的 `ioreadX()` 与 `iowriteX()` 替换。  

**memory region**

对memory region的操作与PIO region不同，比如PCI video card的frame buffer。
1. 首先获得memory region的base address, length, flags。
```c
unsigned long mmio_base   = pci_resource_start(pdev, bar);
unsigned long mmio_length = pci_resource_length(pdev, bar);
unsigned long mmio_flags  = pci_resource_flags(pdev, bar);
```

2. 标记此memory region的拥有者  
```c
request_mem_region(mmio_base, mmio_length, "my_driver");
```
当然也可以使用 `pci_reguest_region()`。  
3. 获取CPU能够访问的memory regions  

+ 获取CPU能够访问的PIO和memory regions

`pci_iomap()`可用于创建PCI BAR的虚拟映射cookie，是对 `ioremap` 的封装。  
```c
/* lib/pci_iomap.c*/
/* pci_iomap - create a virtual mapping cookie for a PCI BAR
 * @dev: PCI device that owns the BAR
 * @bar: BAR number
 * @maxlen: length of the memory to map, If you want to get access to
 * the complete BAR without checking for its length first, pass %0 here.
 */
void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long maxlen)
{
    return pci_iomap_range(dev, bar, 0, maxlen);
}
void __iomem *pci_iomap_range(struct pci_dev *dev,
                  int bar,
                  unsigned long offset,
                  unsigned long maxlen)
{
    resource_size_t start = pci_resource_start(dev, bar);
    resource_size_t len = pci_resource_len(dev, bar);
    unsigned long flags = pci_resource_flags(dev, bar);

    if (len <= offset || !start)
        return NULL;
    len -= offset;
    start += offset;
    if (maxlen && len > maxlen)
        len = maxlen;
    if (flags & IORESOURCE_IO)
        return __pci_ioport_map(dev, start, len);
    if (flags & IORESOURCE_MEM)
        return ioremap(start, len);
    /* What? */
    return NULL;
}
```

`pci_iomap()`返回 `__iomem` 地址后，kernel就可以对MMIO进行 `read[b|w|l|q]()` 和 `write[b|w|l|q]()` （*include/asm-generic/io.h*）的读写操作了。  

从PIO或者MMIO内存 IOMEM 读取或写入的通用API包括： `ioread[8|16|16be|32|32be](void __iomem *addr)` 和 `iowrite[8|16|16be|32|32be](u[8|16|32] val, void __iomem *addr)` （*lib/iomap.c*）的读写操作了。 


+ 注册IRQ handler  

许多设备都支持基于引脚的中断（pin-based interrupts）和消息信号中断（MSI），注册一个引脚中断：  
```c
int request_irq(unsigned int irq, irq_handler_t handler, unsigned long flags, 
	const char *name, void *dev);
```
根据 *Documentation/PCI/pci.txt* ，所有基于引脚的中断都需要将flags设置为 IRQF_SHARED。  

+ DMA  allocation and mmap


[how to instantiate and use a dma driver linux module](https://stackoverflow.com/a/17915149)  
[mmaping MMIO and DMA Regions, Case Studies with QEMU Virtual Devices](http://web.archive.org/web/20151126082733/http://nairobi-embedded.org/mmap_mmio_dma.html)  

强烈建议先看一遍 [Documentation/DMA-API-HOWTO.txt](https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt) 和 [Dynamic DMA mapping using the generic device](https://www.kernel.org/doc/Documentation/DMA-API.txt)  。  

DMA mapping分为 Consistent DMA mappings 和 Streaming DMA mappings。  
Consistent DMA mappings 保证CPU和Device访问数据的一致性，不需要显式flush数据，可以理解为同步操作，一般在驱动初始化时map并在驱动卸载时unmap；但 Streaming DMA mappings 是只为一次DMA transfer时map并在使用后unmap，如果要使用多次则需要则显式flush，可以理解为异步操作。  
kernel doc建议使用consistent DMA mapping。  

分配大内存的DMA regions使用`dma_alloc_coherent()`，分配小的DMA regions，使用 `dma_pool_create()`。    

```c
/*include/linux/dma-mapping.h*/
#include <linux/dma-mapping.h>
void * dma_alloc_coherent(struct device *dev, size_t size,
               dma_addr_t *dma_handle, gfp_t flag)

void dma_free_coherent(struct device *dev, size_t size, void *cpu_addr,
              dma_addr_t dma_handle)

struct dma_pool * dma_pool_create(const char *name, struct device *dev,
			size_t size, size_t align, size_t alloc);
```
`dma_alloc_coherent()` 返回虚拟地址，并且返回总线地址 `dma_handle` 给设备使用。  

**DMA寻址限制**  

默认情况下，kernel认为设备的DMA可以访问系统总线地址为32位。如果有寻址限制，则在probe函数中，通知kernel设备的DMA寻址限制，那么需要设置DMA mask。  
```c
// 设置 streaming 和 coherent DMA mask 
int dma_set_mask_and_coherent(struct device *dev, u64 mask);
// 设置 streaming DMA mask
int dma_set_mask(struct device *dev, u64 mask);
// 设置 coherent DMA mask
int dma_set_coherent_mask(struct device *dev, u64 mask);
```
比如，设置只能驱动低24位地址，
```c
if (dma_set_mask(dev, DMA_BIT_MASK(24))) {
		dev_warn(dev, "mydev: 24-bit DMA addressing not available\n");
		goto ignore_this_device;
	}
```

内核分配完内存后，需要将CPU的内存物理地址和设备的总线地址关联起来，即IOMMU将DMA bus address翻译到physical address。  
`dma_map_[single|sg|page]()` 的工作就是将内核虚拟地址与DMA总线地址映射。  

`dma_map_single()` 映射单个region，`dma_map_sg()` 映射多个regions的scatterlists，`dma_map_page()` 映射HIGHMEM memory，传入 page/offset 参数。需要做好错误处理。  

**mmaping the DMA allocation**

```c
/*include/linux/dma-mapping.h*/
/**
 * dma_mmap_attrs - map a coherent DMA allocation into user space
 * @dev: valid struct device pointer, or NULL for ISA and EISA-like devices
 * @vma: vm_area_struct describing requested user mapping
 * @cpu_addr: kernel CPU-view address returned from dma_alloc_attrs
 * @handle: device-view address returned from dma_alloc_attrs
 * @size: size of memory originally requested in dma_alloc_attrs
 * @attrs: attributes of mapping properties requested in dma_alloc_attrs
 *
 * Map a coherent DMA buffer previously allocated by dma_alloc_attrs
 * into user space.  The coherent DMA buffer must not be freed by the
 * driver until the user space mapping has been released.
 */
static inline int
dma_mmap_attrs(struct device *dev, struct vm_area_struct *vma, void *cpu_addr,
           dma_addr_t dma_addr, size_t size, unsigned long attrs);
#define dma_mmap_coherent(d, v, c, h, s) dma_mmap_attrs(d, v, c, h, s, 0)

static inline dma_addr_t dma_map_single_attrs(struct device *dev, void *ptr,
					      size_t size,
					      enum dma_data_direction dir,
					      unsigned long attrs);
#define dma_map_single(d, a, s, r) dma_map_single_attrs(d, a, s, r, 0)

static inline void dma_unmap_single_attrs(struct device *dev, dma_addr_t addr,
					  size_t size,
					  enum dma_data_direction dir,
					  unsigned long attrs);
#define dma_unmap_single(d, a, s, r) dma_unmap_single_attrs(d, a, s, r, 0)
...
static inline int dma_map_sg_attrs(struct device *dev, struct scatterlist *sg,
                   int nents, enum dma_data_direction dir,
                   unsigned long attrs);
#define dma_map_sg(d, s, n, r) dma_map_sg_attrs(d, s, n, r, 0)
#define dma_unmap_sg(d, s, n, r) dma_unmap_sg_attrs(d, s, n, r, 0)

static inline dma_addr_t dma_map_page_attrs(struct device *dev,
                        struct page *page,
                        size_t offset, size_t size,
                        enum dma_data_direction dir,
                        unsigned long attrs);
#define dma_map_page(d, p, o, s, r) dma_map_page_attrs(d, p, o, s, r, 0)
#define dma_unmap_page(d, a, s, r) dma_unmap_page_attrs(d, a, s, r, 0)
...
// map direction
DMA_NONE		no direction (used for debugging)
DMA_TO_DEVICE		data is going from the memory to the device
DMA_FROM_DEVICE		data is coming from the device to the memory
DMA_BIDIRECTIONAL	direction isn't known
```

对于streaming DMA mapping而言，需要同步操作，具体就是
1. CPU读取已经被设备DMA写入的值之前（指定了 `DMA_FROM_DEVICE` 的mapping）
2. 在CPU使用DMA写入了值之后（指定了 `DMA_TO_DEVICE`的mapping）
3. 如果指定了 `DMA_BIDIRECTIONAL` ，则在处理内存前和后都需要调用同步操作。  

同步操作API为：  
```c
void dma_sync_single_for_cpu(struct device *dev, dma_addr_t dma_handle,
				size_t size,
				enum dma_data_direction direction)

void dma_sync_single_for_device(struct device *dev, dma_addr_t dma_handle,
				   size_t size,
				   enum dma_data_direction direction)

void dma_sync_sg_for_cpu(struct device *dev, struct scatterlist *sg,
			    int nents,
			    enum dma_data_direction direction)

void dma_sync_sg_for_device(struct device *dev, struct scatterlist *sg,
			       int nents,
			       enum dma_data_direction direction)

```

[Linux kernel device driver to DMA from a device into user-space memory](https://stackoverflow.com/questions/5539375/linux-kernel-device-driver-to-dma-from-a-device-into-user-space-memory)谈到了userspace到DEVICE的DMA方法。   

1. userspace使用mmap方式
即driver在mmap中通过申请物理地址连续buffer并将其进行DMA mapping并将此地址返回给userspace。  
```
get_free_pages()->dma_map_page()
dma_alloc_coherent() -> dma_mmap_coherent()
```
2. userspace使用ioctl方式调用
使用 scatter/gather lists 来搜集可能物理地址不连续的buffer。 
```
get_user_pages() -> sg_set_page(sg, page_list[i], PAGE_SIZE, 0) -> dma_map_sg()  
```

## Driver Unregister  

PCI对 `struct pci_driver *` 指向的对象进行PCI驱动的注销。  
```
void pci_unregister_driver(struct pci_driver *drv);
```

而真正注销操作，需要在 `struct pci_driver *`对象的`remove`回调函数实现。  
对应的，使用完PCI设备后，卸载驱动的流程为：  

	Disable the device from generating IRQs
	Release the IRQ (free_irq())
	Stop all DMA activity
	Release DMA buffers (both streaming and coherent)
	Unregister from other subsystems (e.g. scsi or netdev)
	Release MMIO/PIO resources (pci_release_region)
	Disable the device (pci_disable_device())



# Linux PCI Driver Demo：

+ [a simple PCI driver, BAR IO Port with a virtual device](http://www.zarb.org/~trem/kernel/pci/pci-driver.c)  
+ [LDD3 pci_skel.c](https://github.com/jesstess/ldd3-examples/blob/master/examples/pci/pci_skel.c)  
+ [QEMU edu PCI, QEMU device](https://github.com/qemu/qemu/blob/v2.12.0/hw/misc/edu.c)
+ [QEMU edu PCI, specification](https://github.com/qemu/qemu/blob/v2.12.0/docs/specs/edu.txt)
+ [QEMU edu PCI, guest kernel module driver](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/master/kernel_modules/qemu_edu.c)   
+ [QEMU edu PCI, guest userspace script](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/master/rootfs_overlay/lkmc/qemu_edu.sh)  

```c

/**
 * This table holds the list of (VendorID,DeviceID) supported by this driver
 *
 */
static struct pci_device_id pci_ids[] = {
	{ PCI_DEVICE(0xabcd, 0xabcd), },
	{ 0, }
};

/**
 * 将pci设备导出到用户空间
 */
MODULE_DEVICE_TABLE(pci, pci_ids);

int pci_probe(struct pci_dev *dev,struct pci_device_id *id){
	struct pci_privdata *data=NULL;
	struct resource *resource=NULL;
	if(pci_enable_device(dev)){
	}
	data = kmalloc(sizeof(struct pci_privdata),GFP_KERNEL);
	data->phy_addr 	= pci_resource_start(dev,1);
	data->size      = pci_resource_len(dev,1);
	data->flags     = pci_resource_flags(dev,1);
	if(data->flags != IORESOURCE_MEM){
	}
	//resource=request_mem_region(data->phy_addr,data->size,NAME);
	resource = pci_request_region(data->phy_addr,data->size,NAME);
	if(!resource){
	}
	//6, io内存映射，映射到内核虚拟地址空间;
	data->addr = pci_iomap(data->phy_addr, 1, data->size);
	pci_set_drvdata(dev,data);
	//8,设置PCI设备为DMA主设备;
	pci_set_master(dev);
	return 0;
}

int  pci_remove(struct pci_dev *dev){
	//与probe函数相反的顺序将probe函数中的资源都释放掉;
	return 0;
}

static struct pci_driver pchar_driver = {
	.name		= "pci-char",
	.id_table	= pci_ids, 
	.probe		= pci_probe,
	.remove     = pci_remove,
};

static void __init pci_module_init(void){
	pci_register_driver(&pci_driver);
}

static void __exit pci_module_exit(void){
	pci_unregister_driver(&pci_driver);
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
12. [Writing a PCI Device Driver, A Tutorial with a QEMU Virtual Device](http://web.archive.org/web/20151115031755/http://nairobi-embedded.org/linux_pci_device_driver.html)  
13. [Kernel Doc: How To Write Linux PCI Drivers](https://www.kernel.org/doc/html/latest/PCI/pci.html)  
14. [Accessing PCI Regions](http://www.embeddedlinux.org.cn/essentiallinuxdevicedrivers/final/ch10lev1sec3.html)  