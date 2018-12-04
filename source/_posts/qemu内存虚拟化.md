---
title: qemu-kvm内存虚拟化
date: 2018-07-20 16:59:26
tags:
- QEMU
- vm
categories:
- QEMU
---
qemu-kvm的内存虚拟化方案，是由qemu和kvm共同完成的，所以可以分为两部分。qemu完成内存的申请，kvm实现内存的管理。内部实现及其复杂，本篇博客尽量整理搜集相关资料，目的是分析出如何将 `GPA` 转换成 `HVA` 。
<!-- more -->

# 词汇约定

|缩写|意义|
|----|-----|
|VA | Virtual Address, 虚拟地址|
|PA | Physical Address, 物理地址 |
|PML4 | Page Map Level 4|
|PDPT | Page Directory Table|
|PD | Page Directory|
|PT| Page Table|
|PGD| Page Global Directory|
|PUD| Page Upper Directory|
|PMD| Page Middle Directory|
|GVA| Guest Virtual Address|
|GPA| Guest Physical Address|
|HVA| Host Virtual Address|
|HPA| Host Physical Address|
|GFN| Guest Frame Number|
|PFN| Host Page Frame Number|
|SPT| Shadow Page Table，影子页表|

# 页表

64位CPU上支持 48 位的虚拟地址寻址空间，和 52 位的物理地址寻址空间。Linux采用4级页表机制将虚拟地址（VA）转换成物理地址(PA)，先从页表的 `基地址寄存器 (CR3)` 中读取页表的起始地址，然后加上页号得到对应页的页表项，从中取出页的物理地址，再加上偏移量得到 PA。

分级的查询页表过程为：
Page Map Level 4(PML4) => Page Directory Pointer Table(PDPT) => Page Directory(PD) => Page Table(PT)

在某些地方被称为： Page Global Directory(PGD) => Page Upper Directory(PUD) => Page Middle Directory(PMD) => Page Table(PT)

# QEMU内存虚拟化

QEMU 利用 mmap 系统调用，在进程的虚拟地址空间中申请连续的大小的空间，作为 Guest 的物理内存。

在这样的架构下，内存地址访问有四层映射：

> GVA -> GPA -> HVA -> HPA

GVA - GPA 的映射由 guest OS 负责维护，而 HVA - HPA 由 host OS 负责维护。
我们重点要研究的是 `GPA -> HVA` 的映射。
常用的实现有 `SPT(Shadow Page Table)` 和 `EPT/NPT` ，前者通过软件维护影子页表，后者通过硬件特性实现二级映射。

## SPT

## EPT/NPT

# 内存数据结构

QEMU虚拟内存最重要的几个数据结构 `AddressSpace` 、 `MemoryRegion` 、 `MemoryRegionSection` 、 `RAMBlock` 、 `kvm_userspace_memory_region` 。

## AddressSpace
```
/**
 * AddressSpace: describes a mapping of addresses to #MemoryRegion objects
 */
struct AddressSpace {
    /* All fields are private. */
    char *name;
    MemoryRegion *root;
    struct FlatView *current_map;
    int ioeventfd_nb;
    struct MemoryRegionIoeventfd *ioeventfds;
    struct AddressSpaceDispatch *dispatch;
    struct AddressSpaceDispatch *next_dispatch;
    MemoryListener dispatch_listener;

    QTAILQ_ENTRY(AddressSpace) address_spaces_link;
};
```

`AddressSpace` 表示的CPU/设备看到的地址空间，比如内存地址空间 `AddressSpace address_space_memory;` 和IO地址空间 `AddressSpace address_space_io;` 。
每个 `AddressSpace` 一般包含一系列 MemoryRegion ： AddressSpace 的 `root` 指向根级 MemoryRegion ，该 MemoryRegion 有可能有自己的若干个 subregion ，于是形成树状结构。
所有的AddressSpace通过结构中的address_spaces_link连接成链表，表头保存在全局的AddressSpace结构中。

## MemoryRegion

```
struct MemoryRegion {
    /* All fields are private - violators will be prosecuted */
    const MemoryRegionOps *ops;
    const MemoryRegionIOMMUOps *iommu_ops;
    void *opaque;
    struct Object *owner;
    MemoryRegion *parent;
    Int128 size;
    hwaddr addr;
    void (*destructor)(MemoryRegion *mr);
    ram_addr_t ram_addr;
    bool subpage;
    bool terminates;
    bool romd_mode;
    bool ram;
    bool readonly; /* For RAM regions */
    bool enabled;
    bool rom_device;
    bool warning_printed; /* For reservations */
    bool flush_coalesced_mmio;
    MemoryRegion *alias;
    hwaddr alias_offset;
    int priority;
    bool may_overlap;
    QTAILQ_HEAD(subregions, MemoryRegion) subregions;
    QTAILQ_ENTRY(MemoryRegion) subregions_link;
    QTAILQ_HEAD(coalesced_ranges, CoalescedMemoryRange) coalesced;
    const char *name;
    uint8_t dirty_log_mask;
    unsigned ioeventfd_nb;
    MemoryRegionIoeventfd *ioeventfds;
    NotifierList iommu_notify;
};
```
`MemoryRegion` 是树状结构，有多种类型，可以表示一段 `ram` ，`rom` ，`MMIO` 。 `alias` 表示一个MemoryRegion的一部分区域。


```
static MemoryRegion *system_memory;
static MemoryRegion *system_io;
```



## MemoryRegionSection

```
/**
 * MemoryRegionSection: describes a fragment of a #MemoryRegion
 *
 * @mr: the region, or %NULL if empty
 * @address_space: the address space the region is mapped in
 * @offset_within_region: the beginning of the section, relative to @mr's start
 * @size: the size of the section; will not exceed @mr's boundaries
 * @offset_within_address_space: the address of the first byte of the section
 *     relative to the region's address space
 * @readonly: writes to this section are ignored
 */
struct MemoryRegionSection {
    MemoryRegion *mr;
    AddressSpace *address_space;
    hwaddr offset_within_region;
    Int128 size;
    hwaddr offset_within_address_space;
    bool readonly;
};
```

# 参考
1. [QEMU内存虚拟化源码分析](https://www.anquanke.com/post/id/86412)
2. [qemu-kvm内存虚拟化1](https://www.cnblogs.com/ck1020/p/6729224.html)
3. [QEMU学习笔记——内存](https://www.binss.me/blog/qemu-note-of-memory/)
4. [QEMU memory](https://github.com/GiantVM/doc/blob/master/memory.md)