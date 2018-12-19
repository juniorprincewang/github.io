---
title: QEMU-KVM内存虚拟化
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

![QEMU 内存结构图](../QEMU内存虚拟化/QEMU memory.png)

## PCDIMMDevice
PC DIMM内存设备模拟，通过 QOM(qemu object model) 定义的虚拟内存条。可通过 QEMU 命令行进行管理。通过增加/移除该对象实现 VM 中内存的热插拔。
```
/include/hw/mem/pc-dimm.h
/**
 * PCDIMMDevice:
 * @addr: starting guest physical address, where @PCDIMMDevice is mapped.
 *         Default value: 0, means that address is auto-allocated.
 * @node: numa node to which @PCDIMMDevice is attached.
 * @slot: slot number into which @PCDIMMDevice is plugged in.
 *        Default value: -1, means that slot is auto-allocated.
 * @hostmem: host memory backend providing memory for @PCDIMMDevice
 */
typedef struct PCDIMMDevice {
    /* private */
    DeviceState parent_obj;

    /* public */
    uint64_t addr;          // 映射到的起始 GPA
    uint32_t node;             // 映射到的 numa 节点
    int32_t slot;               // 插入的内存槽编号，默认为 -1，表示自动分配
    HostMemoryBackend *hostmem;     // 对应的 backend
} PCDIMMDevice;
```

## HostMemoryBackend
通过 QOM 定义的一段 Host 内存，为虚拟内存条提供内存。可通过 QMP 或 QEMU 命令行进行管理。
```
/**
 * @HostMemoryBackend
 *
 * @parent: opaque parent object container
 * @size: amount of memory backend provides
 * @mr: MemoryRegion representing host memory belonging to backend
 */
struct HostMemoryBackend {
    /* private */
    Object parent;

    /* protected */
    uint64_t size;      // 提供内存大小
    bool merge, dump;
    bool prealloc, force_prealloc, is_mapped, share;
    DECLARE_BITMAP(host_nodes, MAX_NODES + 1);
    HostMemPolicy policy;

    MemoryRegion mr;        // 拥有的 MemoryRegion
};
```

## MemoryRegion

```
struct MemoryRegion {
    Object parent_obj;           // 继承自 Object

    /* All fields are private - violators will be prosecuted */

    /* The following fields should fit in a cache line */
    bool romd_mode;
    bool ram;
    bool subpage;
    bool readonly; /* For RAM regions */
    bool rom_device;
    bool flush_coalesced_mmio;
    bool global_locking;
    uint8_t dirty_log_mask;
    bool is_iommu;
    RAMBlock *ram_block;        // 指向对应的 RAMBlock
    Object *owner;

    const MemoryRegionOps *ops;
    void *opaque;
    MemoryRegion *container;    // 指向父 MemoryRegion
    Int128 size;
    hwaddr addr;
    void (*destructor)(MemoryRegion *mr);
    uint64_t align;
    bool terminates;
    bool ram_device;
    bool enabled;
    bool warning_printed; /* For reservations */
    uint8_t vga_logging_count;
    MemoryRegion *alias;         // 指向实体 MemoryRegion
    hwaddr alias_offset;        // 起始地址 (GPA) 在实体 MemoryRegion 中的偏移量
    int32_t priority;
    QTAILQ_HEAD(subregions, MemoryRegion) subregions;       // subregion 链表
    QTAILQ_ENTRY(MemoryRegion) subregions_link;
    QTAILQ_HEAD(coalesced_ranges, CoalescedMemoryRange) coalesced;
    const char *name;
    unsigned ioeventfd_nb;
    MemoryRegionIoeventfd *ioeventfds;
};
```
`MemoryRegion` 是树状结构，有多种类型，可以表示一段 `ram` ，`rom` ，`MMIO` 。 `alias` 表示一个 `MemoryRegion` 的一部分区域。
`MemoryRegion` 也可以表示一个container，这就表示它只是其他若干个 `MemoryRegion` 的容器。在 `MemoryRegion` 中，`ram_block`表示的是分配的实际内存。

`address_space_memory` 的 `root` 为 `system_memory` ，`address_space_io` 的 `root` 为 `system_io` 。
```
static MemoryRegion *system_memory;
static MemoryRegion *system_io;
```

## AddressSpace
```
/include/exec/memory.h
/**
 * AddressSpace: describes a mapping of addresses to #MemoryRegion objects
 */
struct AddressSpace {
    /* All fields are private. */
    struct rcu_head rcu;
    char *name;
    MemoryRegion *root;

    /* Accessed via RCU.  */
    struct FlatView *current_map;    // 指向当前维护的 FlatView，在 address_space_update_topology 时作为 old 比较

    int ioeventfd_nb;
    struct MemoryRegionIoeventfd *ioeventfds;
    QTAILQ_HEAD(memory_listeners_as, MemoryListener) listeners;
    QTAILQ_ENTRY(AddressSpace) address_spaces_link;
};
```

`AddressSpace` 表示的CPU/设备看到的地址空间，比如内存地址空间 `AddressSpace address_space_memory;` 和IO地址空间 `AddressSpace address_space_io;` 。
每个 `AddressSpace` 一般包含一系列 `MemoryRegion` ： AddressSpace 的 `root` 指向根级 `MemoryRegion` ，该 `MemoryRegion` 有可能有自己的若干个 `subregion` ，于是形成树状结构。
所有的 `AddressSpace` 通过结构中的 `address_spaces_link` 连接成链表，表头保存在全局的 `AddressSpace` 结构中。



## RAMBlock
```
/include/exec/ram_addr.h

struct RAMBlock {
    struct rcu_head rcu;                // 用于保护 Read-Copy-Update
    struct MemoryRegion *mr;            // 对应的 MemoryRegion
    uint8_t *host;                       // 对应的 HVA
    ram_addr_t offset;                   // 在 ram_list 地址空间中的偏移 (要把前面 block 的 size 都加起来)
    ram_addr_t used_length;             // 当前使用的长度
    ram_addr_t max_length;                  
    void (*resized)(const char*, uint64_t length, void *host);
    uint32_t flags;
    /* Protected by iothread lock.  */
    char idstr[256];
    /* RCU-enabled, writes protected by the ramlist lock */
    QLIST_ENTRY(RAMBlock) next;
    QLIST_HEAD(, RAMBlockNotifier) ramblock_notifiers;
    int fd;                                 // 映射文件的文件描述符
    size_t page_size;                       // page 大小，一般和 host 保持一致
    /* dirty bitmap used during migration */
    unsigned long *bmap;
    /* bitmap of pages that haven't been sent even once
     * only maintained and used in postcopy at the moment
     * where it's used to send the dirtymap at the start
     * of the postcopy phase
     */
    unsigned long *unsentmap;
    /* bitmap of already received pages in postcopy */
    unsigned long *receivedmap;
};
```
在这里，`host` 指向了动态分配的内存，用于表示实际的虚拟机物理内存，而 `offset` 表示了这块内存在虚拟机物理内存中的偏移。每一个 `ram_block` 还会被连接到全局的 `ram_list` 链表上。 

`AddressSpace` 的 `root`及其子树形成了一个虚拟机的物理地址，但是在往kvm进行设置的时候，需要将其转换为一个平坦的地址模型，也就是从0开始的。这个就用`FlatView` 表示，一个 `AddressSpace` 对应一个 `FlatView` 。

```
/* Flattened global view of current active memory hierarchy.  Kept in sorted
 * order.
 */
struct FlatView {
    struct rcu_head rcu;
    unsigned ref;
    FlatRange *ranges;
    unsigned nr;
    unsigned nr_allocated;
    struct AddressSpaceDispatch *dispatch;
    MemoryRegion *root;
};
```

## RAMList
`ram_list` 是一个全局变量，以链表的形式维护了所有的 `RAMBlock` 。

```
include/exec/ramlist.h
typedef struct RAMList {
    QemuMutex mutex;
    RAMBlock *mru_block;
    /* RCU-enabled, writes protected by the ramlist lock. */
    QLIST_HEAD(, RAMBlock) blocks;
    DirtyMemoryBlocks *dirty_memory[DIRTY_MEMORY_NUM];
    uint32_t version;
    QLIST_HEAD(, RAMBlockNotifier) ramblock_notifiers;
} RAMList;

exe.c
RAMList ram_list = { .blocks = QLIST_HEAD_INITIALIZER(ram_list.blocks) };
```



## MemoryRegionSection

在内存虚拟化中，还有一个重要的结构是 `MemoryRegionSection` ，这个结构通过函数 `section_from_flat_range` 可由 `FlatRange` 转换过来。
```
/**
 * MemoryRegionSection: describes a fragment of a #MemoryRegion
 *
 * @mr: the region, or %NULL if empty
 * @fv: the flat view of the address space the region is mapped in
 * @offset_within_region: the beginning of the section, relative to @mr's start
 * @size: the size of the section; will not exceed @mr's boundaries
 * @offset_within_address_space: the address of the first byte of the section
 *     relative to the region's address space
 * @readonly: writes to this section are ignored
 */
struct MemoryRegionSection {
    MemoryRegion *mr;   // 指向所属 MemoryRegion
    FlatView *fv;
    hwaddr offset_within_region;     // 起始地址 (HVA) 在 MemoryRegion 内的偏移量
    Int128 size;
    hwaddr offset_within_address_space;  // 在 AddressSpace 内的偏移量，如果该 AddressSpace 为系统内存，则为 GPA 起始地址
    bool readonly;
};
```

# 参考
1. [QEMU内存虚拟化源码分析](https://www.anquanke.com/post/id/86412)
2. [qemu-kvm内存虚拟化1](https://www.cnblogs.com/ck1020/p/6729224.html)
3. [QEMU学习笔记——内存](https://www.binss.me/blog/qemu-note-of-memory/)
4. [QEMU memory](https://github.com/GiantVM/doc/blob/master/memory.md)
5. [QEMU Internals: How guest physical RAM works](http://blog.vmsplice.net/2016/01/qemu-internals-how-guest-physical-ram.html)