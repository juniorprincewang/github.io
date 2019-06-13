---
title: Linux内核内存分配
date: 2018-11-24 11:13:48
tags:
- kmalloc
categories:
- [linux]
---
本篇博客整理Linux内核的内存分配相关知识，包括 `kmalloc` ...(补充)
<!-- more -->


# kmalloc
```
#include <linux/slab.h>
void *kmalloc(size_t size, int flags);
```

## 参数flags

```
GFP_ATOMIC
GFP_KERNEL
GFP_USER
```

### 内存区域

分为 正常内存、 DMA内存和高端内存。



## 参数size

Linux 处理内存分配通过创建一套固定大小的内存对象池， `kmalloc` 最大分配内存是 **128 KB** ，如果想要分配更多，还有其他方法。


# 后备缓存 Lookaside Caches

```
kmem_cache_t *kmem_cache_create(const char *name, size_t size,
	size_t offset,
	unsigned long flags,
	void (*constructor)(void *, kmem_cache_t *,
		unsigned long flags),
	void (*destructor)(void *, kmem_cache_t *,
		unsigned long flags));
```

基于 Slab 缓存的 scull代码在源码`scullc`中。

## 内存池 mempools

内核中的内存，有的地方不允许分配失败，因此，内核提供了内存池(`mempool`)的抽象。它是一类后备缓存。
驱动代码中的 `mempools` 的使用应当少使用。

# 请求页get_free_page

```
get_zeroed_page(unsigned int flags);
__get_free_page(unsigned int flags);
__get_free_pages(unsigned int flags, unsigned int order);

void free_page(unsigned long addr);
void free_pages(unsigned long addr, unsigned long order);
```
`flags` 和 `kmalloc`的参数一致， `order` 表示在请求的或释放的页数的以 2 为底的对数，比如`order`=2，分配8个页。

性能提升有一些，但是主要是有效的内存使用率提高了，最大优势是内存使用自由。

使用整页的 scull: `scullp`


### alloc_page 接口

Linux 页分配器的真正核心是一个称为 alloc_pages_node 的函数:
```
struct page *alloc_pages_node(int nid, unsigned int flags,
	unsigned int order);

```
`nid` 是要分配内存的 NUMA 节点 ID, `flags` 是通常的 `GFP_` 分配标志, 以及 `order` 是分配的大小。 返回值是一个指向描述分
配的内存的第一个(可能许多)页结构的指针，或者，失败时返回 NULL。
```
struct page *alloc_pages(unsigned int flags, unsigned int order);
struct page *alloc_page(unsigned int flags);


void __free_page(struct page *page);
void __free_pages(struct page *page, unsigned int order);
void free_hot_page(struct page *page);
void free_cold_page(struct page *page);
```

### vmalloc

`vmalloc` 不鼓励使用,从 vmalloc 获得的内存用起来稍微低效些, 并且, 在某些体系上, 留给 vmalloc 的地址空间的数量相对小。
```
#include <linux/vmalloc.h>
void *vmalloc(unsigned long size);
void vfree(void * addr);
void *ioremap(unsigned long offset, unsigned long size);
void iounmap(void * addr);
```

`kmalloc` 和 `_get_free_pages` 返回的内存地址也是虚拟地址. 它们的实际值在寻址物理地址前仍然由 MMU (内存管理单元, 常常是 CPU 的一部分)管理。

`vmalloc` 的一个小的缺点在于它无法在原子上下文中使用。

使用虚拟地址的 scull : `scullv`。

# 获得大量缓冲

## 在启动时获得专用的缓冲

如果你真的需要一个大的物理上连续的缓冲, 最好的方法是在启动时请求内存来分配它。在启动时分配是获得连续内存页而避开 `__get_free_pages` 施加的对缓冲大小限制的唯一 方法。

启动时内存分配通过调用下面一个函数进行:
```
#include <linux/bootmem.h>
void *alloc_bootmem(unsigned long size);
void *alloc_bootmem_low(unsigned long size);
void *alloc_bootmem_pages(unsigned long size);
void *alloc_bootmem_low_pages(unsigned long size);

void free_bootmem(unsigned long addr, unsigned long size);
```

