---
title: Linux Kernel 中的 radix tree
date: 2020-09-02 10:50:53
tags:
- radix
categories:
- linux
---

Linux基数树（radix tree）是将 指针 与 long 整数键值相关联的机制，它存储有效率，并且可快速查询，用于指针与整数值的映射（如：IDR机制）、内存管理等。   
radix树为稀疏树提供了有效的存储，代替固定尺寸数组提供了键值到指针的快速查找。   
<!-- more -->

Radix tree 是一种多叉搜索树，树的叶子结点是实际的数据条目。每个结点有一个固定的、 2^n 指针指向子结点 (每个指针称为槽 slot，n 为划分的基的大小)。  

MMU的 page table walk 可以看作一种 radix tree，将虚拟地址划分成不同的字段来逐级访问。  
# 内核中的 Radix Tree  

Linux 4.20 之前的内核使用 Radix Tree 管理很多内核基础数据结构，其中包括 IDR 机制。 但 Linux 4.20 之后，内核采用新的数据结构 xarray 代替了 Radix Tree。内核关于 Radix Tree 的源码位于：

[include/linux/radix-tree.h](https://github.com/torvalds/linux/blob/v4.18/include/linux/radix-tree.h)  
[lib/radix-tree.c](https://github.com/torvalds/linux/blob/v4.18/lib/radix-tree.c)  

在 Linux 4.20 之前的内核中，Radix Tree 作为重要的基础数据，内核定义了一下数据结构 对 Radix Tree 进行维护。

# 数据结构  

```c
struct radix_tree_node {
    unsigned char   shift;      /* Bits remaining in each slot */
    unsigned char   offset;     /* Slot offset in parent */
    unsigned char   count;      /* Total entry count */
    unsigned char   exceptional;    /* Exceptional entry count */
    struct radix_tree_node *parent;     /* Used when ascending tree */
    struct radix_tree_root *root;       /* The tree we belong to */
    union {
        struct list_head private_list;  /* For tree user */
        struct rcu_head rcu_head;   /* Used when freeing node */
    };
    void __rcu  *slots[RADIX_TREE_MAP_SIZE];
    unsigned long   tags[RADIX_TREE_MAX_TAGS][RADIX_TREE_TAG_LONGS];
};
```

+ `shift` 成员用于指向 当前节点占用所有的偏移； 
+ `offset` 存储该节点在父节点的 slot 的偏移； 
+ `count` 表示 当前节点有多少个 slot 已经被使用； 
+ `exceptional` 表示当前节点有多少个 exceptional 节点； 
+ `parent` 指向父节点；参数 root 指向根节点；参数 slots 是数组，数组的成员 指向下一级的节点； 
+ `tags` 用于标识当前节点包含了指定 tag 的节点数。  
+ `slots` 指向了孩子节点，RADIX_TREE_MAP_SIZE通常为 `1<<4` 或者 `1<<6`。
+ `RADIX_TREE_MAX_TAGS` 为 3，即最多支持3种标签。  
+ `RADIX_TREE_TAG_LONGS` 的长度使得可以放下所有子节点的tag（一个tag占1位），最多 RADIX_TREE_MAP_SIZE 个。



```c
struct radix_tree_root {
    spinlock_t              xa_lock;
    gfp_t           gfp_mask;
    struct radix_tree_node  __rcu *rnode;
};
```

树的根节点由 `struct radix_tree_root` 来描述。   
+ `xa_lock` 是一个自旋锁；  
+ `gfp_mask` 用于标识 radix-tree 的属性以及 radix-tree 节点申请 内核的标识，比如 `GFP_ATOMIC`。
+ `rnode` 指向 radix-tree 的根节点。  

# 原理  

作为树结构，树的根节点由 结构体 `struct radix_tree_root` 表示，每个树节点由 `struct radix_tree_node` 进行维护，树的叶子节点是保存的 指针。  

radix tree 的slot记录着下一层的指针，叶子节点是  `0x00 - data pointer`， 内部节点是 `0x01 - internal entry` ，exceptional 节点 `0x10 - exceptional entry`，exceptional 节点与 internal 节点类似。  

存储原理是 将长整型index 按照从左往右每 6 bits 为一个字段做索引，逐层找到internal node的slots入口，最终找到 存储的指针。  

为了增加检索效率，内部节点结构体使用bitmap记录slot使用情况，即 tag 成员。

# radix tree opts  

+ 初始化  

首先需要先声明 `struct radix_tree_root my_radix_tree;` 根节点变量，然后再通过宏
`INIT_RADIX_TREE(root, mask)` 初始化。  

`mask` 是 `gfp_mask` 。  

+ radix_tree_insert

```c
int radix_tree_insert(struct radix_tree_root *root, unsigned long index, void *item);
```

将一个新的 radix_tree_node 添加到 radix-tree。  
首先判断要加入的index是否超过了 radix tree 的maxindex。超过了就需要添加新的internal node，这样会增加树的高度。添加方式就是分配新的internal node，作为新的root node入口，将老root node入口作为新node的slots[0]元素，新node的shift需要在老shift上加6。  

+ radix_tree_lookup  

```c
void *radix_tree_lookup(const struct radix_tree_root *, unsigned long index);
```

radix-tree 将 index 拆分成多个索引，从根节点开始，在每一层节点的 slots 数组里找到指定的 入口地址，然后进入下一层继续查找，直到找到最后一个 slot，如果找到，那么就返回 私有数据；如果没有找到，则返回对应的错误码。  

+ radix_tree_delete

```c
void *radix_tree_delete(struct radix_tree_root *root, unsigned long index);
```

`radix_tree_delete()` 用于删除一个 radix tree 节点。  
如果删除节点要引起tree shrink，那么树的高度就降低。  
将最后 node 的slots 值替换成 NULL。  

+ iterator  

```c
/**
 * radix_tree_for_each_slot - iterate over non-empty slots
 *
 * @slot:   the void** variable for pointer to slot
 * @root:   the struct radix_tree_root pointer
 * @iter:   the struct radix_tree_iter pointer
 * @start:  iteration starting index
 *
 * @slot points to radix tree slot, @iter->index contains its index.
 */
#define radix_tree_for_each_slot(slot, root, iter, start)       \
    for (slot = radix_tree_iter_init(iter, start) ;         \
         slot || (slot = radix_tree_next_chunk(root, iter, 0)) ;    \
         slot = radix_tree_next_slot(slot, iter, 0))
```

# 参考资料  

[Data Structures in the Linux Kernel - Radix tree](https://0xax.gitbooks.io/linux-insides/content/DataStructures/linux-datastructures-2.html)  
[详解Linux内核Radix树算法的实现](http://sourcelink.top/2019/09/26/linux-kernel-radix-tree-analysis/)   
[基数树(radix tree) 详细内容](https://blog.csdn.net/joker0910/article/details/8250085)   
[BiscuitOS: Radix Tree 原理和代码](https://biscuitos.github.io/blog/RADIX-TREE/)  
[BiscuitOS： Radix-Tree-api](https://biscuitos.github.io/blog/RADIX-TREE_SourceAPI/)   