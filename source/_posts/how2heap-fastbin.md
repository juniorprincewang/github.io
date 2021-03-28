---
title: 堆溢出之fastbin
date: 2017-08-23 13:59:56
tags:
- 堆溢出
- pwn
- heap
categories:
- [security,pwn]
---

简单的fastbin堆溢出漏洞利用。

<!-- more -->

# 简介

关于Linux下堆管理，需要研读参考文献[1](http://www.freebuf.com/articles/system/104144.html), [2](http://www.freebuf.com/articles/security-management/105285.html)。

需要指出的是，fastbin是个单向链表，仅仅使用fd指针，用LIFO算法实现chuck的链接。fastbins数组中的每个fastbin元素均指向了链表尾部的chunk，而尾节点通过fd指向前一个节点。

# 一个栗子

关于fastbin溢出的例子并不多，下面这道题是很好的学习样例。

这里面观察`create`创建了0x24大小的堆块，并赋值给全局变量`ptr`，在`del`中释放掉`ptr`但是并未置空。
`free`后的chunk被fastbins回收，但是`ptr`指针任然指向了它，我们可以将我们想要的地址写入此块中，再`malloc`两次，获得的`ptr`指针就指向了我们的目标地址。进而可以做目标函数的GOT表地址覆盖。

这里有点困惑的是，在将堆内存赋值给`ptr`后，`ptr`指向的地址是它本身。这样我们可以写入任意要覆盖的地址。详细过程见下面。

free 后
```
-peda$ p main_arena.fastbinsY 
$1 = {0x0, 0xed9000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}
```

第一次edit后
```
gdb-peda$ x/8xg 0xed9000
0xed9000:	0x0000000000000000	0x0000000000000031
0xed9010:	0x0000000000602098	0x3131313131313131
0xed9020:	0x3131313131313131	0x0031313131313131
0xed9030:	0x0000000000000457	0x0000000000020fd1
```
第一次malloc
```
gdb-peda$ p main_arena.fastbinsY 
$2 = {0x0, 0x602098 <completed>, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}

gdb-peda$ x/8xg 0x602098
0x602098 <completed.6962>:	0x0000000000000000	0x0000000000000030
0x6020a8 <info+8>:	0x0000000000ed9010	0x0000000000000000
0x6020b8:	0x0000000000000000	0x0000000000000000
0x6020c8:	0x0000000000000000	0x0000000000000000
```

第二次malloc
```
gdb-peda$ p main_arena.fastbinsY 
$3 = {0x0, 0x1c6d010, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}

b-peda$ x/8xg 0x602098
0x602098 <completed.6962>:	0x0000000000000000	0x0000000000000031
0x6020a8 <info+8>:	0x0000000000ed9010	0x0000000000000000
0x6020b8:	0x0000000000000000	0x0000000000000000
0x6020c8:	0x0000000000000000	0x0000000000000000
```
但是执行这条语句
```
   0x400955 <create+4>:	mov    edi,0x24
   0x40095a <create+9>:	call   0x400700 <malloc@plt>
===>   0x40095f <create+14>:	
    mov    QWORD PTR [rip+0x201742],rax        # 0x6020a8 <info+8>
```
也就是
```
  result = malloc(0x24uLL);
  ptr = result;
```
 得到的ptr附近内存分布为
```
 gdb-peda$ x/8xg 0x602098
0x602098 <completed.6962>:	0x0000000000000000	0x0000000000000031
===> 0x6020a8 <info+8>:	0x00000000006020a8	0x0000000000000000
0x6020b8:	0x0000000000000000	0x0000000000000000
0x6020c8:	0x0000000000000000	0x0000000000000000
```
也就是说ptr指向了ptr自己。


我没弄懂的地方在于，*ptr*是指针，它保存的是地址，所以读入数据后，保存的地址是`0x00000000006020a8`，不是将读入的内容覆盖在`0x6020a8`上。

## 流程

1. malloc一个堆块
2. free掉该堆
3. 将`ptr_addr-16`(64位程序，prev_size和size各占8位)写入上述堆
4. malloc使fasbinsY指向`ptr_addr-16`
5. malloc使`ptr`指向bss段，也就是它自己。
6. 将atoi的GOT表地址写入ptr中。
7. 通过printf函数，泄露出atoi的实际内存地址。
8. 根据libc中atoi与system的相对偏移量，计算出system在内存中的实际地址。
9. 将system内存地址通过`ptr`写入atoi的内存地址。
10. 再次执行程序，在运行到`atoi`函数时，输入`/bin/sh`。

文件的下载地址[fastIsfast](/img/how2heap-fastbin/fastIsfast)，[libc-2.23-64.so](/img/how2heap-fastbin/libc-2.23-64.so)。


# 参考文章
[1] [Linux堆内存管理深入分析（上）](http://www.freebuf.com/articles/system/104144.html)
[2] [Linux堆内存管理深入分析（下）](http://www.freebuf.com/articles/security-management/105285.html)
[3] [XCTF Day 11](https://www.xctf.org.cn/library/details/66bf2f67bdaeb06136a3624e632a548441fb4b38/)
