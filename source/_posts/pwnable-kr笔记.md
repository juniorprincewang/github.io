---
title: pwnable.kr笔记
date: 2017-08-13 15:23:28
tags:
- pwnable.kr
categories:
- ctf
- pwn
---

pwnable.kr算是pwn入门级别的题目，做一遍记录下大概的知识点。
<!-- more -->
# 大致流程

1. 检查软件的详细信息，得到是32位或64位的ELF。
```
file software
或者
binwalk software
```
2. 运行软件，了解软件的流程
3. 使用gdb工具调试软件
```
# 加载软件，不显示额外信息
gdb -q software
# 加载
```

将代码重新编译成可执行文件，关闭gcc编译器优化以启用缓冲区溢出。

1. 禁用ASLR
```
sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'
```

2. 禁用canary：
```
gcc overflow.c -o overflow -fno-stack-protector
```



# [Toddler's Bottle]

## fd

## collision

## bof

## flag

	Papa brought me a packed present! let's open it.
	Download : http://pwnable.kr/bin/flag

	This is reversing task. all you need is binary

这道题说的很明确，对软件逆向，而且是个`packed`软件。


运行软件


## random

本题就考察的是对rand函数的理解。随机数生成器需要设置随机种子。如果rand未设置，rand会在调用时自动设置随机数种子为1。rand()产生的是伪随机数，每次执行的结果相同。若要不同，需要调用srand()初始化函数。
利用gdb调试，rand()每次确实生成相同的数`0x6b8b4567`。
所以可以利用异或得：
```
key = 0x6b8b4567^0xdeadbeef = 3039230856
```

## unlink


