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


## unlink

