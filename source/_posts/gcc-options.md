---
title: gcc options
date: 2021-04-09 17:00:05
tags:
- gcc
categories:
- [linux]
---

本文总结了GCC选线中常见的选项，包括控制预处理选项（`-D`、`-U`）。  

<!-- more -->

# GCC 选项

[GCC Options列表](https://gcc.gnu.org/onlinedocs/gcc/Option-Index.html)。  

## 预处理选项

和宏定义相关的 `-D` 与 `-U` 。  

+ `-D name`
预定义宏`name`，定义为1。  
+ `-D name=definition`
定义宏`name`为`definition`。
+ `-U name`
取消**内置**或之前`-D`选项的定义的宏。  

其他选项参见 [3.13 Options Controlling the Preprocessor](https://gcc.gnu.org/onlinedocs/gcc/Preprocessor-Options.html)。  