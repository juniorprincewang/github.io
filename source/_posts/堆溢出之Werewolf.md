---
title: 堆溢出之Werewolf
date: 2017-09-01 15:11:42
tags:
- 堆溢出
categories:
- [pwn]
---

堆溢出small bin的unlink利用

<!-- more -->

这里需要安装gdb的堆调试工具`libheap`。安装步骤详见<https://github.com/cloudburst/libheap/blob/master/docs/InstallGuide.md>。
我这里是ubuntu操作系统，先安装`pip3`。然后

```
sudo apt-get install python3-pip
```
需要在`~/.gdbinit`中写入，这里的python路径通过`pip3 show libheap`来发现。
```
python import sys
python sys.path.append("/home/doubleloop/Envs/libheap/lib/python3.4/site-packages")
python from libheap import *
```

关闭`alarm(0x38u);`
```
gdb-peda$ handle SIGALRM print nopass
Signal        Stop	Print	Pass to program	Description
SIGALRM       No	Yes	No		Alarm clock
```


