---
title: Linux dd 命令
date: 2018-11-20 09:41:04
tags:
- dd
categories:
- linux
---
Linux中dd命令用于复制文件并对原文件的内容进行转换和格式化处理。dd命令功能很强大，本篇博客按照网上的资源整理下来。

<!-- more -->

# 参数

```
bs=<字节数>：同时设置读入/输出的块大小为bytes个字节。
ibs=<字节数>：一次读入bytes个字节，即指定一个块大小为bytes个字节。
obs=<字节数>：一次输出bytes个字节，即指定一个块大小为bytes个字节。
cbs=<字节数>：转换时，每次只转换指定的字节数。
count=<区块数>：仅读取指定的区块数。
if=文件名：输入文件名，缺省为标准输入。即指定源文件。
of=<文件>：of=文件名：输出文件名，缺省为标准输出。即指定目的文件。
seek=<区块数>：一开始输出时，跳过指定的区块数；
skip=<区块数>：一开始读取时，跳过指定的区块数；
conv=<关键字>：指定文件转换的方式；
	ascii：转换ebcdic为ascii
	ebcdic：转换ascii为ebcdic
	ibm：转换ascii为alternate ebcdic
	block：把每一行转换为长度为cbs，不足部分用空格填充
	unblock：使每一行的长度都为cbs，不足部分用空格填充
	lcase：把大写字符转换为小写字符
	ucase：把小写字符转换为大写字符
	swab：交换输入的每对字节
	noerror：出错时不停止
	notrunc：不截短输出文件
	sync：将每个输入块填充到ibs个字节，不足部分用空（NUL）字符补齐。
status=LEVEL，要打印到stderr的信息的级别; 'none'除了错误消息之外都不会打印，'noxfer'会不打印最终的统计信息，'progress'会定期显示转移统计信息。
```

块大小的计量单位是1字节，也有其他的计量单位，可以添加的后缀为： `c` = 1字节（1B)， `w`= 2字节（2B）, `b`= 1块（512B）, `k`= 1千字节（1024B）, `M` = 兆字节（1024KB）, `G`=吉字节（1024MB）。

# 例子

+ 将备份文件恢复到指定盘
```
dd if=/root/image of=/dev/hdb
```

+ 创建一个大小为256M的文件，其中 `/dev/zero` 是一个字符设备，会不断返回0值字节（\0）：
```
dd if=/dev/zero of=/swapfile bs=1M count=256
```

+ 将testfile文件中的所有英文字母转换为大写，然后转成为testfile_1文件：
```
dd if=testfile of=testfile_1 conv=ucase 
```

+ 读取pci设备文件的寄存器值

```
dd bs=4 status=none if=/dev/pci_dev count=1 skip=0 | od -An -t x1
```
+ 写入pci设备文件中指定位置
```
printf `\xFF\xFF\xFF\xFF` | dd bs=4 status=none of=/dev/pci_dev count=1 seek=2
```

# 参考
1. [dd命令](http://man.linuxde.net/dd)
