---
title: 编译安装报错 "virtual memory exhausted Cannot allocate memory"
date: 2018-07-29 16:55:23
tags:
- linux
- virtual memory
categories:
- solutions
---

在1GB内存的阿里云主机中，编译cmake时候报错，`virtual memory exhausted: Cannot allocate memory` 。 
<!-- more -->

# 原因

虚拟机安装时没有设置swap或者设置内存太小。 

```
# free -mh
              total        used        free      shared  buff/cache   available
Mem:           992M        157M        717M        2.6M        117M        698M
Swap:            0B          0B          0B
```
1GB内存足够编译软件，这里主要是没有设置 `swap`。

`swap` 是啥？

# 解决办法

增加swap大小。

# 创建swap文件（目录可以自己指定
```
# dd if=/dev/zero of=/var/swap bs=1024 count=1024000  
1024000+0 records in
1024000+0 records out
1048576000 bytes (1.0 GB, 1000 MiB) copied, 9.40488 s, 111 MB/s
```
建立swap
```
# mkswap  /var/swap 
Setting up swapspace version 1, size = 1000 MiB (1048571904 bytes)
no label, UUID=83ddb587-df84-402d-84af-ce21689b3235
```
启动swap
```
# swapon /var/swap
swapon: /var/swap: insecure permissions 0644, 0600 suggested.
```
现在再看内存使用情况
```
# free -mh
              total        used        free      shared  buff/cache   available
Mem:           992M        158M         67M        2.6M        766M        679M
Swap:          999M          0B        999M
```

参考
1. [编译安装报错"virtual memory exhausted: Cannot allocate memory"](http://muchfly.iteye.com/blog/2296506)
2. [virtual memory exhausted: Cannot allocate memory](http://www.cnblogs.com/chenpingzhao/p/4820814.html)
