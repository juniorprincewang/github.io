---
title: centos启动加载驱动
date: 2018-04-03 11:31:36
tags:
- centos
- driver
categories:
- linux
---

本篇博客记载了在centos启动时，自动加载内核模块的探索过程。
<!--more-->

要想在CentOS中自动加载内核模块 `mimosa.ko` ，需要在 `/etc/sysconfig/modules/` 目录中增加一个脚本，在此脚本中加载所需的模块。

先切换到 `root` 用户。

```
cd /etc/sysconfig/modules/
vim mimosa.modules
```
在脚本代码为：

```
#！/bin/sh 
/sbin/modinfo -F filename mimosa > /dev/null 2>&1 
if [ $? -eq 0 ]; then 
    /sbin/modprobe mimosa 
fi
```

修改脚本为可执行。
```
chmod 755 mimosa.modules   //这一步至关重要
```

**注意**

脚本中的 `mimosa` 不带 `.ko` ，这里搜索的是名字。因为 `modprobe` 要到 `/lib/modules/$(uname -r)/` 中搜索内核模块，所以将 `mimosa.ko` 拷贝到 `/lib/modules/$(uname -r)/kernel/` 中，新建文件夹 `misc`。

```
cd /lib/modules/$(uname -r)/kernel
mkdir misc
```

然后执行 `mimosa.modules` 检查脚本是否报错。

果然 ！

> FATAL: Module mimosa not found 

内核加载模块的条目可以查看 `/lib/modules/$(uname -r)/modules.dep` ，并没有找到我们的 `mimosa.ko`。

这里要执行 `depmod` ，这条命令的功能是读取在 `/lib/modules/$(uname -r)/` 目录下的所有模块，分析可加载模块的依赖性，生成 `modules.dep` 文件和映射文件。

执行完后查看 `modules.dep` 文件发现 `mimosa.ko` 加入其中。

再次执行脚本，或者执行 `modprobe mimosa` 成功。*此处不是mimosa.ko*。

参考
[1] [modprobe XXX not found 解决与Depmod命令](https://blog.csdn.net/yeqishi/article/details/5439619)
[2] [“FATAL: Module not found error” using modprobe
](https://stackoverflow.com/questions/3140478/fatal-module-not-found-error-using-modprobe)

