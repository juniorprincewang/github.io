---
title: kernel源码中编译nouveau module
date: 2019-05-20 10:54:03
tags:
- nouveau
- kernel
- module
categories:
- [GPU,nouveau]
---
nouveau可以从单独的module中编译，也可以从kernel source code中单独编译出nouveau.ko。
<!-- more -->

从kernel中编译定制的module分成以下几步骤：

准备内核源码：

```
git clone git://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
```
可以切换到自己需要的分支或者tag。
```
make oldconfig 
make prepare
make scripts
```

`make oldconfig`确保源码中有当前运行的kernel配置。

为了防止出现 
> insmod: ERROR: could not insert module /home/pc/linux/drivers/gpu/drm/nouveau/nouveau.ko: Invalid module format

```
cp -v /usr/src/linux-headers-$(uname -r)/Module.symvers .
```

编译nouveau。

```
make M=drivers/gpu/drm/nouveau -j8
```

安装
```
insmod /home/pc/linux/drivers/gpu/drm/nouveau/nouveau.ko
```
成功！


or 

为customed nouveau内核腾地儿。  
```
mv -v /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko.backup
```

并使用 */lib/modules/$(uname -r)/* 中的内核配置编译nouveau内核模块生成nouveau.ko。  
```
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```


```
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules_install
```
此命令将kernel模块安装在 */lib/modules/$(uname -r)/extra/* 中，以防如果不重新命名nouveau.ko，此命令是不会覆盖在 */lib/modules/$(uname -r)/build/* 内核模块的。

放在一起：  
```
mv -v /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko.backup
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules_install
```

安装：
```
depmod
modprobe -v nouveau
```


[How (recipe) to build only one kernel module?](https://askubuntu.com/questions/515407/how-recipe-to-build-only-one-kernel-module)  
[How to compile a module from downloaded Linux source?](https://stackoverflow.com/questions/19995464/how-to-compile-a-module-from-downloaded-linux-source)