---
title: nouveau安装
date: 2019-01-25 09:58:32
tags:
- nouveau
categories:
- [GPU,nouveau]
---

本篇博客介绍了nouveau安装以及卸载nouveau遇到的*Error: Module nouveau is in use* 问题。

<!-- more -->

# 安装nouveau

[nouveau官网给出了正确的安装姿势](https://nouveau.freedesktop.org/wiki/InstallNouveau/)，分成两种，一种是直接安装各种Linux的操作系统，因为Kernel里面已经自带nouveau了。  而我们要做的是第二种，从源码编译Nouveau再安装。

## 卸载NVIDIA CUDA-8.0

>Nouveau is incompatible with NVIDIA's proprietary driver. If you want to use Nouveau, you first need to remove the proprietary driver from your system.

先删除cuda-8.0驱动

```shell
sudo service lightdm stop
#然后 ctrl+Alt+F1，输入账号，密码登入
cd /usr/local/cuda-8.0/bin
sudo ./uninstall_cua_8.0.pl
```

## 遇到得问题
```
sudo modprobe nvidia
```

>modprobe: ERROR: ../libkmod/libkmod-module.c:832 kmod_module_insert_module() could not find module by name='off'
>modprobe: ERROR: could not insert 'off': Unknown symbol in module, or unknown parameter (see dmesg)

nvidia可能将nouveau禁止启动了，将其开启。

```shell
cat /etc/modprobe.d/nvidia-graphics-drivers.conf
```
>blacklist nouveau
>blacklist lbm-nouveau
>alias nouveau off
>alias lbm-nouveau off

查看 */etc/modprobe.d/blacklist.conf* ，看是否禁用了nouveau

>blacklist nouveau
>options nouveau modeset=0

删除所有 nvidia* packages
```
sudo apt-get remove --purge nvidia-*
```

然后安装nouveau驱动
```
sudo modprobe nouveau
```

查看显卡驱动：

```
sudo lspci -s 01:00.0 -vvv
```
> Kernel driver in use: nvidia
> Kernel modules: ..., nouveau

这里nvidia驱动已经卸载了，为何nouveau还没有生效。  
不要慌，遇到不行的问题，重启试试！  
**重启**
```
reboot
```

启动后
> Kernel driver in use: nouveau

## 编译安装nouveau

nouvea分成了4部分

+ Nouveau DRM : The privileged part of Nouveau, running in the kernel;
+ Libdrm : A library that wraps the DRM & Nouveau IOCTLs, it simplifies memory management and command submission;
+ xf86-video-nouveau : The device-dependent part of X (DDX) for NVIDIA cards whose role is to provide 2D acceleration (EXA) and some simple video acceleration (XV);
+ Mesa : It provides 3D, GPGPU and video acceleration to open source drivers. For more information, please read MesaDrivers.

我们关注的是 Nouveau DRM 和Libdrm。

DRM是在 Kernel-2.6.33 引入的，但是Linus的Linux Kernel总是会比nouveau自己的git更新落后些。


nouveau的 out-of-tree 版本 ，选取 **linux-4.4** 分支。

```shell
git clone https://github.com/skeggsb/nouveau.git
git checkout linux-4.4
```
此版本主要由开发人员或测试人员使用，他们经常使用一些自定义补丁重新编译内核。   
out-of-the-tree 构建只会重新编译nouveau驱动程序，比重新编译完整内核快几个数量级。  
但是，它必须针对**兼容的内核**，因为它依赖于许多内部内核API。  

编译指令为：  
```
cd nouveau/drm
make
```
然后 `insmod` *drm/nouveau/nouveau.ko* 或者 `modprobe nouveau` 。
再查看是否安装成功： `lsmod | grep nouveau`

# rmmod: ERROR: Module nouveau is in use


<https://nouveau.freedesktop.org/wiki/KernelModeSetting/> 中提到了

> Here is an example script to unload Nouveau KMS:
	```
	#!/bin/bash

	echo 0 > /sys/class/vtconsole/vtcon1/bind
	rmmod nouveau
	/etc/init.d/consolefont restart
	rmmod ttm
	rmmod drm_kms_helper
	rmmod drm
	```

上述方法不好使！
每次输入上述命令都会死机，之后只能 `Control+Alt+Delete` 组合键来重启。

## 禁用nouveau

我的办法是启动时候禁用nouveau，然后手动启动nouveau：

打开 */etc/modprobe.d/blacklist.conf* ，并添加禁用nouveau的命令行。
```
blacklist nouveau
```

如果再添加 `options nouveau modeset=0` 命令，就会禁用KMS，我这里不添加此命令，为什么呢？  

KMS自动提供了 nouveaufb ，这是对虚拟console整合到DRM驱动中的framebuffer 驱动，它提供了高分辨率的文本console。

当只有一个fb驱动程序时，它会自动运行。 当有多个时，默认使用第一个fb驱动程序（fb0）。 在CONFIG_VT_HW_CONSOLE_BINDING = n 的移交情况下，第一个fb驱动程序未完全卸载，nouveaufb将变为fb1。 这将导致非工作控制台显示，因为控制台绑定到fb0。

启动KMS的方法： 首先 Kernel 配置 *CONFIG_FRAMEBUFFER_CONSOLE* ，会得到 fbcon.ko 模块。 而且确保此模块要早于 nouveau.ko 载入。 
启动nouveau会默认启动 KMS，但是可以用 modeset=0 禁用它。

确定KMS是否在运行， 查看 */proc/fb* ，如果显示 `nouveaufb` 则证明KMS在运行。
```
root@MAX:/home/max# cat /proc/fb 
0 nouveaufb
```
如果禁用了 KMS，则显示默认的 *VESA VGA*
```
root@MAX:# cat /proc/fb 
0 VESA VGA
root@MAX:# modprobe nouveau
root@MAX:# cat /proc/fb 
0 nouveaufb
```
## Deactivating KMS and unloading Nouveau

framebuffer console保留了 nouveaufb 因此不能直接卸载nouveau，首先要先解除nouveau绑定关系。  
只有解除之后，较早的 fb 驱动或者 VGA console 驱动接管它。  
内核先要配置 *CONFIG_VT_HW_CONSOLE_BINDING* 。  

```
root@MAX:# lsmod | grep nouveau
nouveau              1503232  1
root@MAX:# rmmod nouveau
rmmod: ERROR: Module nouveau is in use
```


```shell
#!/bin/bash

echo 0 > /sys/class/vtconsole/vtcon1/bind
rmmod nouveau
/etc/init.d/consolefont restart
rmmod ttm
rmmod drm_kms_helper
rmmod drm
```

`echo 0 > /sys/class/vtconsole/vtcon1/bind` 将 nouveaufb 从 fb console 驱动(fbcon)中解绑，通常情况下是 `vtcon1` 但也可能是其他情况 `vtcon*` ， 查看 `/sys/class/vtconsole/vtcon*/name` 哪一个是 `frame buffer device` 。  

运行结果：
```
root@MAX:# lsmod | grep nouveau
nouveau              1503232  1
root@MAX:# echo 0 > /sys/class/vtconsole/vtcon1/bind 
root@MAX:# lsmod | grep nouveau
nouveau              1503232  0

```

更新系统文件：

```
update-initramfs -u
```
重启。
再次执行
```
lsmod | grep nouveau 
```
无结果，安装我们自己的驱动
```
insmod nouveau.ko
```
报错！
> insmod: ERROR: could not insert module: Unknown symbol in module 

据说是没有依赖项。
那么好了，由于内核中保留着原来的 *nouveau.ko* ，那么先执行
```
modprobe nouveau
```
查看内核模块

```
root@MAX:/home/max# lsmod | grep nouveau
nouveau              1503232  0
mxm_wmi                16384  1 nouveau
ttm                    98304  1 nouveau
drm_kms_helper        155648  1 nouveau
drm                   364544  4 ttm,drm_kms_helper,nouveau
i2c_algo_bit           16384  1 nouveau
wmi                    20480  3 mxm_wmi,nouveau,asus_wmi
video                  40960  2 nouveau,asus_wmi
```

nouveau 没有被其他模块占用。
而且nouveau 依赖的模块有多个。

```
rmmod nouveau
```
再次安装我们自己的nouveua
```
insmod nouveau.ko
```

这种方法好使了。

从安装到卸载 nouveau 驱动的dmesg：
```
>>>[  369.598143] My Nouveau installed!
[  369.598214] nouveau 0000:01:00.0: NVIDIA GK110B (0f10c0a1)
[  369.719906] nouveau 0000:01:00.0: bios: version 80.80.4e.00.01
[  369.721507] nouveau 0000:01:00.0: fb: 6144 MiB GDDR5
[  369.777643] [TTM] Zone  kernel: Available graphics memory: 8189062 kiB
[  369.777645] [TTM] Zone   dma32: Available graphics memory: 2097152 kiB
[  369.777655] [TTM] Initializing pool allocator
[  369.777661] [TTM] Initializing DMA pool allocator
[  369.777670] nouveau 0000:01:00.0: DRM: VRAM: 6144 MiB
[  369.777671] nouveau 0000:01:00.0: DRM: GART: 1048576 MiB
[  369.777675] nouveau 0000:01:00.0: DRM: TMDS table version 2.0
[  369.777677] nouveau 0000:01:00.0: DRM: DCB version 4.0
[  369.777679] nouveau 0000:01:00.0: DRM: DCB outp 00: 01000f02 00020030
[  369.777681] nouveau 0000:01:00.0: DRM: DCB outp 01: 02000f00 00000000
[  369.777683] nouveau 0000:01:00.0: DRM: DCB outp 02: 08011f82 00020030
[  369.777685] nouveau 0000:01:00.0: DRM: DCB outp 04: 02022f62 00020010
[  369.777687] nouveau 0000:01:00.0: DRM: DCB outp 05: 04833fb6 0f420010
[  369.777688] nouveau 0000:01:00.0: DRM: DCB outp 06: 04033f72 00020010
[  369.777690] nouveau 0000:01:00.0: DRM: DCB conn 00: 00001030
[  369.777692] nouveau 0000:01:00.0: DRM: DCB conn 01: 00020131
[  369.777694] nouveau 0000:01:00.0: DRM: DCB conn 02: 00002261
[  369.777695] nouveau 0000:01:00.0: DRM: DCB conn 03: 00010346
[  369.790383] [drm] Supports vblank timestamp caching Rev 2 (21.10.2013).
[  369.790385] [drm] Driver supports precise vblank timestamp query.
[  369.930687] nouveau 0000:01:00.0: DRM: MM: using COPY for buffer copies
[  370.302176] nouveau 0000:01:00.0: DRM: allocated 1920x1080 fb: 0x60000, bo ffff8800b5165000
[  370.302315] fbcon: nouveaufb (fb0) is primary device
[  370.790138] Console: switching to colour frame buffer device 240x67
[  370.790742] nouveau 0000:01:00.0: fb0: nouveaufb frame buffer device
[  370.800914] [drm] Initialized nouveau 1.3.1 20120801 for 0000:01:00.0 on minor 0
[  495.668980] Console: switching to colour dummy device 80x25
>>>[  512.432376] My Nouveau uninstalled!
[  512.535441] [TTM] Finalizing pool allocator
[  512.535448] [TTM] Finalizing DMA pool allocator
[  512.535567] [TTM] Zone  kernel: Used memory at exit: 0 kiB
[  512.535573] [TTM] Zone   dma32: Used memory at exit: 0 kiB
[  512.537596] [drm] Module unloaded

```

可以看到，我在模块安装和卸载时打印的语句。

