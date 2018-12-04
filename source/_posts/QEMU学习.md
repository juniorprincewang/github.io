---
title: QEMU 学习
date: 2018-11-15 20:22:10
tags:
- QEMU
- emulate
- virtio
- KVM
categories:
- QEMU
---
本篇博客记录了在学习使用`QEMU`时的资料。包括 `QEMU` 的整体架构和命令行。
<!-- more -->


`QEMU`是一个主机上的VMM（virtual machine monitor），通过动态二进制转换来模拟CPU，并提供一系列的硬件模型。
`KVM`（Kernel-Based Virtual Machine）是基于内核的虚拟机，实现对CPU和内存的虚拟化。KVM需要处理器硬件本身支持虚拟化扩展，如intel VT 和AMD AMD-V技术。同时它是Linux内核的一个可加载模块，KVM从Linux 2.6.20以后已被作为内核组件。
从存在形式来看，它包括两个内核模块：kvm.ko用于实现核心虚拟化功能  和  kvm_intel.ko（或 kvm_amd.ko）处理器强相关的模块。 本质上，KVM是管理虚拟硬件设备的驱动，该驱动使用字符设备 `/dev/kvm`（由KVM本身创建）作为管理接口，主要负责 `vCPU` 的创建，虚拟内存的分配，`vCPU`寄存器的读写以及 `vCPU`的运行。

有了KVM以后，guest os的CPU指令不用再经过QEMU来转译便可直接运行，大大提高了运行速度。但KVM的kvm.ko本身只提供了CPU和内存的虚拟化，所以它必须结合QEMU才能构成一个完整的虚拟化技术。


`QEMU-KVM` ： KVM运行在内核空间，QEMU运行在用户空间，实际模拟创建、管理各种虚拟硬件，QEMU将KVM整合了进来，通过 `ioctl` 调用 `/dev/kvm` ，从而将CPU指令的部分交给内核模块来做，KVM实现了CPU和内存的虚拟化，但KVM不能虚拟其他硬件设备，因此QEMU还有模拟IO设备（磁盘，网卡，显卡等）的作用，KVM加上QEMU后就是完整意义上的服务器虚拟化。 由于QEMU纯模拟IO设备的效率不高，一般采用半虚拟化的`VIRTIO`来虚拟IO设备。
kvm加速的伪代码：
```
open("/dev/kvm")
ioctl(KVM_CREATE_VM)
ioctl(KVM_CREATE_VCPU)
for (;;) {
	ioctl(KVM_RUN)
	switch (exit_reason) {
		case KVM_EXIT_IO: /* ... */
		case KVM_EXIT_HLT: /* ... */
	}
}
```
为了使用KVM执行虚拟机代码，QEMU进程打开/dev/kvm并发出KVM_RUN ioctl。 KVM内核模块使用现代Intel和AMD CPU上的硬件虚拟化扩展来直接执行虚拟机代码。 当guest虚拟机访问硬件设备寄存器，或是暂停虚拟机CPU或是执行其他特殊操作时，KVM将退出并将控制权转给QEMU。 此时，QEMU可以模拟操作的预期输出，或者只是客户CPU在暂停的情况下等待下一个客户机中断。

具体分工为：KVM负责对CPU和内存模拟，QEMU负责对IO设备模拟并对各种虚拟设备的创建和调度进行管理。

![QEMU-KVM图](../QEMU学习/QEMU-KVM.webp)

# 开发

## 下载指定版本

```
git clone git://git.qemu.org/qemu.git
```
切换到指定的版本QEMU-2.12
```
git checkout -b stable-2.12
```

## 编译

configure脚本检测所有依赖的库
```
./configure
```
查看能够启用的特征选项：
```
./configure --help
```
如果只编译支持x86_64客户机，那么仅需要给`configure`附带参数 ` --targetlist=x86_64-softmmu` 。

## 开发知识

在目录 `./docs` 中保存了 规格和文档说明。
文档 `./CODING_STYLE` 和 `./HACKING` 分别介绍了QEMU编程遵行的代码风格和详细的编程指导。
通过脚本 `./scripts/checkpatch.pl` 检查补丁文件。

# QEMU内部架构

## QEMU process model

## Main loop

## Device emulation

### Hardware emulation model

### Guest/host device split

### Guest device emulation

+ Devices memory or I/O regions

需要实现设备的读/写处理函数。

+ 中断响应
+ 通过 `info qtree` 来探视设备
+ 通过 `info mtree` 来探视设备内存 


# QEMU monitor

进入QEMU命令行控制界面，可以通过在QEMU 启动的时候指定 `-monitor` 参数；也可以在 QEMU 窗口激活的时候按住 `Ctrl+Alt+2` 进入，切换回工作界面需要按 `Ctrl+Alt+1` 。



# 参考
1. [What's a good source to learn about QEMU? ](https://stackoverflow.com/questions/155109/whats-a-good-source-to-learn-about-qemu)
2. [官方的手册](https://wiki.qemu.org/Manual)
3. [QEMU源码架构和说明](https://vmsplice.net/~stefan/qemu-code-overview.pdf)
4. [QEMU Emulator User Documentation用户手册及命令行参数](http://manpages.ubuntu.com/manpages/trusty/en/man1/qemu.1.html)
5. [QEMU Internals: Big picture overview](http://blog.vmsplice.net/2011/03/qemu-internals-big-picture-overview.html)
6. [QEMU Internals: Overall architecture and threading model](http://blog.vmsplice.net/2011/03/qemu-internals-overall-architecture-and.html) 

[使用 monitor command 监控 QEMU 运行状态](https://www.ibm.com/developerworks/cn/linux/l-cn-qemu-monitor/index.html)
