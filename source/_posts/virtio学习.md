---
title: virtio学习
date: 2018-03-01 10:51:38
tags:
- virtio
- libvirt
categories:
- 虚拟化
---

涉及到virtio的使用和原理学习。
<!-- more -->
# virtio

`virtio`是半虚拟化的解决方案，对半虚拟化Hypervisor的一组通用I/O设备的抽象。它提供了一套上层应用与各 Hypervisor 虚拟化设备（KVM，Xen，VMware等）之间的通信框架和编程接口，减少跨平台所带来的兼容性问题，大大提高驱动程序开发效率。

在完全虚拟化的解决方案中，guest VM 要使用底层 host 资源，需要 Hypervisor 来截获所有的请求指令，然后模拟出这些指令的行为，这样势必会带来很多性能上的开销。半虚拟化通过底层硬件辅助的方式，将部分没必要虚拟化的指令通过硬件来完成，Hypervisor 只负责完成部分指令的虚拟化，要做到这点，需要 guest 来配合，guest 完成不同设备的前端驱动程序，Hypervisor 配合 guest 完成相应的后端驱动程序，这样两者之间通过某种交互机制就可以实现高效的虚拟化过程。

由于不同 guest 前端设备其工作逻辑大同小异（如块设备、网络设备、PCI设备、balloon驱动等），单独为每个设备定义一套接口实属没有必要，而且还要考虑扩平台的兼容性问题，另外，不同后端 Hypervisor 的实现方式也大同小异（如KVM、Xen等），这个时候，就需要一套通用框架和标准接口（协议）来完成两者之间的交互过程，virtio 就是这样一套标准，它极大地解决了这些不通用的问题。


# virtio原理


# virtio的使用

由于传统的QEMU/KVM方式是使用QEMU纯软件模拟I/O设备（网卡、磁盘、显卡），导致效率并不高。在KVM中，可以在客户机使用半虚拟化（paravirtualized drivers）来提高客户机的性能


```
sudo qemu-system-x86_64 -enable-kvm -boot c -drive file=ubuntu16.04.qcow2,if=virtio -m 1024 -netdev type=tap,script=/etc/qemu-ifup,id=net0 -device virtio-net-pci,netdev=net0
```
## qemu-system-x86-64命令行解释



[1] (Virtio)[http://www.linux-kvm.org/page/Virtio]
[2] (QEMU how to setup Tun/Tap + bridge networking)[https://tthtlc.wordpress.com/2015/10/21/qemu-how-to-setup-tuntap-bridge-networking/] 
