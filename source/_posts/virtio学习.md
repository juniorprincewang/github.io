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

由于传统的QEMU/KVM方式是使用QEMU纯软件模拟I/O设备（网卡、磁盘、显卡），导致效率并不高。在KVM中，可以在客户机使用半虚拟化（paravirtualized drivers）来提高客户机的性能。

## QEMU模拟I/O设备得基本原理

当客户机的设备驱动程序（Device Driver）发起I/O请求时，KVM模块中的I/O操作捕获代码会拦截这次I/O请求，然后经过处理后将本次I/O请求的信息存放到I/O共享页（sharing page），并通知用户控件的QEMU程序。QEMU模拟程序获得I/O操作的具体信息后，交给硬件模拟代码（Emulation Code）来模拟出本次的I/O操作，完成后把结果放回I/O共享页中，并通知KVM模块中的I/O操作捕获代码。最后由KVM模块中的捕获代码读取I/O共享页中的操作结果，把结果返回给客户机中。当然，这个操作过程中客户机作为一个QEMU进程在等待I/O时也可能被阻塞。

另外，当客户机通过DMA访问大块I/O时，QEMU模拟程序不会把操作结果放到I/O共享页中，而是通过内存映射的方法将结果直接写到客户机的内存去，然后通过KVM模块高速客户机DMA操作已经完成。

QEMU模拟I/O设备不需要修改客户端操作系统，可以模拟各种各样的硬件设备，但是每次I/O操作的路径比较长，有太多的VMEntry和VMExit发生，需要多次上下文切换（context switch），多次的数据复制。性能方面很差。

## virtio的基本原理

virto由大神Rusty Russell编写（现已转向区块链了。。。），是在Hypervisor之上的抽象API接口，客户机需要知道自己运行在虚拟化环境中，进而根据virtio标准和Hypervisor协作，提高客户机的性能（特别是I/O性能）。

![virtio基本架构](../virtio学习/architecture.gif)

前端驱动（Front-end）是在客户机中存在的驱动程序模块，而后端处理器程序是在QEMU中实现的。

virtio是版虚拟化驱动的方式，其I/O性能几乎可以达到和native差不多的I/O性能。但是virtio必须要客户机安装特定的virtio驱动使其知道是运行在虚拟化环境中，并按照virtio的规定格式进行数据传输。

Linux2.6.24及其以上版本的内核都支持virtio。由于virtio的后端处理程序是在位于用户空间的QEMU中实现的，所以宿主机中只需要比较新的内核即可，不需要特别地编译与virtio相关地驱动。但是客户机需要有特定地virtio驱动程序支持，以便客户机处理I/O操作请求时调用virtio驱动。


### 使用virtio_net

使用virtio_net半虚拟化驱动，可以提高网络吞吐量（throughput）和降低网络延迟（latency），达到原生网卡的性能。

使用virtio_net需要宿主机中的QEMU工具和客户机的virtio_net驱动支持。

#### 检查QEMU是否支持virtio类型的网卡
```
# qemu-system-x86_64 -net nic,model=?
qemu: Supported NIC models: ne2k_pci,i82551,i82557b,i82559er,rtl8139,e1000,pcnet,virtio
```
从输出的支持网卡类型克制，当前qemu-kvm支持virtio网卡类型。


#### 启动客户机，指定分配virtio网卡设备


```
sudo qemu-system-x86_64 -enable-kvm -boot c -drive file=ubuntu16.04.qcow2,if=virtio -m 1024 -netdev type=tap,script=/etc/qemu-ifup,id=net0 -device virtio-net-pci,netdev=net0
```

qemu-system-x86-64命令行解释

- `–enable-kvm` 创建x86的虚拟机需要用到qemu-system-x86_64这个命令，并需要加上`–enable-kvm`来支持kvm加速，不适用KVM加速虚拟机会非常缓慢。
- `boot` 磁盘相关参数，设置客户机启动时的各种选项。`c`表示第一个硬盘。
- `drive` 配置驱动。使用`file`文件作为镜像文件加载到客户机的驱动器中。`if`指定驱动器使用的接口类型，包括了virtio在内。
- `m` 设置客户机内存大小，单位默认为`MB`。也可以用`G`为单位。
- `netdev` 新型的网络配置方法，在宿主机中建立一个网络后端驱动。`TAP`是虚拟网络设备，它仿真了一个数据链路层设备。`TAP`用于创建一个网络桥，使用网桥连接和NAT模式网络的客户机都会用到`TAP`参数。`script`用于设置宿主机在启动客户机时自动执行的网络配置脚本，如果不指定，默认为`/etc/qemu-ifup`，如果不需要执行脚本，则设置`script=no`。`id`用于在宿主机中指定的TAP虚拟设备的`ID`。
- `device` 为虚拟机添加设备。这里添加了`virtio-net-pci`设备，使用了`net0`的TAP虚拟网卡。


```
-device driver[,prop[=value][,...]]
                add device (based on driver)
                prop=value,... sets driver properties
                use '-device help' to print all possible drivers
                use '-device driver,help' to print all possible properties

name "virtio-net-pci", bus PCI, alias "virtio-net"

-netdev tap,id=str[,fd=h][,fds=x:y:...:z][,ifname=name][,script=file][,downscript=dfile]
         [,helper=helper][,sndbuf=nbytes][,vnet_hdr=on|off][,vhost=on|off]
         [,vhostfd=h][,vhostfds=x:y:...:z][,vhostforce=on|off][,queues=n]
                configure a host TAP network backend with ID 'str'
                use network scripts 'file' (default=/etc/qemu-ifup)
                to configure it and 'dfile' (default=/etc/qemu-ifdown)
                to deconfigure it
                use '[down]script=no' to disable script execution
                use network helper 'helper' (default=/usr/lib/qemu/qemu-bridge-helper) to
                configure it
                use 'fd=h' to connect to an already opened TAP interface
                use 'fds=x:y:...:z' to connect to already opened multiqueue capable TAP interfaces
                use 'sndbuf=nbytes' to limit the size of the send buffer (the
                default is disabled 'sndbuf=0' to enable flow control set 'sndbuf=1048576')
                use vnet_hdr=off to avoid enabling the IFF_VNET_HDR tap flag
                use vnet_hdr=on to make the lack of IFF_VNET_HDR support an error condition
                use vhost=on to enable experimental in kernel accelerator
                    (only has effect for virtio guests which use MSIX)
                use vhostforce=on to force vhost on for non-MSIX virtio guests
                use 'vhostfd=h' to connect to an already opened vhost net device
                use 'vhostfds=x:y:...:z to connect to multiple already opened vhost net devices
                use 'queues=n' to specify the number of queues to be created for multiqueue TAP
```


## qemu创建虚拟机

### qemu-img创建虚拟机镜像
虚拟机镜像用来模拟虚拟机的硬盘，在启动虚拟机之前需要创建镜像文件。qemu-img是QEMU的磁盘管理工具，可以用qemu-img创建虚拟机镜像。
```
qemu-img create -f qcow2 ubuntu.qcow2 20G
```

`-f`选项用于指定镜像的格式，`qcow2`格式是QEMU最常用的镜像格式，采用来写时复制技术来优化性能。`ubuntu.qcow2`是镜像文件的名字，`20G`是镜像文件大小。镜像文件创建完成后，可使用`qemu-system-x86`来启动`x86`架构的虚拟机

### 检查KVM是否可用

QEMU使用KVM来提升虚拟机性能，如果不启用KVM会导致性能损失。要使用KVM，首先要检查硬件是否有虚拟化支持：
```
grep -E 'vmx|svm' /proc/cpuinfo
```
如果有输出则表示硬件有虚拟化支持。其次要检查kvm模块是否已经加载：
```
lsmod | grep kvm
```
	kvm_intel             1429990 
	kvm                   4443141 kvm_intel
如果kvm_intel/kvm_amd、kvm模块被显示出来，则kvm模块已经加载。最好要确保qemu在编译的时候使能了KVM，即在执行configure脚本的时候加入了–enable-kvm选项。

### 安装操作系统。
准备好虚拟机操作系统ISO镜像。执行下面的命令启动带有cdrom的虚拟机：
```
qemu-system-x86_64 -m 2048 -enable-kvm ubuntu.qcow2 -cdrom ubuntu.iso
```
- `-m`指定虚拟机内存大小，默认单位是MB， 
- `-enable-kvm`使用KVM进行加速，
- `-cdrom`添加`ubuntu`的安装镜像。

可在弹出的窗口中操作虚拟机，安装操作系统，安装完成后重起虚拟机便会从硬盘(ubuntu.qcow2 )启动。

### 启动虚拟机
启动虚拟机只需要执行:
```
qemu-system-x86_64 -m 2048 -enable-kvm ubuntu.qcow2 
```
即可。



# 参考
[1] [Virtio](http://www.linux-kvm.org/page/Virtio)
[2] [QEMU how to setup Tun/Tap + bridge networking](https://tthtlc.wordpress.com/2015/10/21/qemu-how-to-setup-tuntap-bridge-networking/])
[3] [QEMU 1: 使用QEMU创建虚拟机](https://my.oschina.net/kelvinxupt/blog/265108)
[4] Virtio: towards a de factor standard for virtual I/O devices
[5] [访问qemu虚拟机的五种姿势](http://blog.csdn.net/richardysteven/article/details/54807927)

