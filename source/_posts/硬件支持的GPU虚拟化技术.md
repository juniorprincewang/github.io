---
title: 硬件支持的GPU虚拟化技术
date: 2018-06-18 16:55:38
tags:
- CUDA
categories:
- GPU
- GPU虚拟化
---
本文介绍的内容包括Intel、NVIDIA、AMD提供的对GPU虚拟化的硬件支持。
<!-- more -->

强烈建议先看一遍 18年阿里云郑晓，龙欣推出的 《浅谈GPU虚拟化技术》 系列博客，很系统的总结了Intel、Nvidia、AMD三大厂商的硬件虚拟化技术。  

[阿里云郑晓：浅谈GPU虚拟化技术（第一章）-GPU虚拟化发展史](https://developer.aliyun.com/article/578724)  
[阿里云郑晓：浅谈GPU虚拟化技术（第二章）-GPU直通模式](https://developer.aliyun.com/article/590910)  
[第三章 浅谈GPU虚拟化技术（三）GPU SRIOV及vGPU调度](https://developer.aliyun.com/article/590916)  
[浅谈GPU虚拟化技术（四）- GPU分片虚拟化](https://developer.aliyun.com/article/599189)  
[浅谈GPU虚拟化技术（五）：GPU图形渲染虚拟化的业界难题-VDI的用户体验](https://developer.aliyun.com/article/591405)  

# 商用GPU虚拟化方案  

在虚拟化环境中，GPU使用目前可以分为以下几类：

- GPU pass-through 直通模式，即GPU透传。  
- GPU SR-IOV，目前主要是AMD在采用此种方案  
- GPU分片虚拟化 mdev，包括Intel GVT-g和NVIDIA GRID vGPU
- GPU全虚拟化（VMWare的 vSGA）
- GPU半虚拟化 virtio-gpu

下面重点介绍硬件支持的 GPU虚拟化方案： passthrough、mediated passthrough、SR-IOV。   


# GPU Passthrough  

该模式是最早也是最流行的GPU虚拟化方案。  
直通模式下性能损失最小，硬件驱动无需修改。  
缺点包括不支持热迁移（Live Migration）；不支持GPU资源分割；绕过了hypervisor，因此不能被hypervisor监控

其实现依赖 IOMMU。  

    PCI 直通的技术实现：所有直通设备的PCI 配置空间都是模拟的。而且基本上都只模拟256 Bytes的传统PCI设备，很少有模拟PCIE设备整个4KB大小的。  
    而对PCI设备的PCI bars则绝大部分被mmap到qemu进程空间，并在虚拟机首次访问设备PCI bars的时候建立EPT 页表映射，从而保证了设备访问的高性能。


# IOMMU  

IOMMU 可以看作 Device 的 MMU，提供DMA地址转换、对设备读取和写入的权限检查。这样驱动程序可以直接访问外设，而不需要通过VMM。  

IOMMU需要CPU支持（Intel VT-d/ AMD Vi），并在主板中启用。 

[Linux kernel document: vfio.txt](https://www.kernel.org/doc/Documentation/vfio.txt)  

# GPU分片虚拟化 Mediated passthrough (mdev)

mediated passthrough 把会影响性能的访问（如DMA）直接passthrough给虚拟机，把性能无关，功能性的MMIO访问做拦截并在mdev模块内做模拟。
Mediated是指对MMIO 访问的拦截和emulation，而对DMA transfer的提交通过VFIO的passthrough 直接映射到 VM内部。  
该点子最早来自于 ATC 的论文：  A Full GPU Virtualization Solution with Mediated Pass-Through。  

但是 VFIO的mdev框架是由Nvidia为了GRID vGPU 产品线而引入。 mdev （Mediated devices）的概念由Nvidia率先提出的，并合并到了Linux 内核4.10。

这里不展开对 vfio-mdev 的总结，详见[vfio-mdev逻辑空间分析](https://zhuanlan.zhihu.com/p/28111201)  和 [Documentation / vfio-mediated-device.txt](https://www.mjmwired.net/kernel/Documentation/vfio-mediated-device.txt)。  

GPU分片模式不依赖于 IOMMU，vGPU的cmd提交（内含GPA地址）并不能直接运行于GPU硬件之上，至少需要有一个GPA到HPA的翻译过程。该过程可以通过host端的cmd扫描来修复（KVMGT），NVIDIA GRID vGPU每一个context有其内部page table，会通过修改page table来实现。  
GPU可以被hypervisor监控。  

GPU分片虚拟化的方案被 NVIDIA 与 Intel两家GPU厂家所采用。NVIDIA GRID vGPU系列与Intel的GVT-g（XenGT or KVMGT）。  
若要搭建起来，需要 内核 4.10之后的版本、qemu v2.0 以及GPU mdev驱动（也就是对GPU MMIO访问的模拟）。  


## Intel's Graphics Virtualization Technology (GVT)

Intel提供 `GVT-g` 方案：针对不同的hypervisor，在KVM上，叫 `KVMGT`；而在Xen上，称为 `XenGT`。  

Intel 开源大部分集成显卡GPU的运行机理和软硬件规范。 <https://01.org/linuxgraphics/documentation/hardware-specification-prms>
GVT-g 方案是开源的，可以为任何带集显的Intel CPU（HSW，BDW，SKL系列CPU）提供vGPU，并且也被应用到IoT领域（ARCN hypervisor）。  

GVT-g 的 kernel和mdev驱动源码：<https://github.com/intel/gvt-linux>  
GVT-g QEMU源码 <https://github.com/intel/IGVTg-qemu>  

## NVIDIA GRID

GRID vGPU 是NVIDIA 支持虚拟化的GPU技术，NVIDIA vGPU在特定的GPU卡上支持。  
NVIDIA 最早引入GRID技术的GPU 是 NVIDIA GRID K1。 

[白皮书 NVIDIA GRID: GRAPHICS ACCELERATED VDI WITH THE VISUAL PERFORMANCE OF A WORKSTATION,White Paper | May 2014 ](https://www.nvidia.com/content/grid/resources/White_paper_graphics_accelerated_VDI_v1.pdf)
介绍 GRID 技术是GPU的 MMU 将Host的虚拟地址转换的device的物理地址是隔离的，维护了 256个独立的input buffer，将每个VM提交的命令流隔离到独自的context中。  

GRID技术使用到了hypervisor调度，来自VM的命令流分配到独自的vGPU driver，每个vGPU driver通过隔离的input channel将命令和控制发送到物理GPU上；渲染完成后再传输回remote host。  



现在需要用到的GPU卡，比如Tesla GPUs 产品。  
并且NVIDIA vGPU 需要GPU mdev驱动支持，但是NVIDIA没有开源。使用者需要license激活使用它的完整功能，否则只能使用阉割版。  

GRID vGPU分片虚拟化的方案相对GPU passthrough来说部署比较困难。  
[Virtual GPU Software User Guide：vGPU使用安装流程](https://docs.nvidia.com/grid/5.0/grid-vgpu-user-guide/index.html)，但是里面提到了一点：单个VM不支持多个vGPUs。  
> Note: Multiple vGPUs in a VM are not supported.  


NVIDIA vGPU 目前基于 [Turing 架构](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)、 [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)、 [Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) 、 [Maxwell架构](https://developer.nvidia.com/maxwell-compute-architecture)  。  

vGPU应用在以下四个场景：  
NVIDIA Virtual Compute Server (vCS)  
NVIDIA Quadro Virtual Data Center Workstation (Quadro vDWS)  
NVIDIA GRID Virtual PC (GRID vPC)  
NVIDIA GRID Virtual Applications (GRID vApps)  

通常支持vGPU的NVIDIA产品包括  NVIDIA A100、V100S、RTX 8000、RTX 6000、T4、M10、P6还包括NVIDIA V100, Quadro RTX 8000 (active), Quadro RTX 6000 (active), P40 等。


vGPU 性能会比 bare metal 低 10% 以内，通常少于 5%。  
[NVIDIA VIRTUAL GPU TECHNOLOGY](https://www.nvidia.com/en-us/data-center/virtual-gpu-technology/)  

[如何在产品中使用NVIDIA vGPU](https://www.awcloud.com/3714.html)    
[NVIDIA GPUs FOR VIRTUALIZATION](https://www.nvidia.com/en-us/data-center/graphics-cards-for-virtualization/)  
[nvidia virtualization gpu linecard](https://images.nvidia.com/content/pdf/grid/data-sheet/nvidia-virtualization-gpu-linecard.pdf)  

# GPU SR-IOV  

标准的PCIe的标准 Single Root I/O Virtualization（SR-IOV）。  
SR-IOV 实现依赖 IOMMU，IOMMU的作用是完成GFN到PFN的地址转换。  

## AMD Multiuser GPU (MxGPU)

产品： 针对图形渲染的AMD Firepro S7150 、 针对机器学习的 MI25。  
目前支持 VMware ESXi, KVM and Xen hypervisors。  

[面向虚拟化的 Radeon Pro](https://www.amd.com/zh-hans/graphics/workstation-virtual-graphics)  
[Overview of Single Root I/O Virtualization (SR-IOV)](https://docs.microsoft.com/en-us/windows-hardware/drivers/network/overview-of-single-root-i-o-virtualization--sr-iov-)  

构成分两部分：  
+ A PCIe Physical Function (**PF**)
宿主机的GPU驱动安装到PF上，它管理了所有VF设备的生命周期和调度。  
+ One or more PCIe Virtual Functions (**VFs**)  
QEMU在启动时候通过VFIO模块将VF作为PCI直通设备交给虚拟机。  

SRIOV 通过 IOMMU 对DMA请求进行保护，实现GPA到 HPA 的转换。  

SRIOV 对GPU的虚拟在Host端的GPU硬件，固件和GIM驱动。  



