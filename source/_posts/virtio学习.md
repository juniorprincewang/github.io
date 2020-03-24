---
title: virtio学习
date: 2018-03-01 10:51:38
tags:
- virtio
- libvirt
- QEMU
categories:
- QEMU
---

本文涉及到virtio的规范、原理和使用。
<!-- more -->
# virtio

`virtio`是半虚拟化的解决方案，对半虚拟化Hypervisor的一组通用I/O设备的抽象。它提供了一套上层应用与各 Hypervisor 虚拟化设备（KVM，Xen，VMware等）之间的通信框架和编程接口，减少跨平台所带来的兼容性问题，大大提高驱动程序开发效率。

在完全虚拟化的解决方案中，guest VM 要使用底层 host 资源，需要 Hypervisor 来截获所有的请求指令，然后模拟出这些指令的行为，这样势必会带来很多性能上的开销。半虚拟化通过底层硬件辅助的方式，将部分没必要虚拟化的指令通过硬件来完成，Hypervisor 只负责完成部分指令的虚拟化，要做到这点，需要 guest 来配合，guest 完成不同设备的前端驱动程序，Hypervisor 配合 guest 完成相应的后端驱动程序，这样两者之间通过某种交互机制就可以实现高效的虚拟化过程。

由于不同 guest 前端设备其工作逻辑大同小异（如块设备、网络设备、PCI设备、balloon驱动等），单独为每个设备定义一套接口实属没有必要，而且还要考虑扩平台的兼容性问题，另外，不同后端 Hypervisor 的实现方式也大同小异（如KVM、Xen等），这个时候，就需要一套通用框架和标准接口（协议）来完成两者之间的交互过程，virtio 就是这样一套标准，它极大地解决了这些不通用的问题。
virtio spec 由 [OASIS](https://www.oasis-open.org/standards) 维护，目前已经更新到 [virtio v1.1](https://docs.oasis-open.org/virtio/virtio/v1.1/csprd01/virtio-v1.1-csprd01.html)。

# virtio原理  

## virtio规范  

VirtIO驱动定义了一组规范，只要guest和host按照此规范进行数据操作，就可以使虚拟机IO绕过内核空间而直接再用户空间的两个进程间传输数据，以此达到提高IO性能的目的。  

VirtIO驱动的实现可以有很多种，最广泛的就是VirtIO Over PCI Bus，其它实现：VirtIO Over MMIO和VirtIO Over Channel IO，这里介绍 virtio PCI BUS。  

VirtIO设备必须具备以下几个功能组件：

    设备状态域
    特性标志位
    设备配置空间

### 设备状态域  

VirtIO设备的初始化必须按照以下步骤顺序进行才被认为被识别：

+ 重启设备。 
+ 标记ACKNOWLEDGE状态标志位，表示客户机发现了此设备。 
+ 标记DRIVER状态标志位，表示客户机知道怎么驱动这个设备。 
+ 读取设备特性标志位并给VirtIO设备设置特性标志，在此阶段，驱动可以读取特定设备相关的一些域的数据，驱动依赖检查自己是否支持这些特性，如果支持，就采纳之。 
+ 设置FEATURES_OK状态标志位，此阶段之后，驱动就不能再采纳新的特性了。 
+ 再次读取设备状态，确认FEATURES_OK标志已经被设备成功：如果没有被设置成功，表示这个设备不支持该特性，设备无法就无法使用。 
+ 执行具体设备的设置操作：包括扫描设备关联的virtqueues，总线设置，如果是VirtIO over PCI还需要写设备的VirtIO配置空间，还有virtqueues的使能以及运用。 
+ 最后标记DRIVER_OK状态标志位，自此VirtIO设备就初始化完成，可以使用了。

### 特性标志位

每个VirtIO设备都需要支持各种特性。在设备初始化阶段，驱动读取这个特性标志，然后通知VirtIO设备驱动可以接受的特性子集。  

### 设备配置空间  

设备配置空间，通常用于配置不常变动的参数，或者初始化阶段设置的参数。  
特性标志位包含表示配置空间是否存在的bit位，后面的版本会通过在特性标志位的末尾新添新的bit位来扩展配置空间。

[VirtIO驱动——规范1](https://blog.csdn.net/huang987246510/article/details/81004603)  

## virtio的架构

virto由大神Rusty Russell编写（现已转向区块链了。。。），是在Hypervisor之上的抽象API接口，客户机需要知道自己运行在虚拟化环境中，进而根据virtio标准和Hypervisor协作，提高客户机的性能（特别是I/O性能）。

![virtio基本架构](../virtio学习/architecture.gif)

前端驱动（Front-end driver）是在客户机中存在的驱动程序模块，而后端处理器程序（Back-end driver）是在QEMU中实现的。

virtio是半虚拟化驱动的方式，其I/O性能几乎可以达到和native差不多的I/O性能。但是virtio必须要客户机安装特定的virtio驱动使其知道是运行在虚拟化环境中，并按照virtio的规定格式进行数据传输。

Linux2.6.24及其以上版本的内核都支持virtio。由于virtio的后端处理程序是在位于用户空间的QEMU中实现的，所以宿主机中只需要比较新的内核即可，不需要特别地编译与virtio相关地驱动。但是客户机需要有特定地virtio驱动程序支持，以便客户机处理I/O操作请求时调用virtio驱动。

### 层次结构

![virtio 层次结构](../virtio学习/virtiolayer.png)

每一个virtio设备（例如：块设备或网卡），在系统层面看来，都是一个pci设备。  
PCI设备标识的Vendor ID `0x1AF4` (`PCI_VENDOR_ID_REDHAT_QUMRANET`), and Device ID `0x1000` through `0x107F` 都是virtio device，其中 `0x1000` through `0x103F` 为legacy device，而 `0x1040` through `0x107F` 为 modern device。分别见 *drivers/virtio/virtio_pci_legacy.c* 和 *drivers/virtio/virtio_pci_modern.c*。  
```c
/* drivers/virtio/virtio_pci_common.c */
/* Qumranet donated their vendor ID for devices 0x1000 thru 0x10FF. */
static const struct pci_device_id virtio_pci_id_table[] = {
    { PCI_DEVICE(PCI_VENDOR_ID_REDHAT_QUMRANET, PCI_ANY_ID) },
    { 0 }
};
```

这些设备之间，有共性部分，也有差异部分。
1. 共性部分：
这些设备都需要挂接相应的buffer队列操作 *virtqueue_ops* ，都需要申请若干个buffer队列，当执行io输出时，需要向队列写入数据；都需要执行 `pci_iomap()` 将设备配置寄存器区间映射到内存区间；都需要设置中断处理；等中断来了，都需要从队列读出数据，并通知虚拟机系统，数据已入队。

2. 差异部分：
设备中系统中，如何与业务关联起来。各个设备不相同。例如，网卡在内核中是一个net_device，与协议栈系统关联起来。同时，向队列中写入什么数据，数据的含义如何，各个设备不相同。队列中来了数据，是什么含义，如何处理，各个设备不相同。

如果每个virtio设备都完整实现自己的功能，又会形成浪费。
针对这个现象，virtio又设计了 `virtio_pci` 模块，以处理所有virtio设备的共性部分。这样一来，所有的virtio设备，在系统层面看来，都是一个pci设备，其设备驱动都是virtio_pci。
但是，`virtio_pci` 并不能完整的驱动任何一个设备。  
因此，`virtio_pci` 在probe（接管）每一个设备时，根据每个pci设备的subsystem device id来识别出这具体是哪一种virtio设备，然后相应的向内核注册一个 virtio 设备。见 *include/uapi/linux/virtio_ids.h* 。  
```c
#define VIRTIO_ID_NET       1 /* virtio net */
#define VIRTIO_ID_BLOCK     2 /* virtio block */
#define VIRTIO_ID_CONSOLE   3 /* virtio console */
#define VIRTIO_ID_RNG       4 /* virtio rng */
#define VIRTIO_ID_BALLOON   5 /* virtio balloon */
#define VIRTIO_ID_RPMSG     7 /* virtio remote processor messaging */
#define VIRTIO_ID_SCSI      8 /* virtio scsi */
#define VIRTIO_ID_9P        9 /* 9p virtio console */
#define VIRTIO_ID_RPROC_SERIAL 11 /* virtio remoteproc serial link */
#define VIRTIO_ID_CAIF         12 /* Virtio caif */
#define VIRTIO_ID_GPU          16 /* virtio GPU */
#define VIRTIO_ID_INPUT        18 /* virtio input */
#define VIRTIO_ID_VSOCK        19 /* virtio vsock transport */
#define VIRTIO_ID_CRYPTO       20 /* virtio crypto */
```
当然，在注册 virtio 设备之前， virtio_pci 驱动已经为此设备做了诸多共性的操作。同时，还为设备提供了各种操作的适配接口，例如，一些常用的pci设备操作，还有申请buffer队列的操作。这些操作，都通过 `virtio_config_ops` 结构变量来适配。  

在启动virtio device后，Guest OS中不但在 */sys/bus/pci/devices* 中出现一个对应的pci设备，而且在 */sys/bus/virtio/devices* 中也会出现一个virtio类型设备。  
Guest OS在初始化过程中会扫描pci bus num = 0的host bridge，也就是北桥，通过它来发现其下挂载的一系列pci设备(包括桥设备)，因此新的virtio设备必须对外展现出一个pci的接口。  
如果QEMU成功模拟了一个挂在host bridge上的pci设备，那么Guest OS将会通过PCI总线扫描发现之，继而把该设备添加到系统中，会在 */sys/bus/pci/devices* 目录下出现一个新的pci设备。

在 *virtio-pci* PCI设备probe时候，在 `virtio_pci_probe()` 函数（源码在 *drivers/virtio/virtio_pci_common.c*）中，它将调用 `register_virtio_device()`，后者将把一个virtio类型(`struct  virtio_device`)的设备加入到系统，由于该设备所属的总线是`struct bus_type virtio_bus`(源码在 *drivers/virtio/virtio.c* ), 导致 */sys/bus/virtio/devices/* 目录下出现一个新的设备。

+ [KVM+QEMU世界中的pci总线与virtio总线](http://m.blog.chinaunix.net/uid-29827071-id-5821391.html)   

### 代码层次结构

从虚拟机的角度看，virtio的类层次结构如下图所示。

在顶级的是 `virtio_driver`，它在Guest OS中表示前端驱动程序Front-End简称FE。与该驱动程序匹配的设备由 `virtio_device`（设备在虚拟机操作系统中的表示）封装。这引用 `virtio_config_ops` 结构（它定义配置 `virtio device`的操作）。 `virtio_device` 由 `virtqueue` 引用。最后，每个 `virtqueue` 对象引用 virtqueue_ops 对象，后者定义处理 hypervisor 的驱动程序的底层队列操作。这里需要说明的是，Linux并没有实现论文中的 
`struct virtqueue_ops`，但是实现了对于`virtqueue`操作的函数。下面会讲到。


![virtio基本数据结构层次](../virtio学习/structure.gif)

#### virtio_driver

```c
// https://elixir.bootlin.com/linux/v4.15.4/source/include/linux/virtio.h
/**
 * virtio_driver - operations for a virtio I/O driver
 * @driver: underlying device driver (populate name and owner).
 * @id_table: the ids serviced by this driver.
 * @feature_table: an array of feature numbers supported by this driver.
 * @feature_table_size: number of entries in the feature table array.
 * @feature_table_legacy: same as feature_table but when working in legacy mode.
 * @feature_table_size_legacy: number of entries in feature table legacy array.
 * @probe: the function to call when a device is found.  Returns 0 or -errno.
 * @scan: optional function to call after successful probe; intended
 *    for virtio-scsi to invoke a scan.
 * @remove: the function to call when a device is removed.
 * @config_changed: optional function to call when the device configuration
 *    changes; may be called in interrupt context.
 * @freeze: optional function to call during suspend/hibernation.
 * @restore: optional function to call on resume.
 */
struct virtio_driver {
    struct device_driver driver;
    const struct virtio_device_id *id_table;
    const unsigned int *feature_table;
    unsigned int feature_table_size;
    const unsigned int *feature_table_legacy;
    unsigned int feature_table_size_legacy;
    int (*validate)(struct virtio_device *dev);
    int (*probe)(struct virtio_device *dev);
    void (*scan)(struct virtio_device *dev);
    void (*remove)(struct virtio_device *dev);
    void (*config_changed)(struct virtio_device *dev);
#ifdef CONFIG_PM
    int (*freeze)(struct virtio_device *dev);
    int (*restore)(struct virtio_device *dev);
#endif
};
```

该流程以创建 `virtio_driver` 并通过 `register_virtio_driver` 进行注册开始。   
`virtio_driver` 结构定义上层设备驱动程序(`struct device_driver driver`)、驱动程序支持的设备 ID 的列表（`struct virtio_device_id *id_table`）、一个特性表单（取决于设备类型）（`feature_table`）和一个回调函数列表。

当 hypervisor 识别到与设备列表中的设备 ID 相匹配的新设备时，将调用 `probe` 函数（由 
`virtio_driver` 对象提供）来传入 `virtio_device` 对象。将这个对象和设备的管理数据缓存起来（以独立于驱动程序的方式缓存）。可能要调用 
`virtio_config_ops`函数来获取或设置特定于设备的选项，例如，为 `virtio_blk` 设备获取磁盘的 `Read/Write`状态或设置块设备的块大小，具体情况取决于启动器的类型。

#### virtio_device

```c
// include/linux/virtio.h
/**
 * virtio_device - representation of a device using virtio
 * @index: unique position on the virtio bus
 * @failed: saved value for VIRTIO_CONFIG_S_FAILED bit (for restore)
 * @config_enabled: configuration change reporting enabled
 * @config_change_pending: configuration change reported while disabled
 * @config_lock: protects configuration change reporting
 * @dev: underlying device.
 * @id: the device type identification (used to match it with a driver).
 * @config: the configuration ops for this device.
 * @vringh_config: configuration ops for host vrings.
 * @vqs: the list of virtqueues for this device.
 * @features: the features supported by both driver and device.
 * @priv: private pointer for the driver's use.
 */
struct virtio_device {
    int index;
    bool failed;
    bool config_enabled;
    bool config_change_pending;
    spinlock_t config_lock;
    struct device dev;
    struct virtio_device_id id;
    const struct virtio_config_ops *config;
    const struct vringh_config_ops *vringh_config;
    struct list_head vqs;
    u64 features;
    void *priv;
};

```

注意，`virtio_device` 不包含到 `virtqueue` 的引用（但 `virtqueue` 确实引用了 `virtio_device`）。要识别与该 `virtio_device` 相关联的 `virtqueue`，需要结合使用 `virtio_config_ops` 对象和 `find_vq` 函数。该对象返回与这个 `virtio_device` 实例相关联的虚拟队列。`find_vq` 函数还允许为 `virtqueue` 指定一个回调函数。  

`virtio_driver` 有自己的PCI总线 `virtio_bus`。 `probe`函数用于PCI总线发现设备。比如启动 `virtio_blk` 时，当通过`qemu`启动`guest`的时候如果指定`-device virtio-blk-device`，就会调用`virtio_blk`的 `virtblk_probe` 函数。

#### virtio_config_ops  

对于 `virtio device` 需要指定操作函数，包括PCI配置空间设置和读取，status操作，virtqueue操作，和总线bus操作。  

```c
/* include/linux/virtio_config.h */
/**
 * virtio_config_ops - operations for configuring a virtio device
 * @get: read the value of a configuration field
 *  vdev: the virtio_device
 *  offset: the offset of the configuration field
 *  buf: the buffer to write the field value into.
 *  len: the length of the buffer
 * @set: write the value of a configuration field
 *  vdev: the virtio_device
 *  offset: the offset of the configuration field
 *  buf: the buffer to read the field value from.
 *  len: the length of the buffer
 * @generation: config generation counter
 *  vdev: the virtio_device
 *  Returns the config generation counter
 * @get_status: read the status byte
 *  vdev: the virtio_device
 *  Returns the status byte
 * @set_status: write the status byte
 *  vdev: the virtio_device
 *  status: the new status byte
 * @reset: reset the device
 *  vdev: the virtio device
 *  After this, status and feature negotiation must be done again
 *  Device must not be reset from its vq/config callbacks, or in
 *  parallel with being added/removed.
 * @find_vqs: find virtqueues and instantiate them.
 *  vdev: the virtio_device
 *  nvqs: the number of virtqueues to find
 *  vqs: on success, includes new virtqueues
 *  callbacks: array of callbacks, for each virtqueue
 *      include a NULL entry for vqs that do not need a callback
 *  names: array of virtqueue names (mainly for debugging)
 *      include a NULL entry for vqs unused by driver
 *  Returns 0 on success or error status
 * @del_vqs: free virtqueues found by find_vqs().
 * @get_features: get the array of feature bits for this device.
 *  vdev: the virtio_device
 *  Returns the first 32 feature bits (all we currently need).
 * @finalize_features: confirm what device features we'll be using.
 *  vdev: the virtio_device
 *  This gives the final feature bits for the device: it can change
 *  the dev->feature bits if it wants.
 *  Returns 0 on success or error status
 * @bus_name: return the bus name associated with the device
 *  vdev: the virtio_device
 *      This returns a pointer to the bus name a la pci_name from which
 *      the caller can then copy.
 * @set_vq_affinity: set the affinity for a virtqueue.
 * @get_vq_affinity: get the affinity for a virtqueue (optional).
 */
typedef void vq_callback_t(struct virtqueue *);
struct virtio_config_ops {
    void (*get)(struct virtio_device *vdev, unsigned offset,
            void *buf, unsigned len);
    void (*set)(struct virtio_device *vdev, unsigned offset,
            const void *buf, unsigned len);
    u32 (*generation)(struct virtio_device *vdev);
    u8 (*get_status)(struct virtio_device *vdev);
    void (*set_status)(struct virtio_device *vdev, u8 status);
    void (*reset)(struct virtio_device *vdev);
    int (*find_vqs)(struct virtio_device *, unsigned nvqs,
            struct virtqueue *vqs[], vq_callback_t *callbacks[],
            const char * const names[], const bool *ctx,
            struct irq_affinity *desc);
    void (*del_vqs)(struct virtio_device *);
    u64 (*get_features)(struct virtio_device *vdev);
    int (*finalize_features)(struct virtio_device *vdev);
    const char *(*bus_name)(struct virtio_device *vdev);
    int (*set_vq_affinity)(struct virtqueue *vq, int cpu);
    const struct cpumask *(*get_vq_affinity)(struct virtio_device *vdev,
            int index);
};
```

#### virtqueue

```c
/* include/linux/linux/virtio.h */
/**
 * virtqueue - a queue to register buffers for sending or receiving.
 * @list: the chain of virtqueues for this device
 * @callback: the function to call when buffers are consumed (can be NULL).
 * @name: the name of this virtqueue (mainly for debugging)
 * @vdev: the virtio device this queue was created for.
 * @priv: a pointer for the virtqueue implementation to use.
 * @index: the zero-based ordinal number for this queue.
 * @num_free: number of elements we expect to be able to fit.
 *
 * A note on @num_free: with indirect buffers, each buffer needs one
 * element in the queue, otherwise a buffer will need one element per
 * sg element.
 */
struct virtqueue {
	struct list_head list;
	void (*callback)(struct virtqueue *vq);
	const char *name;
	struct virtio_device *vdev;
	unsigned int index;
    // desc table中free的数量，初始值为 Queue Size
	unsigned int num_free;
	void *priv;
};
```

`virtqueue` 是 **Guest操作系统内存的一部分**，用作Guest和Host的数据传输缓存。  
Host可以在Userspace实现（QEMU），也可以在内核态实现（vHost）。  
它包括了一个可选的回调函数（在 hypervisor 使用缓冲池时调用）、一个到 `virtio_device` 的引用、队列的索引，以及一个引用要使用的底层实现的特殊 `priv` 引用。虽然 `callback` 是可选的，但是它能够动态地启用或禁用回调。  
`virtqueue`数量由设备实现确定，比如网卡有两个virtqueue，一个用于接收数据，另一个用于发送数据。  

针对 virtqueue 的操作包括`add_buf`、`kick`、`get_buf`、`disable_cb`、`enable_cb`等，定义了在guest操作系统和 hypervisor 之间移动命令和数据的方式：

+ `add_buf`

```c
/* drivers/virtio/virtio_ring.c */
int virtqueue_add(struct virtqueue *_vq,
	struct scatterlist *sgs[],
	unsigned int total_sg,
	unsigned int out_sgs,
	unsigned int in_sgs,
	void *data,
	void *ctx,
	gfp_t gfp)
```
add_buf()用于向 queue 中添加一个新的 buffer，参数 data 是一个非空的令牌，用于识别 buffer，当 buffer 内容被消耗后，data 会返回。

该请求以散集列表（scatter-gather list）的形式存在。对于 `add_buf`，guest操作系统提供用于将请求添加到队列的 `virtqueue`、散集列表（地址和长度数组）、用作输出条目（目标是底层 hypervisor）的缓冲池数量，以及用作输入条目（hypervisor 将为它们储存数据并返回到guest操作系统）的缓冲池数量，以及数据。

+ kick & notify
```c
/**
 * virtqueue_kick - update after add_buf
 * @vq: the struct virtqueue
 *
 * After one or more virtqueue_add_* calls, invoke this to kick
 * the other side.
 */
bool virtqueue_kick(struct virtqueue *vq);
/**
 * virtqueue_kick_prepare - first half of split virtqueue_kick call.
 * @vq: the struct virtqueue
 *
 * Instead of virtqueue_kick(), you can do:
 *  if (virtqueue_kick_prepare(vq))
 *      virtqueue_notify(vq);
 *
 * This is sometimes useful because the virtqueue_kick_prepare() needs
 * to be serialized, but the actual virtqueue_notify() call does not.
 */
bool virtqueue_kick_prepare(struct virtqueue *_vq);
/**
 * virtqueue_notify - second half of split virtqueue_kick call.
 */
bool virtqueue_notify(struct virtqueue *vq);
```


当通过 add_buf 向 hypervisor 发出请求时，guest操作系统能够通过 `kick` 函数通知 hypervisor 新的请求。为了获得最佳的性能，guest操作系统应该在通过 kick 发出通知之前将尽可能多的缓冲池装载到 virtqueue。Guest 再调用 `virtqueue_notify()`来通知 host。

+ get_buf()

```c
/**
 * virtqueue_get_buf - get the next used buffer
 * @vq: the struct virtqueue we're talking about.
 * @len: the length written into the buffer
 * Returns NULL if there are no used buffers, or the "data" token
 * handed to virtqueue_add_*().
 */
void *virtqueue_get_buf(struct virtqueue *_vq, unsigned int *len)
```

Guest OS仅需调用该函数或通过提供的 `virtqueue callback` 函数等待通知就可以实现轮询。当Guest OS知道缓冲区可用时，调用 get_buf 返回完成的缓冲区。

该函数返回使用过的 buffer，len 为写入到 buffer 中数据的长度。获取数据，释放 buffer，更新 vring 描述符表格中的 index。

+ virtqueue_disable_cb()

```c
/**
 * virtqueue_disable_cb - disable callbacks
 * @vq: the struct virtqueue we're talking about.
 */
void virtqueue_disable_cb(struct virtqueue *vq);
```

示意 guest 不再需要再知道一个 buffer 已经使用了，也就是关闭 device 的中断。  
驱动会在初始化时注册一个回调函数（在 virtqueue 中由 virtqueue 初始化的 callback 函数），disable_cb()通常在这个 virtqueue 回调函数中使用，用于关闭再次的回调发生。

+ virtqueue_enable_cb()

```c
/**
 * virtqueue_enable_cb - restart callbacks after disable_cb.
 * @vq: the struct virtqueue we're talking about.
 */
bool virtqueue_enable_cb(struct virtqueue *vq);
```

与 disable_cb()刚好相反，用于重新开启设备中断的上报。

### host与guest操作系统之间数据交换流程  

1. guest 添加数据
![virtio 数据交换流-guest add buf](../virtio学习/datachangeflow1.png)
2. guest 通知 host
![virtio 数据交换流-guest 通知 host](../virtio学习/datachangeflow2.png)
3. host读取缓存数据
![virtio 数据交换流-host读取缓存数据](../virtio学习/datachangeflow3.png)
4. host写入缓存数据
![virtio 数据交换流-host写入缓存数据](../virtio学习/datachangeflow4.png)
5. guest读取返回数据
![virtio 数据交换流-guest get buf](../virtio学习/datachangeflow5.png)


### virtio_ring  

+ [VirtIO实现原理——vring数据结构](https://blog.csdn.net/huang987246510/article/details/103739592)   
+ Virtual I/O Device (VIRTIO) Version 1.0 

guest 操作系统（前端）驱动程序通过`virtqueue`与 hypervisor 交互，实现数据的共享。对于 I/O，guest 操作系统提供一个或多个表示请求的缓冲池。

`vring` 是 `virtqueue` 的具体实现方式，在host和guest操作系统之间作内存映射，针对 `vring` 会有相应的描述符表格进行描述。  
前后端驱动通过 `vring` 直接通信，这就绕过了经过 KVM 内核模块的过程，达到提高 I/O 性能的目的。  
框架如下图所示：

![virtqueue实现](../virtio学习/virtio.jpg)

virtio_ring 是 virtio 传输机制的实现，`vring` 引入 `ring buffers` 来作为数据传输的载体。   

从结构上看，virtio_ring 包含 3 部分：  
- Descriptor Table
- Available Ring
- Used Ring


1. Descriptor Table

描述符数组（descriptor table）用于存储真正的buffer，每个描述符都是一个对 buffer 的描述，包含一个 address/length 的配对、下个buffer的指针、两个标志位（下个buffer是否有效和当前buffer是可读/写）。
每个buffer在内部被表示为一个散集列表（scatter-gather），列表中的每个条目表示一个地址和一个长度。  
每个条目的结构体为 `struct vring_desc`，desc table数组大小为 Queue Size。  
```c
/* include/uapi/linux/virtio_ring.h*/
/* Virtio ring descriptors: 16 bytes.  These can chain together via "next". */
struct vring_desc {
    /* Address (guest-physical). */
    __virtio64 addr;
    /* Length. */
    __virtio32 len;
    /* This marks a buffer as continuing via the next field. */
#define VIRTQ_DESC_F_NEXT 1
/* This marks a buffer as device write-only (otherwise device read-only). */
#define VIRTQ_DESC_F_WRITE 2
/* This means the buffer contains a list of buffer descriptors. */
#define VIRTQ_DESC_F_INDIRECT 4
    /* The flags as indicated above. */
    __virtio16 flags;
    /* We chain unused descriptors via this, too */
    /* next desc idx */
    __virtio16 next;
};
```


2. available ring  

available ring用于 guest 端表示哪些描述符链（descriptor chain）当前是可用的。  
available ring由 driver 写，device 读。  
`idx` 表示driver把下一个 descriptor entry 放在ring中的位置。  

```c
struct virtq_avail {
    #define VIRTQ_AVAIL_F_NO_INTERRUPT 1
    le16 flags;
    le16 idx;
    le16 ring[ /* Queue Size */ ];
    le16 used_event; /* Only if VIRTIO_F_EVENT_IDX */
};
```

`flags`：用于指示Host当它处理完buffer，将Descriptor index写入Used Ring之后，是否通过注入中断通知Guest。如果flags设置为0，Host每处理完一次buffer就会中断通知Guest；如果flags为1，不通知Guest。 如果`VIRTIO_F_EVENT_IDX`该特性开启，那么flags的意义将会改变，Guest必须把flags设置为0，然后通过 *used_event* 机制实现通知。
`idx` : 指示Guest下一次添加buffer时，在`ring[]`中取的位置，从0开始。  
`ring[]`：是一个大小为 Queue Size 的数组，entry是存放Descriptor Table Chain的head。  
*used_event* 用作Event方式通知机制，此值用于控制Guest的virtqueue数据发送速度，是QEMU处理avail ring 后的last_avail_idx。  

`struct vring_avail` 的定义如下：  

```c
/*include/uapi/linux/virtio_ring.h*/
struct vring_avail {
	__virtio16 flags;
	__virtio16 idx;
	__virtio16 ring[];
};

```

3. Used Ring  

used ring表示 device记录哪些描述符已经使用。  
used ring 由device 写，driver读取。  

```c
struct virtq_used {
    #define VIRTQ_USED_F_NO_NOTIFY 1
    le16 flags;
    le16 idx;
    struct virtq_used_elem ring[ /* Queue Size */];
    le16 avail_event; /* Only if VIRTIO_F_EVENT_IDX */
};
struct virtq_used_elem {
    /* Index of start of used descriptor chain. */
    le32 id;
    /* Total length of the descriptor chain which was used (written to) */
    le32 len;
};
```

每个used ring中条目 `struct virtq_used_elem` 中 *id* 指定了在 descriptor chain 中的 head 条目，而 *len* 指定了要写入buffer的字节数。  
而 `struct virtq_used` 中的 *flags* 用于指示Guest当它添加完buffer，将Descriptor index写入Avail Ring之后，是否发送notification通知Host。如果flags设置为0，Guest每增加一次buffer就会通知Host，如果flags为1，不通知Host。  
当 `VIRTIO_F_EVENT_IDX` 特性开启时，flags必须被设置成0，Guest使用avail_event方式通知Host。  
`idx` 用于指示Host下一次处理的buffer在Used Ring所的位置。  
`ring[]`：是一个大小为 Queue Size 的数组，entry是存放Descriptor Table Chain的head。  
*avail_event* 用作Event方式通知机制。  

而 `struct vring_used` 定义并不包括 *avail_event*。  

```c
/*include/uapi/linux/virtio_ring.h*/
/* u32 is used here for ids for padding reasons. */
struct vring_used_elem {
	/* Index of start of used descriptor chain. */
	__virtio32 id;
	/* Total length of the descriptor chain which was used (written to) */
	__virtio32 len;
};

struct vring_used {
	__virtio16 flags;
	__virtio16 idx;
	struct vring_used_elem ring[];
};
```

4. vring

需要注意的是，vring的具体内存分布和定义不同，真实内存分布也包括了Descriptor Table，Avail Ring和Used Ring，为了对齐还包括padding，具体为：  
```c
struct vring
{
   // The actual descriptors (16 bytes each)
   struct vring_desc desc[num];
 
   // A ring of available descriptor heads with free-running index.
   __virtio16 avail_flags;
   __virtio16 avail_idx;
   __virtio16 available[num];
   __virtio16 used_event_idx;
 
   // Padding to the next align boundary.
   char pad[];
 
   // A ring of used descriptor heads with free-running index.
   __virtio16 used_flags;
   __virtio16 used_idx;
   struct vring_used_elem used[num];
   __virtio16 avail_event_idx;
}
```

缓冲区的格式、顺序和内容仅对前端和后端驱动程序有意义，FE和BE必须按此分布。  

`struct vring`定义与真实内存分布不同，仅仅记录了queue size和各自得指针。  
```c
/*include/uapi/linux/virtio_ring.h*/
struct vring {
    // qeueue size
	unsigned int num; // ring num is a power of 2
	struct vring_desc *desc;
	struct vring_avail *avail;
	struct vring_used *used;
};
```

从 `vring_init()` 和 `vring_size()` 两个函数可以清楚得得出vring内存分布。  

```c
static inline void vring_init(struct vring *vr, unsigned int num, void *p,
                  unsigned long align)
{
    vr->num = num;
    vr->desc = p;
    vr->avail = p + num*sizeof(struct vring_desc);
    vr->used = (void *)(((uintptr_t)&vr->avail->ring[num] + sizeof(__virtio16)
        + align-1) & ~(align - 1));
}

static inline unsigned vring_size(unsigned int num, unsigned long align)
{
    return ((sizeof(struct vring_desc) * num + sizeof(__virtio16) * (3 + num)
         + align - 1) & ~(align - 1))
        + sizeof(__virtio16) * 3 + sizeof(struct vring_used_elem) * num;
}
```

### `struct vring_virtqueue`

Driver和Device看到的只是`virtqueue`，而`vring`是`virtqueue`的具体实现，`vring_virtqueue` 将`vring`的实现隐藏在了`virtqueue`下面。  

`vring_virtqueue` 关联了 `virtqueue` 和 `vring`，并记录了ring的head和last_used_id等。  

```c
struct vring_desc_state {
    void *data;         /* Data for callback. */
    struct vring_desc *indir_desc;  /* Indirect descriptor, if any. */
};

struct vring_virtqueue {
    struct virtqueue vq;
    /* Actual memory layout for this queue */
    struct vring vring;
    /* Can we use weak barriers? */
    bool weak_barriers;
    /* Other side has made a mess, don't try any more. */
    bool broken;
    /* Host supports indirect buffers */
    bool indirect;
    /* Host publishes avail event idx */
    bool event;
    /* Head of free buffer list. */
    // 当前Descriptor Table中空闲buffer的起始位置
    unsigned int free_head;
    /* Number we've added since last sync. 
     * 上一次通知Host后，Guest往VRing上添加了多少次buffer
     * 每添加一次buffer，num_added加1，每kick一次Host清空
    */
    unsigned int num_added;
    /* Last used index we've seen. 
     * 记录的 used ring中上次使用的的idx。  
    */
    u16 last_used_idx;
    /* Last written value to avail->flags */
    u16 avail_flags_shadow;
    /* Last written value to avail->idx in guest byte order 
     * 记录的 avail ring中下次添加buffer时应该放的idx。  
     * Guest每添加一次buffer，avail_idx_shadow加1指向
    */
    u16 avail_idx_shadow;
    /* How to notify other side. FIXME: commonalize hcalls! */
    bool (*notify)(struct virtqueue *vq);
    /* DMA, allocation, and size information */
    bool we_own_ring;
    size_t queue_size_in_bytes;
    dma_addr_t queue_dma_addr;
#ifdef DEBUG
    /* They're supposed to lock for us. */
    unsigned int in_use;
    /* Figure out if their kicks are too delayed. */
    bool last_add_time_valid;
    ktime_t last_add_time;
#endif
    /* Per-descriptor state. */
    struct vring_desc_state desc_state[];
};

#define to_vvq(_vq) container_of(_vq, struct vring_virtqueue, vq)
```

`VIRTIO_RING_F_INDIRECT_DESC` 决定了 `struct vring_virtqueue`->indirect，即Description Table使用二级索引来记录buffer。  
即申请并构建Description Chain来保存buffer，再将此Description Chain的地址和大小添加到vring中的条目。  


#### qemu vring实现  

+ [VIRTIO VRING工作机制分析](http://oenhan.com/virtio-vring)  

```c
vm_setup_vq()
```

#### virtio 数据传输实例  

+ [VirtIO实现原理——数据传输演示](https://blog.csdn.net/huang987246510/article/details/103708461#_2)  

这篇博客图文并茂，讲解的非常详细了。  

#### vring event  

+ [VIRTIO中的前后端配合限速分析](https://blog.csdn.net/leoufung/article/details/53584970)  

在 `virtqueue_kick_prepare()` 中  
```c
    old = vq->avail_idx_shadow - vq->num_added;
    new = vq->avail_idx_shadow;
    vq->num_added = 0;

    if (vq->event) {
        needs_kick = vring_need_event(virtio16_to_cpu(_vq->vdev, vring_avail_event(&vq->vring)), new, old);
    } else {
        needs_kick = !(vq->vring.used->flags & cpu_to_virtio16(_vq->vdev, VRING_USED_F_NO_NOTIFY));
    }
```

`new` 表示avail ring中下一次添加buffer的idx，而`old`表示avail ring中上一次添加buffer的idx。  

```c
/* include/uapi/linux/virtio_ring.h */
/* We publish the used event index at the end of the available ring, and vice
 * versa. They are at the end for backwards compatibility. */
#define vring_used_event(vr) ((vr)->avail->ring[(vr)->num])
#define vring_avail_event(vr) (*(__virtio16 *)&(vr)->used->ring[(vr)->num])

/* The following is used with USED_EVENT_IDX and AVAIL_EVENT_IDX */
/* Assuming a given event_idx value from the other side, if
 * we have just incremented index from old to new_idx,
 * should we trigger an event? */
static inline int vring_need_event(__u16 event_idx, __u16 new_idx, __u16 old)
{
    /* Note: Xen has similar logic for notification hold-off
     * in include/xen/interface/io/ring.h with req_event and req_prod
     * corresponding to event_idx + 1 and new_idx respectively.
     * Note also that req_event and req_prod in Xen start at 1,
     * event indexes in virtio start at 0. */
    return (__u16)(new_idx - event_idx - 1) < (__u16)(new_idx - old);
}
```

`event_idx`是从 Used Ring获取的 `avail_event`，`avail_event` 是QEMU处理完avail ring之后通知Guest其处理的进度。  

如果`event_idx`超前于上次添加avail ring的位置，说明后端处理较快，如下图所示，则进行通知（`kick`），否则，不通知。    

|---|old|-----|event_idx|----|new_idx|------|  




# virtio的使用

由于传统的QEMU/KVM方式是使用QEMU纯软件模拟I/O设备（网卡、磁盘、显卡），导致效率并不高。在KVM中，可以在客户机使用半虚拟化（paravirtualized drivers）来提高客户机的性能。

## QEMU模拟I/O设备的基本原理

当客户机的设备驱动程序（Device Driver）发起I/O请求时，KVM模块中的I/O操作捕获代码会拦截这次I/O请求，然后经过处理后将本次I/O请求的信息存放到I/O共享页（sharing page），并通知用户控件的QEMU程序。QEMU模拟程序获得I/O操作的具体信息后，交给硬件模拟代码（Emulation Code）来模拟出本次的I/O操作，完成后把结果放回I/O共享页中，并通知KVM模块中的I/O操作捕获代码。最后由KVM模块中的捕获代码读取I/O共享页中的操作结果，把结果返回给客户机中。当然，这个操作过程中客户机作为一个QEMU进程在等待I/O时也可能被阻塞。

另外，当客户机通过DMA访问大块I/O时，QEMU模拟程序不会把操作结果放到I/O共享页中，而是通过内存映射的方法将结果直接写到客户机的内存去，然后通过KVM模块高速客户机DMA操作已经完成。

QEMU模拟I/O设备不需要修改客户端操作系统，可以模拟各种各样的硬件设备，但是每次I/O操作的路径比较长，有太多的VMEntry和VMExit发生，需要多次上下文切换（context switch），多次的数据复制。性能方面很差。


virtio 有分为guest 中的前端程序和qemu中的后端程序。
virtio中的五种前端程序为

> virtio-blk:/drivers/block/virtio-blk.c
> virtio-net:/drivers/net/virtio-net.c
> virtio-pci:/drivers/virtio/virtio-pci.c
> virtio-ballon:/drivers/virtio/virtio-ballon.c
> virtio-console:/drivers/virtio/virtio-console.c

这五种往下调用`/drivers/virtio/virtio.c` -> `/drivers/virtio/virtio_ring.c`

总结一下virtio的flow：`guest->qemu->host kernel ->hw`，如下图所示。

![virtio 通信架构](../virtio学习/virtiopath.gif)


+ [Virtio devices high-level design](https://projectacrn.github.io/latest/developer-guides/hld/hld-virtio-devices.html)  

### 使用virtio_net

为了让虚拟机能够与外界通信，QEMU为虚拟机提供了网络设备，支持的网络设备为：`ne2k_pci,i82551,i82557b,i82559er,rtl8139,e1000,pcnet,virtio`。
虚拟机的网络设备连接在QEMU虚拟的VLAN中。每个QEMU的运行实例是宿主机中的一个进程，而每个这样的进程中可以虚拟一些VLAN，虚拟机网络设备接入这些VLAN中。当某个VLAN上连接的网络设备发送数据帧，与它在同一个VLAN中的其它网路设备都能接收到数据帧。对虚拟机的网卡没有指定其连接的VLAN号时，QEMU默认会将该网卡连入vlan0。

使用virtio_net半虚拟化驱动，可以提高网络吞吐量（throughput）和降低网络延迟（latency），达到原生网卡的性能。


使用virtio_net需要宿主机中的QEMU工具和客户机的virtio_net驱动支持。

#### 检查QEMU是否支持virtio类型的网卡
```
# qemu-system-x86_64 -net nic,model=?
qemu: Supported NIC models: ne2k_pci,i82551,i82557b,i82559er,rtl8139,e1000,pcnet,virtio
```
从输出的支持网卡类型可知，当前qemu-kvm支持virtio网卡类型。

#### 配置虚拟网桥

本系统的网卡为enp4s0，启动了DHCP。[6]

```
sudo ifconfig enp4s0 down # 关闭enp4s0接口，之后ifconfig命令不显示enp4s0接口
sudo brctl addbr br0	# 增加一个虚拟网桥br0
sudo brctl addif br0 enp4s0	# 在br0中添加一个接口enp4s0
sudo brctl stp br0 off	# 由于只有一个网桥，所以关闭生成树协议
sudo brctl setfd br0 1	#设置br0的转发延迟
sudo brctl sethello br0 1	#设置br0的hello时间
sudo ifconfig br0 0.0.0.0 promisc up	# 打开br0接口
sudo ifconfig enp4s0 0.0.0.0 promisc up	# 打开enp4s0接口
sudo dhclient br0	# 从dhcp服务器获得br0的IP地址
```

查看虚拟网桥列表
```
sudo brctl show br0	
```

	bridge name	bridge id		STP enabled	interfaces
	br0		8000.60a44ce7203e	no		enp4s0
查看br0的各个接口信息
```
sudo brctl showstp br0	
```
	br0
	 bridge id		8000.60a44ce7203e
	 designated root	8000.60a44ce7203e
	 root port		   0			path cost		   0
	 max age		  20.00			bridge max age		  20.00
	 hello time		   1.00			bridge hello time	   1.00
	 forward delay		   1.00			bridge forward delay	   1.00
	 ageing time		 300.00
	 hello timer		   0.00			tcn timer		   0.00
	 topology change timer	   0.00			gc timer		 232.85
	 flags			


	enp4s0 (1)
	 port id		8001			state		     forwarding
	 designated root	8000.60a44ce7203e	path cost		   4
	 designated bridge	8000.60a44ce7203e	message age timer	   0.00
	 designated port	8001			forward delay timer	   0.00
	 designated cost	   0			hold timer		   0.00
	 flags			

#### 配置TAP设备操作:

```
sudo tunctl -t tap1	# 创建一个tap1接口，默认允许root用户访问
sudo brctl addif br0 tap1	 # 在虚拟网桥中增加一个tap1接口
sudo ifconfig tap1 0.0.0.0 promisc up	# 打开tap1接口
```
显示br0的各个接口
```
sudo brctl showstp br0
```

	br0
	 bridge id		8000.46105353cee8
	 designated root	8000.46105353cee8
	 root port		   0			path cost		   0
	 max age		  20.00			bridge max age		  20.00
	 hello time		   1.00			bridge hello time	   1.00
	 forward delay		   1.00			bridge forward delay	   1.00
	 ageing time		 300.00
	 hello timer		   0.00			tcn timer		   0.00
	 topology change timer	   0.00			gc timer		  98.28
	 flags			


	enp4s0 (1)
	 port id		8001			state		     forwarding
	 designated root	8000.46105353cee8	path cost		   4
	 designated bridge	8000.46105353cee8	message age timer	   0.00
	 designated port	8001			forward delay timer	   0.00
	 designated cost	   0			hold timer		   0.00
	 flags			

	tap1 (2)
	 port id		8002			state		       disabled
	 designated root	8000.46105353cee8	path cost		 100
	 designated bridge	8000.46105353cee8	message age timer	   0.00
	 designated port	8002			forward delay timer	   0.00
	 designated cost	   0			hold timer		   0.00
	 flags		

为了在系统启动时能够自动配置虚拟网桥和TAP设备，需要重新编辑`/etc/network/interfaces`。
```
auto enp4s0
iface enp4s0 inet dhcp                  
auto br0
iface br0 inet dhcp
#iface br0 inet static
#address 192.168.0.1
#netmask 255.255.255.0 
#gateway 192.168.0.254 
#dns-nameserver 8.8.8.8
bridge_ports enp4s0
bridge_fd 1
bridge_hello 1
bridge_stp off
                                                                                                                                       
auto tap0
iface tap0 inet manual
#iface tap0 inet static
#address 192.168.0.2 
#netmask 255.255.255.0 
#gateway 192.168.0.254                                                                                                               
#dns-nameserver 8.8.8.8
pre-up tunctl -t tap0 -u root 
pre-up ifconfig tap0 0.0.0.0 promisc up
post-up brctl addif br0 tap0 
```

当然还可以参考[7](https://www.linux-kvm.org/page/Networking)，写脚本来设置网络。

#### 启动客户机，指定分配virtio网卡设备


```
sudo qemu-system-x86_64 -enable-kvm -boot c -drive file=ubuntu16.04.qcow2,if=virtio -m 1024 -netdev type=tap,ifname=tap1,script=no,id=net0 -device virtio-net-pci,netdev=net0
```

qemu-system-x86-64命令行解释

- `–enable-kvm` 创建x86的虚拟机需要用到qemu-system-x86_64这个命令，并需要加上`–enable-kvm`来支持kvm加速，不适用KVM加速虚拟机会非常缓慢。
- `boot` 磁盘相关参数，设置客户机启动时的各种选项。`c`表示第一个硬盘。
- `drive` 配置驱动。使用`file`文件作为镜像文件加载到客户机的驱动器中。`if`指定驱动器使用的接口类型，包括了virtio在内。
- `m` 设置客户机内存大小，单位默认为`MB`。也可以用`G`为单位。
- `netdev` 新型的网络配置方法，在宿主机中建立一个网络后端驱动。`TAP`是虚拟网络设备，它仿真了一个数据链路层设备。`TAP`用于创建一个网络桥，使用网桥连接和NAT模式网络的客户机都会用到`TAP`参数。`ifname`指接口名称。`script`用于设置宿主机在启动客户机时自动执行的网络配置脚本，如果不指定，默认为`/etc/qemu-ifup`，如果不需要执行脚本，则设置`script=no`。`id`用于在宿主机中指定的TAP虚拟设备的`ID`。
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
### virtio Serial

+ [Using QEMU Character Devices](http://web.archive.org/web/20150921130005/http://nairobi-embedded.org/qemu_character_devices.html)  

串口通信的样例代码是：
```
-device virtio-serial-pci \
-chardev socket,path=/tmp/foo,server,nowait,id=foo \
-device virtserialport,chardev=foo,name=org.fedoraproject.port.0
```

QEMU的chardev分为backend和frontend。这会向guest创建设备并暴露出串口端口。 `-device virtio-serial` 选项向虚拟机添加了 `virtio-serial-pci` 设备，`-chardev socket,path=/tmp/foo,server,nowait,id=foo` 创建了backend，以 `/tmp/foo` 为path的 UNIX SOCKET用于通信，id为 foo。  `-device virtserialport,chardev=foo,name=org.fedoraproject.port.0` 创建了frontend，它打开了为此设备打开了一个端口，端口名称为“org.fedoraproject.port.0”，并且将foo的chardev 添加到那个port。 来自[QEMU (简体中文) #Copy and paste](https://wiki.archlinux.org/index.php/QEMU_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87))

客户端需要载入 `virtio_console.ko` 内核模块并将端口 `/dev/vport0p1` 提供给用户态程序。
文件系统属性的位置在 `/sys/class/virtio-ports/vport0p1/name` ，它包含了文本 “org.fedoraproject.port.0”。
添加udev规则，在 `/dev/virtio-ports` 中添加一条链接，`/dev/virtio-ports/org.fedoraproject.port.0 -> /dev/vport0p1` ，写入主机 `/tmp/foo` 的数据会被转发到虚拟机，虚拟机中的应用程序就能够从 `/dev/vport0p1` 或者 `/dev/virtio-ports/org.fedoraproject.port.0` 中读数据。 `/dev/vportNp0` 为首个 `virtio console` 预留。

从[kvm -chardev](https://www.cleancss.com/explain-command/kvm/108550)中可以得到`-chardev` 的选项。 或者从`help` 选项获取，如下
```
-chardev socket,id=id[,host=host],port=port[,to=to][,ipv4][,ipv6][,nodelay][,reconnect=seconds]
         [,server][,nowait][,telnet][,reconnect=seconds][,mux=on|off]
         [,logfile=PATH][,logappend=on|off][,tls-creds=ID] (tcp)
-chardev socket,id=id,path=path[,server][,nowait][,telnet][,reconnect=seconds]
         [,mux=on|off][,logfile=PATH][,logappend=on|off] (unix)
```
创建双向socket流，可以是TCP或者UNIX socket，这取决与 `path` 路径是否设置。


### 检查客户端是否启用 virtio_console.ko

检查内核模块是否包含virtio。
```
grep -i virtio /boot/config-$(uname -r)
```

	CONFIG_NET_9P_VIRTIO=m
	CONFIG_VIRTIO_BLK=y
	CONFIG_SCSI_VIRTIO=m
	CONFIG_VIRTIO_NET=y
	CONFIG_CAIF_VIRTIO=m
	CONFIG_VIRTIO_CONSOLE=y
	CONFIG_HW_RANDOM_VIRTIO=m
	CONFIG_DRM_VIRTIO_GPU=m
	CONFIG_VIRTIO=y
	# Virtio drivers
	CONFIG_VIRTIO_PCI=y
	CONFIG_VIRTIO_PCI_LEGACY=y
	CONFIG_VIRTIO_BALLOON=y
	CONFIG_VIRTIO_INPUT=m
	CONFIG_VIRTIO_MMIO=y
	CONFIG_VIRTIO_MMIO_CMDLINE_DEVICES=y

`CONFIG_VIRTIO_CONSOLE=y` 表示 `virtio_console.ko` 已经编译到内核中，默认启动，不用作为可加载模块载入。

### QEMU客户端模式的 UNIX chardev

启动的参数为：
```
...
-chardev socket,path=/tmp/foo,id=foo \
-device virtio-serial-pci 
-device virtserialport,chardev=foo,name=maxwell,nr=2 \
```

需要先在host上启动监听进程：
```
socat UNIX-LISTEN:/tmp/foo  -
```
否则会报错
> qemu-system-x86_64: -chardev socket,path=/tmp/foo,id=foo: Failed to connect socket /tmp/foo: No such file or directory

启动guest之后，在guest中向设备输入字符串：
```
echo foo > /dev/virtio-ports/maxwell
```
host上会得到消息“foo”。


### QEMU服务器模式的 UNIX chardev

```
...
-chardev socket,path=/tmp/foo,server,nowait,id=foo \
-device virtio-serial-pci 
-device virtserialport,chardev=foo,name=maxwell,nr=2 \
```

需要先在host上启动监听进程，这里使用 `ipython` 交互程序：
```
import socket
 
sock = socket.socket(socket.AF_UNIX)
sock.connect("/tmp/foo")
print sock.recv(1024) 
```
在guest中，向virtio-serial port写数据：
```
printf 'abcd' | dd bs=4 status=none of=/dev/virtio-ports/maxwell count=1 seek=0
```
这样即可在host上收到消息“abcd”。

[这里](https://wiki.qemu.org/Features/ChardevFlowControl)还提到用管道的方式传输数据，我就不在这里实验了。

[Features/VirtioSerial](https://fedoraproject.org/wiki/Features/VirtioSerial#How_To_Test)
[Features/ChardevFlowControl 字符设备控制流](https://wiki.qemu.org/Features/ChardevFlowControl)
[KVM中Virtio-serial_API](https://www.linux-kvm.org/page/Virtio-serial_API)

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

如果没有 `kvm_intel` 模块，再使用kvm功能启动QEMU客户端会报错：
> Could not access KVM kernel module: No such file or directory
> qemu-system-x86_64: failed to initialize KVM: No such file or directory

安装模块 `modprobe kvm-intel` 得到错误信息：
> modprobe: ERROR: could not insert 'kvm_intel': Operation not supported

对于内核错误，通过查看日志文件找问题。 `dmesg`
> kvm: disabled by bios

那么好，关机，启动后设置BIOS，设置完成后一定要关机，再启动。万万不可重启。


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

### qemu monitor

QEMU 监控器是终端窗口，可以执行一些命令来查看当前启动的操作系统一些配置和运行状况。
可以通过 `-monitor stdio` 参数启动。
或者在QEMU窗口中使用快捷键 `Ctrl+Alt+2`， 使用 `Ctrl+Alt+1` 切换回普通的客户机。

### VM Screen Resolution

在启动项中添加 `-vga virtio`， 提供了很高的resolution，然后 `Ctrl + Alt + F` 或者在启动项中添加 `-full-screen` 即可。  
[How to increase the visualized screen resolution on QEMU / KVM?](https://superuser.com/questions/132322/how-to-increase-the-visualized-screen-resolution-on-qemu-kvm)  


# 参考
+ [Virtio](http://www.linux-kvm.org/page/Virtio)
+ [QEMU how to setup Tun/Tap + bridge networking](https://tthtlc.wordpress.com/2015/10/21/qemu-how-to-setup-tuntap-bridge-networking/])
+ [QEMU 1: 使用QEMU创建虚拟机](https://my.oschina.net/kelvinxupt/blog/265108)
+ Virtio: towards a de factor standard for virtual I/O devices
+ [访问qemu虚拟机的五种姿势](http://blog.csdn.net/richardysteven/article/details/54807927)
+ [qemu虚拟机与外部网络的通信](http://blog.csdn.net/shendl/article/details/9468227)
+ [Configuring Guest Networking](https://www.linux-kvm.org/page/Networking)
+ [Virtio：针对 Linux 的 I/O 虚拟化框架](https://www.ibm.com/developerworks/cn/linux/l-virtio/)
+ [Virtio 基本概念和设备操作](https://www.ibm.com/developerworks/cn/linux/1402_caobb_virtio/)
+ virtio: Towards a De-Facto Standard For Virtual I/O Devices
+ [Virtio](https://wiki.osdev.org/Virtio)
+ [Virtio Spec Overview](https://kernelgo.org/virtio-overview.html)  