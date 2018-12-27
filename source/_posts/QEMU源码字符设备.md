---
title: QEMU源码字符设备
date: 2018-12-05 15:59:03
tags:
- QEMU
- virtio
categories:
- QEMU
---
分析QEMU源码中的 `virtconsole` 字符设备文件，选择 `QEMU v2.12.0` 版本。

<!-- more -->

QEMU字符设备有host和guest部分，在当前版本中，使用命令参数 `-chardev` 来创建host部分，用命令参数 `-device` 创建guest设备。即：
```
    -chardev HOST-OPTS...,id=CHR-ID
    -device DEVNAME,chardev=CHR-ID,DEV-OPTS...
```
`DEVNAME` 依赖于机器类型，因此，对于 `pc` 类型，
```
* -serial becomes -device isa-serial,iobase=IOADDR,irq=IRQ,index=IDX
```
这可以控制 I/O ports 和 IRQs。
而要研究的 `virtioconsole` 的启动参数：
```
  -device virtio-serial-pci,class=C,vectors=V,ioeventfd=IOEVENTFD,max_ports=N
  -device virtconsole,is_console=NUM,nr=NR,name=NAME
```

# 命名规则

+ 变量是lower_case_with_underscores; 
+ 结构类型名称是CamelCase。 
+ 枚举类型名和函数类型名称也是CamelCase。
+ 标量类型名称是lower_case_with_underscores_ending_with_a_t，就像POSIX的uint64_t。
+ 以qemu_作前缀的函数是包装标准库函数。

# 结构体
串口端口结构体
+ VirtIOSerialPort
	+ DeviceState dev; 这是一个Object结构体
	+ QTAILQ_ENTRY(VirtIOSerialPort) next; 
	+ VirtIOSerial \*vser;	此变量可以得到 virtio device 和 qdev bus
	+ VirtQueue \*ivq, \*ovq; 一个输入virtqueue，用于从guest读数据，一个输出virtqueue，用于向guest写数据
	+ char \*name; 此名称将发送给guest虚拟机并通过sysfs导出。 guest可以根据此信息创建符号链接。 \
	名称采用反向fqdn格式，如org.qemu.console.0
	+ uint32_t id; 此ID有助于识别guest和host之间的端口。客户端发送带有此ID的"tag"，其中包含它发送的每个数据包，然后host可以找出将此数据发送到哪个关联设备
	+ VirtQueueElement \*elem; 这是我们从virtqueue中弹出的元素。 消耗guest数据的慢速后端（例如qemu chardevs的文件后端）可能导致guest阻塞，直到所有输出被刷新。 这是我们不希望的，所以我们记下弹出的最后一个元素，并在后端再次写入后继续使用它。
	+ uint32_t iov_idx;	弹出的elem在iov buffer的索引
    + uint64_t iov_offset;	弹出的elem在iov buffer的偏移量
    + QEMUBH \*bh; 	在非节流时，我们使用下半部分来调用flush_queued_data。
    + bool guest_connected;	 相关的guest设备是否open
    + bool host_connected;	host上的设备是否为IO打开
    + bool throttled;	应用是否向要接受数据

串口端口结构体类
+ VirtIOSerialPortClass
	+ DeviceClass parent_class;	这是个DeviceClass结构体
    + bool is_console;	此设备是否和guest中的hvc绑定， hvc是啥？
    + DeviceRealize realize函数;	每个port新设备被bus发现后调用的回调函数
    + DeviceUnrealize unrealize函数; 当一个port被热拔出或移除后调用的函数
    + set_guest_connected函数;	对于guest event的回调函数，guest打开设备或关闭设备。
    + enable_backend函数;	启用或关闭virtio serial port的backend；
    + guest_ready函数;	 Guest准备好接受数据
	+ have_data函数; guest向端口写入数据，数据在此函数中处理；
	+ guest_writable; 每次guest将buffer入队列后，这依赖于guest端和host端已连接。

串口，包含了设备和BUS
+ struct VirtIOSerial
	+ VirtIODevice parent_obj;
	+ VirtQueue \*c_ivq, \*c_ovq;
    + VirtQueue \*\*ivqs, \*\*ovqs;
    + VirtIOSerialBus bus;
    + QTAILQ_HEAD(, VirtIOSerialPort) ports;
    + QLIST_ENTRY(VirtIOSerial) next;
    + uint32_t \*ports_map;
    + struct VirtIOSerialPostLoad \*post_load;
    + virtio_serial_conf serial;
    + uint64_t host_features;
};

# virtconsole

virtconsole属于字符设备chardev，分为backend和frontend，backend位于源码树 `hw/char/virt-console.c`中。它对应的frontend前端驱动位于[`linux/drivers/char/virtio_console.c`](https://elixir.bootlin.com/linux/v4.4.139/source/drivers/char/virtio_console.c)。

## virtserialport设备

最先注册的Type变量 `virtserialport_info` ，从名字可以看出这是串口端口类型。
它的Type名称是 `TYPE_VIRTIO_CONSOLE_SERIAL_PORT` ，即 `virtserialport` 。
父类型名称是 `TYPE_VIRTIO_SERIAL_PORT` ，类实例对象Object分配的大小取结构体 `VirtConsole` 的大小，设置了类对象ObjectClass的初始化函数。
```
#define TYPE_VIRTIO_CONSOLE_SERIAL_PORT "virtserialport"
typedef struct VirtConsole {
    VirtIOSerialPort parent_obj;

    CharBackend chr;
    guint watch;
} VirtConsole;

static const TypeInfo virtserialport_info = {
    .name          = TYPE_VIRTIO_CONSOLE_SERIAL_PORT,
    .parent        = TYPE_VIRTIO_SERIAL_PORT,
    .instance_size = sizeof(VirtConsole),
    .class_init    = virtserialport_class_init,
};
```

`virtserialport_info` 在类对象初始化函数里面，
```
static void virtserialport_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_CLASS(klass);

    k->realize = virtconsole_realize;
    k->unrealize = virtconsole_unrealize;
    k->have_data = flush_buf;
    k->set_guest_connected = set_guest_connected;
    k->enable_backend = virtconsole_enable_backend;
    k->guest_writable = guest_writable;
    dc->props = virtserialport_properties;
}
```
逐行分析， `DEVICE_CLASS(klass)` 将ObjectClass 转换成名称为 `TYPE_DEVICE` 的 Object类型，即 `DeviceClass`。
```
#define DEVICE_CLASS(klass) OBJECT_CLASS_CHECK(DeviceClass, (klass), TYPE_DEVICE)
```
在`VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_CLASS(klass);` 中

```
#define VIRTIO_SERIAL_PORT_CLASS(klass) \
     OBJECT_CLASS_CHECK(VirtIOSerialPortClass, (klass), TYPE_VIRTIO_SERIAL_PORT)

typedef struct VirtIOSerialPortClass {
    DeviceClass parent_class;
    ...
    }VirtIOSerialPortClass;
```

`VIRTIO_SERIAL_PORT_CLASS` 同样将ObjectClass转换成名称为 `TYPE_VIRTIO_SERIAL_PORT` 的 Object类型，即 `VirtIOSerialPortClass`。
+ `VirtIOSerialPortClass` 的父类是 `DeviceClass` 。
+ `realize` 新设备在bus上被发现时的初始化函数；
+ `unrealize` 设备在bus上被热拔货移除后的销毁函数；
+ `have_data` guest向端口写入数据，数据在此函数中处理；
+ `set_guest_connected` guest事件的回调函数，用于guest打开\关闭设备；
+ `enable_backend` 启用或关闭virtio serial port的backend；
+ `guest_writable` 每次guest将buffer入队列后 。


而设置DeviceClass的属性props仅仅设置了 `chardev` 参数，此参数是 `struct CharBackend` ，此结构体还没仔细研究。
在启动设备时候，命令行参数为： `-device virtserialport,CHR-ID,DEV-OPTS...`。
```
static Property virtserialport_properties[] = {
    DEFINE_PROP_CHR("chardev", VirtConsole, chr),
    DEFINE_PROP_END_OF_LIST(),
};
```

### virtio-serial bus的接口


```
/* Interface to the virtio-serial bus */

/*
 * Open a connection to the port
 *   Returns 0 on success (always).
 */

int virtio_serial_open(VirtIOSerialPort *port);

/*
 * Close the connection to the port
 *   Returns 0 on success (always).
 */
int virtio_serial_close(VirtIOSerialPort *port);

/*
 * Send data to Guest
 */
ssize_t virtio_serial_write(VirtIOSerialPort *port, const uint8_t *buf,
                            size_t size);

/*
 * Query whether a guest is ready to receive data.
 */
size_t virtio_serial_guest_ready(VirtIOSerialPort *port);

/*
 * Flow control: Ports can signal to the virtio-serial core to stop
 * sending data or re-start sending data, depending on the 'throttle'
 * value here.
 */
void virtio_serial_throttle_port(VirtIOSerialPort *port, bool throttle);
```


### `VirtConsole` 结构体的继承关系是

```
\include\hw\virtio\virtio-serial.h
/*
 * This is the state that's shared between all the ports.  Some of the
 * state is configurable via command-line options. Some of it can be
 * set by individual devices in their initfn routines. Some of the
 * state is set by the generic qdev device init routine.
 */
struct VirtIOSerialPort {
    DeviceState dev;
    ...
    }

/**
 * DeviceState:
 * @realized: Indicates whether the device has been fully constructed.
 *
 * This structure should not be accessed directly.  We declare it here
 * so that it can be embedded in individual device state structures.
 */
struct DeviceState {
    /*< private >*/
    Object parent_obj;
    ...
}
```


	`VirtConsole`
	==> `VirtIOSerialPort`
	==> `DeviceState`
	==> `Ojbect`


## virtconsole设备
说完了 `virtserialport` 端口，在来看看另一设备 `virtconsole` 。
设备名称是 `virtconsole` ，它的父类是 `TYPE_VIRTIO_CONSOLE_SERIAL_PORT`。
设置了类对象ObjectClass的初始化函数。

```
static const TypeInfo virtconsole_info = {
    .name          = "virtconsole",
    .parent        = TYPE_VIRTIO_CONSOLE_SERIAL_PORT,
    .class_init    = virtconsole_class_init,
};
```

顺腾摸瓜找下去，
```
\hw\char\virtio-serial-bus.c
static const TypeInfo virtio_serial_port_type_info = {
    .name = TYPE_VIRTIO_SERIAL_PORT,
    .parent = TYPE_DEVICE,
 	...
};

\include\hw\qdev-core.h
#define TYPE_DEVICE "device"
```
这些设备的继承关系是：

	"virtconsole" 
	==> TYPE_VIRTIO_CONSOLE_SERIAL_PORT("virtserialport") 
	==> TYPE_VIRTIO_SERIAL_PORT("virtio-serial-port") 
	==> TYPE_DEVICE("device")


`virtconsole_info` 类型的类对象初始化函数如下。 将类对象转换成类型为TYPE_VIRTIO_SERIAL_PORT 的 结构体 `VirtIOSerialPortClass`，设置此设备为guest中的hvc。
```
static void virtconsole_class_init(ObjectClass *klass, void *data)
{
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_CLASS(klass);

    k->is_console = true;
}
```
`virtserialport` 和 `virtconsole` 模拟的串口，他们使用的设备为PCI设备 `virtio-serial-pci` ，用于连接virtio serial设备和console port的bus为  `virtio-serial-bus` 。

# virtio-serial-bus
总线类型实例的注册有三个类型信息： `virtser_bus_info` 、 `virtio_serial_port_type_info` 、 `virtio_device_info` ，下面分别展开介绍。
```
\hw\char\virtio-serial-bus.c
static void virtio_serial_register_types(void)
{
    type_register_static(&virtser_bus_info);
    type_register_static(&virtio_serial_port_type_info);
    type_register_static(&virtio_device_info);
}
```

## virtser_bus_info

```
#define TYPE_VIRTIO_SERIAL_BUS "virtio-serial-bus"
#define VIRTIO_SERIAL_BUS(obj) \
      OBJECT_CHECK(VirtIOSerialBus, (obj), TYPE_VIRTIO_SERIAL_BUS)

static void virtser_bus_class_init(ObjectClass *klass, void *data)
{
    BusClass *k = BUS_CLASS(klass);
    k->print_dev = virtser_bus_dev_print;
}

static const TypeInfo virtser_bus_info = {
    .name = TYPE_VIRTIO_SERIAL_BUS,
    .parent = TYPE_BUS,
    .instance_size = sizeof(VirtIOSerialBus),
    .class_init = virtser_bus_class_init,
};
```
设备 "virtio-serial-bus" 并没有作更多有意思的事情，仅仅在类对象初始化函数中，设置了输出设备信息的函数指针。
对象的大小设置为 `VirtIOSerialBus` 结构体的大小。


### VirtIOSerialBus

结构体 `VirtIOSerialBus` 是最顶端的 virtio-serial总线，将作为设备运行。
父类型是 `BusState`，而 `BusState` 结构体的父类型也是 `Object` 。

```
/* The virtio-serial bus on top of which the ports will ride as devices */
struct VirtIOSerialBus {
    BusState qbus;

    /* This is the parent device that provides the bus for ports. */
    VirtIOSerial *vser;

    /* The maximum number of ports that can ride on top of this bus */
    uint32_t max_nr_ports;
};

struct BusState {
    Object obj;
    DeviceState *parent;
    char *name;
    ...
};
```
## virtio_serial_port_type_info

```
static const TypeInfo virtio_serial_port_type_info = {
    .name = TYPE_VIRTIO_SERIAL_PORT,
    .parent = TYPE_DEVICE,
    .instance_size = sizeof(VirtIOSerialPort),
    .abstract = true,
    .class_size = sizeof(VirtIOSerialPortClass),
    .class_init = virtio_serial_port_class_init,
};
```
+ instance_size： Object的大小为结构体 `VirtIOSerialPort` 的大小。
+ abstract： 此TypeInfo结构体不能被直接实例化
+ class_size ： ObjectClass的大小为结构体 `VirtIOSerialPortClass` 的大小。
+ class_init ： 在父类初始化完成后，执行virtio_serial_port_class_init。


```
static void virtio_serial_port_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *k = DEVICE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_INPUT, k->categories);
    k->bus_type = TYPE_VIRTIO_SERIAL_BUS;
    k->realize = virtser_port_device_realize;
    k->unrealize = virtser_port_device_unrealize;
    k->props = virtser_props;
}
```

在类初始化函数中，设置了DeviceClass中总线类型bus_type为前文提到的另一设备 `TYPE_VIRTIO_SERIAL_BUS` 。设置了静态属性，并实现了虚函数 realize 与 unrealize。

## virtio_device_info
`virtio_device_info ` 实现的是virtio串口设备。
```
static const TypeInfo virtio_device_info = {
    .name = TYPE_VIRTIO_SERIAL,
    .parent = TYPE_VIRTIO_DEVICE,
    .instance_size = sizeof(VirtIOSerial),
    .class_init = virtio_serial_class_init,
    .interfaces = (InterfaceInfo[]) {
        { TYPE_HOTPLUG_HANDLER },
        { }
    }
};

\include\hw\virtio\virtio-serial.h
#define TYPE_VIRTIO_SERIAL "virtio-serial-device"
#define VIRTIO_SERIAL(obj) \
        OBJECT_CHECK(VirtIOSerial, (obj), TYPE_VIRTIO_SERIAL)
```

`virtio-serial-device` 的父类是virtio 设备，该对象是 `VirtIOSerial`。 在类对象初始化中，实现了类 `DeviceClass` 、 `VirtioDeviceClass` 、 `HotplugHandlerClass` 中的属性和虚函数。

```
static void virtio_serial_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);
    HotplugHandlerClass *hc = HOTPLUG_HANDLER_CLASS(klass);

    QLIST_INIT(&vserdevices.devices);

    dc->props = virtio_serial_properties;
    dc->vmsd = &vmstate_virtio_console;
    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);
    vdc->realize = virtio_serial_device_realize;
    vdc->unrealize = virtio_serial_device_unrealize;
    vdc->get_features = get_features;
    vdc->get_config = get_config;
    vdc->set_config = set_config;
    vdc->set_status = set_status;
    vdc->reset = vser_reset;
    vdc->save = virtio_serial_save_device;
    vdc->load = virtio_serial_load_device;
    hc->plug = virtser_port_device_plug;
    hc->unplug = qdev_simple_device_unplug_cb;
}
```
此设备貌似在当前文件中与总线串口端口没有任何关系，它在virtio_serial_pci中得到引用。

```
hw\virtio\virtio-pci.c
static void virtio_serial_pci_instance_init(Object *obj)
{
    VirtIOSerialPCI *dev = VIRTIO_SERIAL_PCI(obj);

    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                                TYPE_VIRTIO_SERIAL);
}
```



# virtio-serial-pci




	"virtconsole" 
	==> TYPE_VIRTIO_CONSOLE_SERIAL_PORT("virtserialport") 
	==> TYPE_VIRTIO_SERIAL_PORT("virtio-serial-port") 
	==> TYPE_DEVICE("device")

=> virtio_serial_port_class_init
=> virtserialport_class_init
=> virtconsole_class_init

=> virtser_bus_class_init
=> virtio_serial_class_init
=> virtio_serial_pci_class_init
启动
=> virtio_serial_pci_instance_init
=> virtio_serial_pci_realize
=> virtio_serial_device_realize
=> virtser_port_device_realize
=> virtconsole_realize


# 参考文献

1. [GettingStartedDevelopers#Getting_to_know_the_code](https://wiki.qemu.org/Documentation/GettingStartedDevelopers#Getting_to_know_the_code)
2. [QDEV](http://www.linux-kvm.org/images/f/fe/2010-forum-armbru-qdev.pdf)
3. 源码树 `docs/qdev-device-use.txt`
4. [Documentation/QOMConventions](https://wiki.qemu.org/Documentation/QOMConventions)
5. [[Qemu-devel] qdev for programmers writeup](https://lists.nongnu.org/archive/html/qemu-devel/2011-07/msg00842.html)
6. [QEMU/Devices/Virtio](https://en.wikibooks.org/wiki/QEMU/Devices/Virtio)
7. [Virtio-serial_API 包括Guest API 和 Host API](http://www.linux-kvm.org/page/Virtio-serial_API)
