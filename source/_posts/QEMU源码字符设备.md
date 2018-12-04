---
title: QEMU源码字符设备
date: 2018-12-04 15:59:03
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

# virtconsole

位于源码树 `hw/char/virt-console.c`中。它对应的前端驱动位于[`linux/drivers/char/virtio_console.c`](https://elixir.bootlin.com/linux/v4.4.139/source/drivers/char/virtio_console.c)。

最先注册的Type变量 `virtserialport_info` ，从名字可以看出这是串口端口类型。它的Type名称是 `TYPE_VIRTIO_CONSOLE_SERIAL_PORT` ，父类型名称是 `TYPE_VIRTIO_SERIAL_PORT` ，类实例对象Object分配的大小取结构体 `VirtConsole` 的大小，设置了类对象ObjectClass的初始化函数。
```
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




```
static const TypeInfo virtconsole_info = {
    .name          = "virtconsole",
    .parent        = TYPE_VIRTIO_CONSOLE_SERIAL_PORT,
    .class_init    = virtconsole_class_init,
};
```


# 参考文献

1. [GettingStartedDevelopers#Getting_to_know_the_code](https://wiki.qemu.org/Documentation/GettingStartedDevelopers#Getting_to_know_the_code)
2. [QDEV](http://www.linux-kvm.org/images/f/fe/2010-forum-armbru-qdev.pdf)
3. 源码树 `docs/qdev-device-use.txt`
4. [Documentation/QOMConventions](https://wiki.qemu.org/Documentation/QOMConventions)
5. [[Qemu-devel] qdev for programmers writeup](https://lists.nongnu.org/archive/html/qemu-devel/2011-07/msg00842.html)

