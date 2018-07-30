---
title: qemu源码添加设备
date: 2018-07-23 15:21:39
tags:
- qemu
- emulate
- virtio
categories:
- qemu
---
本篇博客记录在qemu源码中添加新的设备文件，以添加virtio设备为例子。
<!-- more -->

qemu中需要模拟出设备与总线的关系。因为

	在主板上，一个device会通过bus与其他的device相连接，一个device上可以通过不同的bus端口连接到其他的device，而其他的device也可以进一步通过bus与其他的设备连接，同时一个bus上也可以连接多个device，这种device连bus、bus连device的关系，qemu是需要模拟出来的。为了方便模拟设备的这种特性，面向对象的编程模型也是必不可少的。

# QOM

`QEMU Object Model (QOM)` 模型提供了注册类型的框架，这些类型包括 总线bus、接口interface、设备device等。

以 `QEMU v2.12.0` 版本中的 `hw/misc/pci-testdev.c` 文件为例子，分析QOM创建新类型的过程。

`QOM` 创建新类型时需要用到的数据结构为： `OjectClass` 、 `Object` 、 `TypeInfo` ，基本结构定义在 `include\qom\object.h` ， 文件中的注释非常详细，对数据结构的字段说明和QOM模型的用法。 `TypeImpl` 定义在 `qom/object.c` 中，没有注释。

+ Object: 是所有对象的base Object
+ ObjectClass: 是所有类的基类
+ TypeInfo：是用户用来定义一个Type的工具型的数据结构
+ TypeImpl：对数据类型的抽象数据结构

用户定义了一个TypeInfo，然后调用type_register(TypeInfo )或者type_register_static(TypeInfo )函数，就会生成相应的TypeImpl实例，将这个TypeInfo注册到全局的TypeImpl的hash表中。
TypeInfo的属性与TypeImpl的属性对应，实际上qemu就是通过用户提供的TypeInfo创建的TypeImpl的对象。


## 模块注册
向QOM模块注册自己，这里 `type_init` 是一个宏，在 `include/qemu/module.h` 中。
```
static const TypeInfo pci_testdev_info = {
    .name          = TYPE_PCI_TEST_DEV,	/*类型的名字*/
    .parent        = TYPE_PCI_DEVICE, /*父类的名字*/
    .instance_size = sizeof(PCITestDevState), /*必须向系统说明对象的大小，以便系统为对象的实例分配内存*/
    .class_init    = pci_testdev_class_init, /*在类初始化时就会调用这个函数，将虚拟函数赋值*/
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void pci_testdev_register_types(void)
{
    type_register_static(&pci_testdev_info);
}

type_init(pci_testdev_register_types)
```
这里就是调用会调用 pci_testdev_register_types ，这个函数以 pci_testdev_info 为参数调用了 type_register_static 类型注册函数。
这一过程的目的就是利用 `TypeInfo` 构造出一个 `TypeImpl` 结构，之后插入到一个hash表之中，这个hash表以 `ti->name` ，也就是 `info->name` 为key，value就是生根据 `TypeInfo` 生成的 `TypeImpl` 。

## Class的初始化

在定义新类型中，实现了父类的虚拟方法，那么需要定义新的class的初始化函数，并且在TypeInfo数据结构中，给TypeInfo的 `class_init` 字段赋予该初始化函数的函数指针。

```
PCITestDevState->PCIDeviceClass->DeviceClass->ObjectClass
```

```
static void pci_testdev_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->realize = pci_testdev_realize;
    k->exit = pci_testdev_uninit;
    k->vendor_id = PCI_VENDOR_ID_REDHAT;
    k->device_id = PCI_DEVICE_ID_REDHAT_TEST;
    k->revision = 0x00;
    k->class_id = PCI_CLASS_OTHERS;
    dc->desc = "PCI Test Device";
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    dc->reset = qdev_pci_testdev_reset;
}

//父对象必须是该对象数据结构的第一个属性，以便实现父对象向子对象的cast

```



# 参考
1. [How to add a new device in QEMU source code?](https://stackoverflow.com/questions/28315265/how-to-add-a-new-device-in-qemu-source-code)
2. [QEMU中的对象模型——QOM（介绍篇）](https://blog.csdn.net/u011364612/article/details/53485856)
3. [QOM介绍](http://terenceli.github.io/%E6%8A%80%E6%9C%AF/2017/01/08/qom-introduction)

