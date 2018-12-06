---
title: QEMU设备的对象模型QOM
date: 2018-07-23 15:21:39
tags:
- QEMU
- QOM
categories:
- QEMU
---
本篇博客记录在QEMU源码中添加新的设备文件，主要介绍QEMU对象模型`QOM`，以`hw/misc/pci-testdev.c` 和 `hw/misc/edu.c`设备为例子。

<!-- more -->

qemu中需要模拟出设备与总线的关系。因为
在主板上，一个device会通过bus与其他的device相连接，一个device上可以通过不同的bus端口连接到其他的device，其他的device也可以进一步通过bus与其他的设备连接。
同时一个bus上也可以连接多个device，这种device连bus、bus连device的关系，qemu是需要模拟出来的。为了方便模拟设备的这种特性，面向对象的编程模型也是必不可少的。

# QOM

`QEMU Object Model (QOM)` 模型提供了注册类型的框架，这些类型包括 总线bus、接口interface、设备device等。

以 `QEMU v2.12.0` 版本中的 `hw/misc/pci-testdev.c` 文件为例子，分析QOM创建新类型的过程。
>pci-testdev
>>	A PCI "device" that when read and written tests PCI accesses.

> edu
>> 		A PCI device that supports testing both INTx and MSI interrupts and DMA transfers.

`QOM` 创建新类型时需要用到的数据结构为： 类对象`OjectClass` 、 类实例对象`Object` 、 类型对象`TypeInfo` ，基本结构定义在 `include\qom\object.h` ， 文件中的注释非常详细，对数据结构的字段说明和QOM模型的用法。 `TypeImpl` 定义在 `qom/object.c` 中，没有注释。

+ ObjectClass: 是所有类对象的基类，仅仅保存了一个整数 `type` 。
+ Object: 是所有对象的 基类`Base Object` ， 第一个成员变量为指向 `ObjectClass` 的指针。
+ TypeInfo：是用户用来定义一个 `Type` 的工具型的数据结构。
+ TypeImpl：对数据类型的抽象数据结构，TypeInfo的属性与TypeImpl的属性对应。

对象的初始化分为4步：
1. 将 TypeInfo 注册 TypeImpl
2. 实例化 Class（ObjectClass）
3. 实例化 Instance(Object)
4. 添加 Property

`TypeInfo` 结构体里面的字段：

	/**
	 * TypeInfo:
	 * @name: The name of the type.
	 * @parent: The name of the parent type.
	 * @instance_size: The size of the object (derivative of #Object).  If
	 *   @instance_size is 0, then the size of the object will be the size of the
	 *   parent object.
	 * @instance_init: This function is called to initialize an object.  The parent
	 *   class will have already been initialized so the type is only responsible
	 *   for initializing its own members.
	 * @instance_post_init: This function is called to finish initialization of
	 *   an object, after all @instance_init functions were called.
	 * @instance_finalize: This function is called during object destruction.  This
	 *   is called before the parent @instance_finalize function has been called.
	 *   An object should only free the members that are unique to its type in this
	 *   function.
	 * @abstract: If this field is true, then the class is considered abstract and
	 *   cannot be directly instantiated.
	 * @class_size: The size of the class object (derivative of #ObjectClass)
	 *   for this object.  If @class_size is 0, then the size of the class will be
	 *   assumed to be the size of the parent class.  This allows a type to avoid
	 *   implementing an explicit class type if they are not adding additional
	 *   virtual functions.
	 * @class_init: This function is called after all parent class initialization
	 *   has occurred to allow a class to set its default virtual method pointers.
	 *   This is also the function to use to override virtual methods from a parent
	 *   class.
	 * @class_base_init: This function is called for all base classes after all
	 *   parent class initialization has occurred, but before the class itself
	 *   is initialized.  This is the function to use to undo the effects of
	 *   memcpy from the parent class to the descendants.
	 * @class_finalize: This function is called during class destruction and is
	 *   meant to release and dynamic parameters allocated by @class_init.
	 * @class_data: Data to pass to the @class_init, @class_base_init and
	 *   @class_finalize functions.  This can be useful when building dynamic
	 *   classes.
	 * @interfaces: The list of interfaces associated with this type.  This
	 *   should point to a static array that's terminated with a zero filled
	 *   element.
	 */

归纳起来：
TypeInfo中定义了如下几类信息：

	1. Name
		包括自己的Name，Parent的Name。
	2. Class
		ObjectClass的信息包括，class_size，class_data，class相关函数：class_base_init，class_init，class_finalize。
		这些函数都是为了初始化，释放结构体ObjectClass。
	3. Instance
		对象Object信息包括：instance_size，instance相关函数：instance_post_init，instance_init，instance_finalize。
		这些函数都是为了初始化，释放结构体Object。
	4. 其他信息
		abstract是否为抽象。interface数组。



用户定义了一个TypeInfo，然后调用 `type_register(TypeInfo)` 或者 `type_register_static(TypeInfo)` 函数，就会生成相应的TypeImpl实例，将这个TypeInfo注册到全局的TypeImpl的hash表中。
TypeInfo的属性与TypeImpl的属性对应，实际上qemu就是通过用户提供的TypeInfo创建的TypeImpl的对象。

TYPE_DEVICE has a pure virtual method 'init' which is a bit of a misnomer.  The 'init' method is called after construction but before the guest is started for the first time.  In QOM nomenclature, we call this realize.

## 模块注册

向QOM模块注册自己，类似于Linux驱动的注册，通过 `type_init` 宏注册，它在 `include/qemu/module.h` 中。
这个宏调用发生在 `qemu main` 函数之前。
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
这里会调用 `pci_testdev_register_types` ，这个函数以 `pci_testdev_info` 为参数调用了 `type_register_static` 类型注册函数。
这一过程的目的就是利用 `TypeInfo` 构造出一个 `TypeImpl` 结构，之后插入到一个hash表之中，这个hash表以 `ti->name` （TypeImpl ti），也就是 `info->name` 为key，value就是根据 `TypeInfo` 生成的 `TypeImpl` 。
在 `pci_testdev_info` 中得字段 `name` 定义了我们将来启动此设备时候传参 `-device ` 后面跟的值。

## Class的初始化

现在已经有了一个TypeImpl的哈希表。下一步就是初始化每个type了，这一步可以看成是class的初始化，可以理解成每一个type对应了一个class，接下来会初始化class。

由于在初始化每个type时候，调用到的是 `type_initialize` 函数。 
```
static void type_initialize(TypeImpl *ti)
```
如果 `ti->class` 已经存在说明已经初始化了，直接返回。如果有 `parent`，会递归调用 `type_initialize`，即调用父对象的初始化函数。

这里type也有一个层次关系，即QOM 对象的层次结构。在 `pci_testdev_info` 结构的定义中，我们可以看到有一个 `.parent` 域，值为 `TYPE_PCI_DEVICE` 。
这说明 `TYPE_PCI_TEST_DEV` 的 父类型 是 `TYPE_PCI_DEVICE` ，在 `hw/pci/pci.c` 中可以看到 `pci_device_type_info` 的父type是 `TYPE_DEVICE`

	static const TypeInfo pci_device_type_info = {
	    .name = TYPE_PCI_DEVICE,
	    .parent = TYPE_DEVICE,
	    .instance_size = sizeof(PCIDevice),
	    .abstract = true,
	    .class_size = sizeof(PCIDeviceClass),
	    .class_init = pci_device_class_init,
	    .class_base_init = pci_device_class_base_init,
	};

依次往上溯我们可以得到这样一条type的链，
> TYPE_PCI_TEST_DEV->TYPE_PCI_DEVICE->TYPE_DEVICE->TYPE_OBJECT

之后，最重要的就是调用 `parent->class_base_init` 以及 `ti->class_init` 了，这相当于C++里面的构造基类的数据。
在定义新类型中，实现了父类的虚拟方法，那么需要定义新的class的初始化函数，并且在TypeInfo数据结构中，给TypeInfo的 `class_init` 字段赋予该初始化函数的函数指针。
我们以一个 `class_init` 为例，

```
static void pci_testdev_class_init(ObjectClass *klass, void *data)
{
	//父对象必须是该对象数据结构的第一个属性，以便实现父对象向子对象的cast
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
```
`class_init` 构造函数钩子。 这个函数将负责初始化Type的 `ObjectClass` 实例。
这里从 `ObjectClass` 转换成了 `DeviceClass` 。为什么这么转换，还需要从 `	Class` 的层次结构说起。

有这么一种层次 
> PCIDeviceClass->DeviceClass->ObjectClass
可以看成C++中的继承关系，即当然基类就是 `ObjectClass` ，越往下包含的数据越具体。
将一个父类的指针转换为子类的指针是不安全的，为了实现这种转换，各类需要提供强制类型转换的宏
以下面这句为例：
```
PCIDeviceClass *c = PCI_DEVICE_CLASS(kclass);
```
宏 `PCI_DEVICE_CLASS` 定义在 `hw/pci/pci.h` 中。
```
#define PCI_DEVICE_GET_CLASS(obj) \
        OBJECT_GET_CLASS(PCIDeviceClass, obj, TYPE_PCI_DEVICE)

#define PCI_DEVICE_CLASS(klass) \
        OBJECT_CLASS_CHECK(PCIDeviceClass, klass, TYPE_PCI_DEVICE)

#define PCI_DEVICE(obj) \
        OBJECT_CHECK(PCIDevice, obj, TYPE_PCI_DEVICE)
```

其中 `OBJECT_CHECK` 、 `OBJECT_CLASS_CHECK` 、 `OBJECT_GET_CLASS` 的定义如下：

`OBJECT_CHECK` 用于转换`Object`，`OBJECT_CLASS_CHECK` 用于转换 `ObjcetClass` ， `OBJECT_GET_CLASS` 用于从`Object` 获取 `Class`。
```
/**
 * OBJECT_CHECK:
 * @type: The C type to use for the return value.
 * @obj: A derivative of @type to cast.
 * @name: The QOM typename of @type
 *
 * A type safe version of @object_dynamic_cast_assert.  Typically each class
 * will define a macro based on this type to perform type safe dynamic_casts to
 * this object type.
 *
 * If an invalid object is passed to this function, a run time assert will be
 * generated.
 */
#define OBJECT_CHECK(type, obj, name) \
    ((type *)object_dynamic_cast_assert(OBJECT(obj), (name), \
                                        __FILE__, __LINE__, __func__))
```
```
/**
 * OBJECT_CLASS_CHECK:
 * @class_type: The C type to use for the return value.
 * @class: A derivative class of @class_type to cast.
 * @name: the QOM typename of @class_type.
 *
 * A type safe version of @object_class_dynamic_cast_assert.  This macro is
 * typically wrapped by each type to perform type safe casts of a class to a
 * specific class type.
 */
#define OBJECT_CLASS_CHECK(class_type, class, name) \
    ((class_type *)object_class_dynamic_cast_assert(OBJECT_CLASS(class), (name), \
                                               __FILE__, __LINE__, __func__))
```
```
/**
 * OBJECT_GET_CLASS:
 * @class: The C type to use for the return value.
 * @obj: The object to obtain the class for.
 * @name: The QOM typename of @obj.
 *
 * This function will return a specific class for a given object.  Its generally
 * used by each type to provide a type safe macro to get a specific class type
 * from an object.
 */
#define OBJECT_GET_CLASS(class, obj, name) \
    OBJECT_CLASS_CHECK(class, object_get_class(OBJECT(obj)), name)
```

在 `PCI_DEVICE_CLASS` 这个宏定义中，最终会进入 `object_class_dynamic_cast` 函数，在该函数中，根据class对应的type以及typename对应的type，判断是否能够转换，判断的主要依据就是type_is_ancestor， 这个判断target_type是否是type的一个祖先，如果是当然可以进行转换，否则就不行。

总之，就是从最开始的 `TypeImpl` 初始化了每一个type对应的 `ObjectClass *class` ，并且构建好了各个Class的继承关系。

## Object的构造

我们上面已经看到了type哈希表的构造以及 `Class` 的初始化，接下来讨论具体设备的创建。

```
typedef struct PCITestDevState {
    /*< private >*/
    PCIDevice parent_obj;
    /*< public >*/

    MemoryRegion mmio;
    MemoryRegion portio;
    IOTest *tests;
    int current;
} PCITestDevState;

// hw/pci/pci.h
struct PCIDevice {
    DeviceState qdev;
    ...
}

// hw/qdev-core.h
struct DeviceState {
    /*< private >*/
    Object parent_obj;
    ...
}
```
Object 的继承
> PCITestDevState->PCIDevice->DeviceState->Object

Object的创建由 ` k->realize = pci_testdev_realize;` 函数实现，不同于type和class的构造，`Object` 当然是根据需要创建的，只有在命令行指定了设备或者是热插一个设备之后才会有 `Object` 的创建。 `Class` 和 `Object` 之间是通过 `Object` 的 `class域` 联系在一起的。

# 属性

属性分为类对象(ObjectClass)属性和类实例对象(Object)属性，存储于 properties 成员中。properties 是一个 GHashTable ，存储了 属性名 到 ObjectProperty 的映射。

## 属性对象
属性对象包含属性名称、类型、描述，类型对应的属性结构，以及相应访问函数。
```
typedef struct ObjectProperty
{
    gchar *name;
    gchar *type;
    gchar *description;
    ObjectPropertyAccessor *get;
    ObjectPropertyAccessor *set;
    ObjectPropertyResolve *resolve;
    ObjectPropertyRelease *release;
    void *opaque;
} ObjectProperty;
```
## 静态属性

凡是在代码中就已经定义好名称和类型的属性，都是静态属性。包括在初始化过程中添加 和 props 。

### 初始化过程中添加

类实例  初始化函数`instance_init`中调用的 `object_property_add` 函数。 该属性会直接加到类实例对象的properties中。

### props

一些类对象会在 `class_init` 中设置 `ObjectClass` 对象的 `props` 成员。 例如：
```
static Property test_properties[] = {
    DEFINE_PROP_STRING("test_string", struct_type, , struct_type_property),
    DEFINE_PROP_END_OF_LIST()
};
```
### 查看
可通过命令查看设备的静态属性，参数为设备 `TypeInfo` 的 `name` ：

```
x86_64-softmmu/qemu-system-x86_64 -device pci-testdev,?
```
## 动态属性
指在运行时动态进行添加的属性。比如用户通过参数传入了一个设备，需要作为属性和其它设备关联起来。

典型的动态属性就是 `child<>` 和 `link<>` (因为其类型就是这样构造的，后文简称child和link) 。

### child
`child` 实现了 组成composition关系，表示一个设备(parent)创建了另外一个设备(child)，parent掌控child的生命周期，负责向其发送事件。一个device只能有一个parent，但能有多个child。这样就构成一棵组合树。

通过 `object_property_add_child` 添加 child：
```
=> object_property_add           
	将 child 作为 obj 的属性，属性名name，类型为 "child<child的类名>"，同时getter为object_get_child_property，没有setter
=> child->parent = obj
```

### link

link实现了backlink关系，表示一个设备引用了另外一个设备，是一种松散的联系。两个设备之间能有多个link关系，可以进行修改。它完善了组合树，使其构成构成了一幅有向图。

通过 `object_property_add_link` 添加link：
```
=> 创建 LinkProperty ，填充目标(child)的信息
=> object_property_add            将 LinkProperty 作为 obj 的属性，属性名name，类型为 "link<child的类名>"，同时getter为 object_get_link_property 。如果传入了check函数，则需要回调，设置setter为 object_set_link_property
```
# 编译

在完成了此文件后，需要添加到 `Makefile.objs` 中，QEMU好知道编译并将其连接到二进制文件中。格式为：
```
common-obj-$(CONFIG_PCI_TESTDEV) += pci-testdev.o
```
或者
```
obj-$(CONFIG_PCI) += pci-testdev.o
```
其中，配置选项 `CONFIG_PCI_TESTDEV` 启用后，`pci-testdev.c` 才会被编译并链接到qemu模拟器中。 `CONFIG_PCI_TESTDEV` 在文件 `config-all-devices.mak` 中配置。


# 总结

QOM的对象构造分成三部分，第一部分是type的构造，这是通过TypeInfo构造一个TypeImpl的哈希表，这是在main之前完成的。
第二部分是class的构造，这是在main中进行的，这两部分都是全局的，也就是只要编译进去了的QOM对象都会调用。
第三部分是object的构造，这是构造具体的对象实例，在命令行指定了对应的设备时，才会创建object。


# 参考
1. [How to add a new device in QEMU source code?](https://stackoverflow.com/questions/28315265/how-to-add-a-new-device-in-qemu-source-code)
2. [QEMU中的对象模型——QOM（介绍篇）](https://blog.csdn.net/u011364612/article/details/53485856)
3. [QOM介绍](https://terenceli.github.io/%E6%8A%80%E6%9C%AF/2017/01/08/qom-introduction)
4. [[System] Emulate a PCI device with Qemu](http://tic-le-polard.blogspot.com/2015/01/emulate-pci-device-with-qemu.html)
5. [How to create a custom PCI device in QEMU](http://gougougebj.lofter.com/post/1d1f2c54_a59500e)
6. [qemu-object-model](https://github.com/Gyumeijie/qemu-object-model)
7. [Features/QOM](https://wiki.qemu.org/Features/QOM)
8. [Qemu之QOM（Qemu Object Module）](https://www.jianshu.com/p/20d4820f34de)
9. [QEMU学习笔记——QOM(Qemu Object Model)](https://www.binss.me/blog/qemu-note-of-qemu-object-model/)
10. [How to debug qemu devices](https://stfpeak.github.io/2017/07/15/how-to-debug-qemu-devices/)
11.  [QEMU漏洞挖掘](https://www.tuicool.com/articles/MzqYbia)
12. [Documentation/QOMConventions QOM编程规范](https://wiki.qemu.org/Documentation/QOMConventions)
13. [QEMU's instance_init() vs. realize()](http://people.redhat.com/~thuth/blog/qemu/2018/09/10/instance-init-realize.html)
14. [QEMU/KVM VM Execution Basics](https://dev.goodwu.net/post/2018-03-25-qemu-kvm-exec-basics/)
15. [QOM说明文档](https://lists.gnu.org/archive/html/qemu-devel/2012-08/msg02271.html)