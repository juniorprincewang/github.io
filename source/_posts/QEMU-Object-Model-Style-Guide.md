---
title: QEMU Object Model Style Guide
date: 2018-12-04 09:11:33
tags:
- QEMU
- QOM
categories:
- QEMU
---

本篇博客主要讲述了QOM编程的教程，翻译了[QEMU Object Model Style Guide](https://lists.gnu.org/archive/html/qemu-devel/2012-08/msg02271.html) 。
<!-- more ------------------------------------------------->

# 概述

本文档是QOM的分步教程。 这里面没有提供具体的API，代码中包含内联文档和API详细信息可在相应的头文件中找到。

# 动机
----------
QEMU广泛使用面向对象编程。 但是由于QEMU是用C语言编写的，因此这些OOP概念通常使用非常不同的机制来实现相同的目标。 
QOM的目标是为QEMU中的所有OOP使用单一基础架构，提高一致性，并且可以长期保持可维护性。

QOM提供了一个通用的基础设施：

 - 类型管理
    - 注册类型
    - 枚举注册类型
 - 继承
    - 单亲继承
    - 继承层次结构的内省
    - 通过无状态接口进行多重继承
 - 多态性
    - 基于类的多态性
    - 虚拟和纯虚拟方法
    - 构造函数/析构函数链接
 - 对象属性
    - 动态属性注册（绑定到对象）
    - 属性内省
    - 访问权限
    - 访问挂钩
 - 类型转换
    - 运行时检查向上转型/向下转型
    - 完全支持转型上下链（包括接口）
 - 对象枚举
    - 表达对象之间的关系
    - 能够使用符号路径引用对象
    - 表示为有向图

虽然QOM有很多高级概念，但主要的设计目标是使简单的概念易于实现。

# 创建简单类型

理解QOM最简单的方法就是举例子。 这是创建从Object派生的新类型作为父类的典型示例。 在此示例中，所有代码都将存在于单个C源文件中。

```
    #include "qemu/object.h"
```
这是包含核心QOM基础结构的头文件。 它具有最小的依赖性，便于单元测试。

```    
    #define TYPE_MY_TYPE "my-type"
    #define MY_TYPE(obj) OBJECT_CHECK(MyType, (obj), TYPE_MY_TYPE)
```

所有QOM类型都应至少定义两个宏。 
第一个宏是类型Type名称的符号版本。 它应该始终采用`TYPE_ + upper（typename）`的形式。 类型名称通常应遵循QAPI的命名规则，这意味着破折号` - `优先于下划线 `_`。
第二个宏是一个转换宏。 第一个参数是类型结构，其余参数是不言而喻的。 即使C文件当前未使用强制转换宏，也应始终遵循此形式。
```
    typedef struct MyType MyType;
    
    struct MyType
    {
        Object parent_obj;
    
        /*< private >*/
        int foo;
    };
```

在声明结构时，应使用前向声明。 由于在定义类时需要它，这对于一致性很有用。
第一个元素必须是父类型，并且应该命名为`parent_obj`或 `parent` 。 使用QOM类型时，应避免直接访问此成员，而是依赖于转换宏。

转换宏隐藏了实现中的继承层次结构。 这使得通过更改层次结构而不必改变许多地方的代码，可以更容易地重构代码。
```
    static TypeInfo my_type_info = {
        .name = TYPE_MY_TYPE,
        .parent = TYPE_OBJECT,
        .instance_size = sizeof(MyType),
    };
    
    static void register_types(void)
    {
        type_register_static(&my_type_info);
    }
    
    type_init(register_types);
```

必须在QOM基础结构中注册所有QOM类型。 注册后，用户可以枚举类型，创建对象以及与对象交互，而无需任何其他代码。

所有类型都必须设置 `name` 和 `parent` 参数。 `类型宏` 应始终用于这些参数。 几乎所有类型都应该设置 `instance_size` 参数，如果没有指定，它将从其父项继承。

最后，应提供模块初始化函数。 此处显示的命名约定应该在所有新代码中使用。

通常，一个C文件应该注册一种类型。 此规则有许多有效的例外，但只要有可能，类型应拆分为单独的C文件。


# 使用方法创建类型

与QOM的下一个最常见的交互是创建一个将从另一个类型Type继承的类型Type。 这通常涉及添加类并实现可以覆盖子类的虚方法。 以下差异显示了我们需要扩展前一个示例以允许使用多态继承的更改。
```
    @@ -1,10 +1,25 @@
    +#ifndef QEMU_MY_TYPE_H
    +#define QEMU_MY_TYPE_H
    +
     #include "qemu/object.h"
```
此示例假定初始声明将拆分为单独的文件头。 为了简化示例，保护用于显示头文件的开始和结束位置。

```
     #define TYPE_MY_TYPE "my-type"
     #define MY_TYPE(obj) \
         OBJECT_CHECK(MyType, (obj), TYPE_MY_TYPE)
    +#define MY_TYPE_CLASS(klass) \
    +    OBJECT_CLASS_CHECK(MyTypeClass, (klass), TYPE_MY_TYPE)
    +#define MY_TYPE_GET_CLASS(obj) \
    +    OBJECT_GET_CLASS(MyTypeClass, (obj), TYPE_MY_TYPE)
```
添加类时，我们需要在类型定义中再添加两个宏。
第一个宏是类转换宏。 这看起来非常类似于对象强制转换宏，而是将类作为参数。
我们添加的第二个宏允许用户从对象Object获取类Class指针。 方法调度需要最后一个宏。
```
     typedef struct MyType MyType;
    +typedef struct MyTypeClass MyTypeClass;
    +
    +struct MyTypeClass
    +{
    +    ObjectClass parent_klass;
    +
    +    void (*bar)(MyType *obj, int foo);
    +};
```
类看起来与对象非常相似，因为它表示为C结构，第一个成员必须是父类型的类。
通常，类只包含函数指针，但可以包含类的数据成员。 每个函数指针的第一个参数应始终是对象类型。
```  
     struct MyType
     {
    @@ -14,10 +29,35 @@ struct MyType
         int foo;
     };
     
    +void my_type_bar(MyType *obj, int foo);
    +
    +#endif
```
应该提供辅助函数来进行方法调度。 这提高了可读性和便利性。
```
    +
    +static void my_type_default_bar(MyType *obj, int foo)
    +{
    +    /* do nothing */
    +}
    +
    +void my_type_bar(MyType *obj, int foo)
    +{
    +    MyTypeClass *mc = MY_TYPE_GET_CLASS(obj);
    +
    +    mc->bar(obj, foo);
    +}
    +
    +static void my_type_class_init(ObjectClass *klass, void *data)
    +{
    +    MyTypeClass *mc = MY_TYPE_CLASS(klass);
    +
    +    mc->bar = my_type_default_bar;
    +}
    +
     static TypeInfo my_type_info = {
         .name = TYPE_MY_TYPE,
         .parent = TYPE_OBJECT,
         .instance_size = sizeof(MyType),
    +    .class_size = sizeof(MyTypeClass),
    +    .class_init = my_type_class_init,
     };
     
     static void register_types(void)
```
为了为类型添加新类，我们需要在 `TypeInfo` 中指定类的大小。 我们还需要提供一个初始化类的函数。 只为任何类型创建和初始化类一次，因此无论此类型创建了多少对象，都将调用此函数一次。

类初始化函数应遵循 `typename +'_ class_init'`的命名约定。 类初始化函数应该将 klass参数强制转换为适当的类型，然后适当地重载方法。
在这个例子中，我们将方法初始化为一个没有用处的虚函数。 这是因为'foo'是一个虚方法，这意味着如果基类不想覆盖行为，则不需要实现该函数。

如果我们没有初始化该方法，那么该函数将是一个纯虚方法，这意味着子类必须实现该函数。 QOM无法强制执行此要求，因此应在包装函数中小心检查NULL。

包装函数只是调度方法。 除了调度方法之外，它不应该实现任何逻辑或行为。 包装器函数可以检查NULL并返回错误或断言。

# 实现设备和重载方法

大多数QOM用户不会实现从 `TYPE_OBJECT` 派生的对象。 相反，通常QOM用户将从 `TYPE_DEVICE` 或其他一些基类派生，并且还必须实现虚拟方法。

在此示例中，我们将 `MyType` 更改为从 `TYPE_DEVICE` 继承，然后填充所需的纯虚方法。
```
    @@ -16,14 +16,14 @@ typedef struct MyTypeClass MyTypeClass;
     
     struct MyTypeClass
     {
    -    ObjectClass parent_klass;
    +    DeviceClass parent_klass;
     
         void (*bar)(MyType *obj, int foo);
     };
     
     struct MyType
     {
    -    Object parent_obj;
    +    DeviceState parent_obj;
     
         /*< private >*/
         int foo;
```
更改父类型是微不足道的，因为它只需要修改结构。 这是通过强制转换宏执行所有强制转换的好处之一。 它简化了重构过程。
```
    @@ -45,16 +45,27 @@ void my_type_bar(MyType *obj, int foo)
         mc->bar(obj, foo);
     }
     
    +static int my_type_realize(DeviceState *dev)
    +{
    +    MyType *my = MY_TYPE(dev);
    +
    +    my->foo = 1;
    +
    +    return 0;
    +}
    +
     static void my_type_class_init(ObjectClass *klass, void *data)
     {
         MyTypeClass *mc = MY_TYPE_CLASS(klass);
    +    DeviceClass *dc = DEVICE_CLASS(klass);
     
         mc->bar = my_type_default_bar;
    +    dc->init = my_type_realize;
     }
```
`TYPE_DEVICE` 有一个纯虚方法 `init`，这有点用词不当。 `init` 方法在构造之后但在guest虚拟机第一次启动之前调用。 在QOM命名法中，我们称之为 `realize`。 在某个时间点，`TYPE_DEVICE` 将被重构以重命名 `init` 方法以实现，但是现在，我们必须忍受这种不一致。
```
     static TypeInfo my_type_info = {
         .name = TYPE_MY_TYPE,
    -    .parent = TYPE_OBJECT,
    +    .parent = TYPE_DEVICE,
         .instance_size = sizeof(MyType),
         .class_size = sizeof(MyTypeClass),
         .class_init = my_type_class_init,
```


# 使用实例初始化

QDev要求通过 `init` 和 `exit` 方法进行所有初始化和破坏。 由于QDev没有构造函数和析构函数的概念，因此实现链接的类型通常以不一致的方式完成。

作为 `TypeInfo` 结构的一部分，QOM有一个 `instance_init` 和 `instance_finalize` 方法，它们分别充当构造函数和析构函数。 这些函数从子类开始调用，并通过QOM处理类型层次结构。

任何可以独立于用户提供的状态进行初始化的状态都应该初始化为构造函数的一部分。
```
    @@ -33,6 +33,13 @@ void my_type_bar(MyType *obj, int foo);
     
     #endif
     
    +static void my_type_initfn(Object *obj)
    +{
    +    MyType *my = MY_TYPE(obj);
    +
    +    my->foo = 1;
    +}
    +
     static void my_type_default_bar(MyType *obj, int foo)
     {
         /* do nothing */
    @@ -47,10 +54,6 @@ void my_type_bar(MyType *obj, int foo)
     
     static int my_type_realize(DeviceState *dev)
     {
    -    MyType *my = MY_TYPE(dev);
    -
    -    my->foo = 1;
    -
         return 0;
     }
     
    @@ -69,6 +72,7 @@ static TypeInfo my_type_info = {
         .instance_size = sizeof(MyType),
         .class_size = sizeof(MyTypeClass),
         .class_init = my_type_class_init,
    +    .instance_init = my_type_initfn,
     };
     
     static void register_types(void)
```
由于 `foo` 可以在不依赖于用户提供的状态的情况下进行初始化，因此我们可以将该逻辑完全移至构造函数。 不幸的是，`DeviceState` init函数必须保留，因为它是纯虚拟的，但它现在是微不足道的。

# 用户提供的状态（属性）

QEMU中大多数对象的共同特性是希望允许用户在初始创建期间或在运行时调整对象的参数。 属性Properties提供了执行此操作的通用框架。

属性丰富而复杂，这里不会详尽介绍。 有关详尽的文档，请参阅 `qemu/object.h` 头文件中的文档。

Most interactions with properties will happen through convenience functions that make adding properties easier for typical users.  In the case of our example, we'll add properties using the qdev static property interface.
大多数与属性的交互都是通过便利函数实现的，这使得典型用户更容易添加属性。 在我们的示例中，我们将使用qdev静态属性接口添加属性。
```
    @@ -27,6 +27,7 @@ struct MyType
     
         /*< private >*/
         int foo;
    +    int max_vectors;
     };
```
对于静态属性，属性就对应于对象结构的成员。 qdev中的基础结构可以在调用DeviceState::init()之前随时自动更改此成员的值。 这意味着任何依赖于作为属性的成员的初始化必须在DeviceState::init方法中完成。
```
     void my_type_bar(MyType *obj, int foo);
    @@ -54,9 +55,20 @@ void my_type_bar(MyType *obj, int foo)
     
     static int my_type_realize(DeviceState *dev)
     {
    +    MyType *mt = MY_TYPE(dev);
    +
    +    if (mt->max_vectors > 100) {
    +        return -EINVAL;
    +    }
    +
         return 0;
     }
```
对于此示例，我们只是验证包含一个合理的值的属性并没有实现。
```
    +static Property my_type_properties[] = {
    +    DEFINE_PROP_INT("max-vectors", MyType, max_vectors, 0),
    +    DEFINE_PROP_END_OF_LIST(),
    +};
    +
     static void my_type_class_init(ObjectClass *klass, void *data)
     {
         MyTypeClass *mc = MY_TYPE_CLASS(klass);
    @@ -64,6 +76,7 @@ static void my_type_class_init(ObjectClass *klass,
     
         mc->bar = my_type_default_bar;
         dc->init = my_type_realize;
    +    dc->props = my_type_properties;
     }
     
     static TypeInfo my_type_info = {
```
使用 `TYPE_DEVICE` 类中的静态类变量注册静态属性。基类可以使用此方法添加静态属性，子类将自动继承它们。

# 子属性和链接属性

QOM中其他常见类型的属性是子属性和链接属性。 与静态属性一样，有一些特殊的帮助器可以将这些属性添加到对象中。
```
    @@ -25,6 +25,9 @@ struct MyType
     {
         DeviceState parent_obj;
     
    +    Pin *in;
    +    Pin out;
    +
         /*< private >*/
         int foo;
         int max_vectors;
```

首先，我们必须在该对象中添加将保存属性的struct成员。 链接是指向另一个对象的指针，在C中表示为指针。子属性是嵌入对象，通过在对象结构中嵌入struct成员来表示。

子属性的生命周期与父对象相关联。 IOW，当MyType的一个对象被销毁时，嵌入其中的'out'对象将被自动销毁。

链接属性将保存对其指向的对象的引用，但不控制它指向的对象的生命周期。 也就是说，当MyType的一个对象被破坏时，'in'指向的对象不一定会被破坏，尽管它的引用计数会减少。
```
    @@ -39,6 +42,11 @@ static void my_type_initfn(Object *obj)
         MyType *my = MY_TYPE(obj);
     
         my->foo = 1;
    +
    +    object_initialize(&my->out, TYPE_PIN);
    +    object_property_add_child(obj, "out", OBJECT(&my->out), NULL);
    +
    +    object_property_add_link(obj, "in", TYPE_PIN,
    +                             (Object **)&my->in, NULL);
     }
     
     static void my_type_default_bar(MyType *obj, int foo)
```
要将属性添加到对象，我们需要首先初始化子对象，然后添加属性。 这应该始终在构造函数中完成。
