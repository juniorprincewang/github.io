---
title: 编写Linux内核模块
date: 2018-11-16 19:37:38
tags:
- kernel
categories:
- linux
---
Linux 内核模块是设备和用户应用程序之间的桥梁，可以通过标准系统调用，为应用程序屏蔽设备细节。本篇文章就记录下写内核模块需要注意的点。
<!-- more -->

在编写Linux内核模块（驱动）这个过程中，应该掌握如下一些知识：

1. 驱动开发人员应该有良好的C语言基础，并能灵活地应用C语言的结构体、指针、宏等基本语言结构。
另外，Linux系统使用的C编译器是GNU C编译器，所以对GNU C标准的C语言也应该有所了解。

2. 驱动开发人员应该有良好的硬件基础。虽然不要求驱动开发人员具有设计电路的能力，但也应该对芯片手册上描述的接口设备有清楚的认识。
常用的设备有SRAM、Flash、UART、IIC和USB等。

3. 驱动开发人员应该对Linux内核源代码有初步的了解。例如一些重要的数据结构和函数等。

4. 驱动开发人员应该有多任务程序设计的能力，同时驱动中也会使用大量的自旋锁、互斥锁和信号量等。

# 内核模块和应用程序区别
内核模块不是应用程序，从一开始就没有 `main()` 函数。内核模块和普通应用程序的区别有： 

+ 非顺序执行：
		内核模块使用初始化函数将自身注册并处理请求，初始化函数运行后就结束了。
		内核模块处理的请求在模块代码中定义。这和常用于图形用户界面（graphical-user interface，GUI）应用的事件驱动编程模型比较类似。 
+ 没有自动清理：
		任何由内核模块申请的内存，必须要模块卸载时手动释放，否则这些内存将无法使用，直到系统重启。 
+ 不要使用 printf() 函数：
		内核代码无法访问为 Linux 用户空间编写的库。内核模块运行在内核空间，它有自己独立的地址空间。内核空间和用户空间的接口被清晰的定义和控制。
		内核模块可以通过 printk() 函数输出信息，这些输出可以在用户空间查看到。 
+ 会被中断：
		内核模块一个概念上困难的地方在于他们可能会同时被多个程序 / 进程使用。构建内核模块时需要小心，以确保在发生中断的时候行为一致和正确。
+ 更高级的执行特权：
		通常内核模块会比用户空间程序分配更多的 CPU 周期。这看上去是一个优势，然而需要特别注意内核模块不会影响到系统的综合性能。
+ 无浮点支持：
		对用户空间应用，内核代码使用陷阱（trap）来实现整数到浮点模式的转换。然而在内核空间中这些陷阱难以使用。
		替代方案是手工保存和恢复浮点运算，这是最好的避免方式，并将处理留给用户空间代码。

# 内核的并发
内核编程中有几个并发的来源。 
1. 自然的, Linux 系统运行多个进程, 在同一时间, 不止一个进程能够试图使用你的驱动。
2. 大部分设备能够中断处理器; 中断处理异步运行, 并且可能在你的驱动试图做其他事情的同一时间被调用。
3. 在对称多处理器系统( SMP )上运行, 驱动可能在多个 CPU 上并发执行。

# 内核模块编程

这里使用 Derek Molloy 的 `hello.c` 编程代码来学习。 编写一个内核最最基本的框架，需要引用的头文件和函数。

```
#include <linux/init.h>             // 用于标记函数的宏，如 __init、__exit
#include <linux/module.h>           // 加载内核模块到内核使用的核心头文件 
#include <linux/kernel.h>           // 包含内核使用的类型、宏和函数 

MODULE_LICENSE("GPL");              ///< 许可类型，它会影响到运行时行为 
MODULE_AUTHOR("Derek Molloy");      ///< 作者，当使用 modinfo 命令时可见 
MODULE_DESCRIPTION("A simple Linux driver for the BBB.");  ///< 模块描述，参见 modinfo 命令 
MODULE_VERSION("0.1");              ///< 模块版本 

static char *name = "world";        ///< 可加载内核模块参数示例，这里默认值设置为“world”
module_param(name, charp, S_IRUGO); ///< 参数描述。charp 表示字符指针（char ptr），S_IRUGO 表示该参数只读，无法修改 
MODULE_PARM_DESC(name, "The name to display in /var/log/kern.log");  ///< 参数描述 

/** @brief 可加载内核模块初始化函数 
 *  static 关键字限制了该函数的可见范围为当前 C 文件。
 *  __init 宏表示对于内置驱动（不是可加载内核模块），该函数只在初始化的时候执行，
 *  在此之后，该函数可以废弃，且内存可以被回收。
 *  @return 当执行成功返回 0
 */
static int __init helloBBB_init(void){
   printk(KERN_INFO "EBB: Hello %s from the BBB LKM!\n", name);
   return 0;
}

/** @brief 可加载内核模块清理函数 
 *  和初始化函数类似，它是静态（static）的。__exit 函数表示如果这个代码是给内置驱动（非可加载内核模块）使用，该方法是不需要的。 
 */
static void __exit helloBBB_exit(void){
   printk(KERN_INFO "EBB: Goodbye %s from the BBB LKM!\n", name);
}

/** @brief 内核模块必须使用 linux/init.h 头文件提供的 module_init() 和 module_exit() 宏，
 *  它们标识了在模块插入时的初始化函数和移除时的清理函数（如上描述）
 */
module_init(helloBBB_init);
module_exit(helloBBB_exit);
```

+ 第 5 行：语句 `MODULE_LICENSE("GPL")` 提供了（通过 `modinfo` ）该模块的许可条款，这让使用这个内核模块的用户能够确保在使用自由软件。由于内核是基于 GPL 发布的，许可的选择会影响内核处理模块的方式。如果对于非 GPL 代码选择“专有”许可，内核将会把模块标记为“污染的（tainted）”，并且显示警告。对 GPL 有非污染（non-tainted）的替代品，比如“GPL 版本 2”、“GPL 和附加权利”、“BSD/GPL 双许可”、“MIT/GPL 双许可”和“MPL/GPL 双许可”。更多内容可以查看 `linux/module.h` 头文件。 
+ 第 10 行：名字（字符类型指针）被声明为静态，并且被初始化包含字符串“hello”。 `在内核模块中应该避免使用全局变量`，这比在应用程序编程时更加重要，因为全局变量被整个内核共享。应该使用 `static` 关键字来限制变量在模块中的作用域。如果必须使用全局变量，在变量名上增加前缀确保在模块中是唯一的。 
+ 第 11 行： `module_param(name, type, permissions)` 宏有三个参数，`名字`（展示给用户的参数名和模块中的变量名）、 `类型`（参数类型，即 byte、int、uint、long、ulong、short、ushort、bool、逆布尔 invbool 或字符指针之一）和 `权限`（这是当使用 sysfs 时对参数的访问权限。值 0 禁用该项，而值为 `S_IRUGO` 运行用户/组/其他有读权限，参阅访问权限模式位指南）。 
+ 第 20 和 28 行：函数可以是任何名字（如 `helloBBB_init()` 和 `helloBBB_exit()` ），但是必须向 `module_init()` 和 `module_exit()` 宏传入相同的名字，如第 35 和 36 行。
+ 第 21 行： `printk()` 和 `printf()` 行数的使用方式类似，可以在内核模块代码的任何地方调用该函数。唯一重要却别是当调用 `printk()` 函数时，必须提供日志级别。日志级别在 `linux/kern_levels.h` 头文件中定义，它的值为 KERN_EMERG、KERN_ALERT、KERN_CRIT、KERN_ERR、KERN_WARNING、KERN_NOTICE、KERN_INFO、KERN_DEBUG 和 KERN_DEFAULT 之一。该头文件通过 `linux/printk.h` 文件被包含在 `linux/kernel.h` 头文件中。

# 编译模块

构建内核模块需要 Makefile 文件，事实上是一个特殊的 `kbuild Makefile` 。 构建本文示例的内核模块所需要的 `kbuild Makefile` 文件参见下面代码。
详细的内核编译指南，参见内核源码的 [`Document/kbuild` 目录](https://www.kernel.org/doc/Documentation/kbuild/)下发现的文件。

构建 Hello World 可加载内核模块需要的 Makefile 文件 
```
obj-m+=hello.o

all:
    make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
    make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

Makefile 文件第一行被成为目标定义，它定义了需要构建的模块（hello.o）。这条命令表明有一个模块要
从目标文件 hello.o 建立， 在从目标文件建立后结果模块命名为 hello.ko。它的语法惊人的复杂，例如 `obj-m` 定义了`可加载模块目标` ，`obj-y` 表示内置的对象目标。
当模块需要从多个目标文件构建时，语法会变得更加复杂。 
如果你有一个模块名为 module.ko, 是来自 2 个源文件( 姑且称之为, file1.c 和 file2.c ), 正确的书写应当是:
```
obj-m := module.o
module-objs := file1.o file2.o
```
Makefile 文件中需要提醒的内容和普通 Makefile 文件类似。 `$(shell uname -r) 命令返回当前内核构建版本`，这确保了一定程度的可移植性。 `-C` 选项在执行任何 make 任务前将目录切换到内核目录，它在那里会发现内核的顶层 makefile。 `M=$(PWD)` 变量赋值告诉 make 命令实际工程文件存放位置，在试图建立模块 (modules) 目标前，回到你的模块源码目录，而此目标会在 ` obj-m` 变量里面找模块列表。对于外部内核模块来说，modules 目标是默认目标。另一种目标是 modules_install，它将安装模块（make命令必须使用超级用户权限执行且需要提供模块安装路径）。 

# 加载和卸载模块

通过 `insmod` 命令将模块插入内核，通过 `rmmod` 命令删除模块。要查询内核中当前的模块，使用 `lsmod` 命令。因为模块可以依赖于其他模块，所以可以用 `depmod` 命令构建一个依赖项文件。要在模块之前自动加载依赖模块，可以使用 `modprobe` 命令（ `insmod` 的包装器）。最后，您可以使用 `modinfo` 命令读取 LKM 的模块信息 。

`insmod` 命令和 `modprobe` 都可以动态加载驱动模块。不过 `modprobe` 可以解决加载模块时的依赖关系，它是通过 `/lib/modules/$(shell uname -r)/modules.dep(.bb)` 文件来查找依赖关系的；而 `insmod` 不能解决依赖问题，但是 `insmod` 可以在任何目录下执行。 如果要加载的驱动模块还依赖其他ko驱动模块的话，就只能将模块拷贝到上述的特定目录，`depmod` 后再 `modprobe`。
还有一点需要注意的是 `insmod` 加载模块需要后缀名 `.ko` ，而 `modprobe` 的模块名称不需要后缀名。
+ `insmod` 动态加载 hello 模块。
``` 
insmod hello.ko
```
+ `modprobe` 动态加载 hello 模块。 
`modprobe` 会读取驱动模块安装目录下的modules.dep文件，从而分析出各个模块的依赖性的。因此，在 `depmod` 后再去执行 `modprobe hello` 。
其中 `depmod` 会在 `/lib/modules/$(shell uname -r)/`目录下生成 `modules.dep` 和 `modules.dep.bb` 文件，表明模块的依赖关系。
```
cp hello.ko  /lib/modules/$(shell uname -r)/
depmod 
modprobe hello
```

+ `lsmod` 查看加载的驱动模块列表

```
lsmod | grep hello
Module                  Size  Used by
hello                  16384  0
```
除了 `lsmod` 命令可以查看，还可以直接查看文件系统。
```
cat /proc/modules | grep hello
hello 16384 0 - Live 0xffffffffc1288000 (OE)
```
这里查看到的信息和lsmod命令提供的相同，但是它同时提供了**已加载模块在当前内核内存中的偏移量**，这个数据在调试时非常有用。 

可加载内核模块在 `/sys/module` 目录下也有目录项，它提供了用户直接访问自定义参数状态的方式。
```
root@ubuntu:~# ll /sys/module/hello/
总用量 0
drwxr-xr-x   5 root root    0 11月 19 19:14 ./
drwxr-xr-x 184 root root    0 11月 19 19:14 ../
-r--r--r--   1 root root 4096 11月 19 19:10 coresize
drwxr-xr-x   2 root root    0 11月 19 19:10 holders/
-r--r--r--   1 root root 4096 11月 19 19:15 initsize
-r--r--r--   1 root root 4096 11月 19 19:15 initstate
drwxr-xr-x   2 root root    0 11月 19 19:15 notes/
-r--r--r--   1 root root 4096 11月 19 19:10 refcnt
drwxr-xr-x   2 root root    0 11月 19 19:15 sections/
-r--r--r--   1 root root 4096 11月 19 19:15 srcversion
-r--r--r--   1 root root 4096 11月 19 19:15 taint
--w-------   1 root root 4096 11月 19 19:10 uevent
root@ubuntu:~# cat /sys/module/hello/taint 
OE

```

+ `rmmod` 卸载模块驱动
```
rmmod hello
rmmod hello.ko
```
此处删除的是模块名称，可以是 lsmod显示的模块名称，也可以是对应的ko文件名。

当然还可以使用 `modprobe` 的 `-r` 选项。
```
modprobe -r hello           # 注意这里无需输入.ko后缀
depmod                      # 更新modules.dep和modules.dep.bb文件，记录模块的依赖关系
```


+ `modinfo` 获得模块信息 

通过 `modinfo` 命令，可以获得模块的信息，这个命令能够识别出模块的描述、作者和定义的任何模块参数：
```
root#ubuntu:~# modinfo hello.ko 
filename:       /home/ubuntu/hello.ko
license:        Dual BSD/GPL
srcversion:     31FE72DA6A560C890FF9B3F
depends:        
retpoline:      Y
vermagic:       4.4.0-139-generic SMP mod_unload modversions retpoline 
```

# 宏 EXPORT_SYMBOL

Linux-2.4之前，默认的非static 函数和变量都会自动导入到kernel 空间， 而Linux-2.6之后默认不导出所有的符号，所以使用 `EXPORT_SYMBOL()` 做标记。

## EXPORT_SYMBOL宏的作用

`EXPORT_SYMBOL` 标签内定义的函数或者符号对全部内核代码公开，不用修改内核代码就可以在内核模块中直接调用。
即使用 `EXPORT_SYMBOL` 可以将一个函数以符号的方式导出给其他模块使用。
符号的意思就是函数的入口地址，或者说是把这些符号和对应的地址保存起来的，在内核运行的过程中，可以找到这些符号对应的地址的。

这里要和System.map做一下对比：
System.map 中的是连接时的函数地址。连接完成以后，在2.6内核运行过程中，是不知道哪个符号在哪个地址的。
EXPORT_SYMBOL 的符号， 是把这些符号和对应的地址保存起来，在内核运行的过程中，可以找到这些符号对应的地址。
在模块加载中，其本质就是动态链接到内核。
如果在模块中引用了内核或其它模块的符号，就要 `EXPORT_SYMBOL` 这些符号，这样才能找到对应的地址连接。

## EXPORT_SYMBOL使用方法
	1.在模块函数定义之后使用 `EXPORT_SYMBOL(函数名)`
	2.在调用该函数的模块中使用 `extern` 对要使用的符号或者函数进行声明
	3.首先加载定义该函数的模块，再加载调用该函数的模块

## EXPORT_SYMBOL示范
比如有两个驱动模块：Module A和Module B，其中Module B使用了Module A中的export的函数，因此在Module B的Makefile文件中必须添加：
```
KBUILD_EXTRA_SYMBOLS += /path/to/ModuleA/Module.symvers
export KBUILD_EXTRA_SYMBOLS
```

这样在编译Module B时，才不会出现Warning，提示说func1这个符号找不到，而导致编译得到的ko加载时也会出错。
```
// Module A (mod_a.c)
#include<linux/init.h>
#include<linux/module.h>
#include<linux/kernel.h>
 
static int func1(void)
{
       printk("In Func: %s...\n",__func__);
       return 0;
}
 
// Export symbol func1
EXPORT_SYMBOL(func1);
 
static int __init hello_init(void)
{
       printk("Module 1，say hello world!\n");
       return 0;
}
 
static void __exit hello_exit(void)
{
       printk("Module 1,Exit!\n");
}
 
module_init(hello_init);
module_exit(hello_exit);

```

```
// Module B (mod_b.c)
#include<linux/init.h>
#include<linux/kernel.h>
#include<linux/module.h>
extern int functl(void);
static int func2(void)
{
       func1();
       printk("In Func: %s...\n",__func__);
       return 0;
}
 
static int __init hello_init(void)
{
       printk("Module 2,is used Module 1 function!\n");
       func2();
       return 0;
}
 
static void __exit hello_exit(void)
{
       printk("Module 2,Exit!\n");
}
 
module_init(hello_init);
module_exit(hello_exit);

```
在驱动加载的时候，一定要先加载定义function1的Module A模块，然后再加载调用function1的Module B的驱动模块。
```
insmod Module_A.ko
insmod Module_B.ko

```



[Linux内核—EXPORT_SYMBOL宏的使用](https://blog.csdn.net/zengxianyang/article/details/50611828)

# 参考
1. [Writing a Linux Kernel Module — Part 1: Introduction](http://derekmolloy.ie/writing-a-linux-kernel-module-part-1-introduction/)
2. [编写Linux内核模块——第一部分：前言](http://www.infoq.com/cn/articles/linux-kernel-module-part01)
3. [Linux 可加载内核模块剖析](https://www.ibm.com/developerworks/cn/linux/l-lkm/index.html#artrelatedtopics)