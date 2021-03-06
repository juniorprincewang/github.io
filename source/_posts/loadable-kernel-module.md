---
title: 可加载内核模块编程
date: 2018-03-22 19:30:24
tags:
- kernel
- LKM
categories:
- [linux,kernel]
---

由于自己要动手修改virtio源码，需要重新编写位于客户机前端virtio驱动，因此需要了解可加载内核模块（loadable kernel module, LKM）。在网上和书里面找了些资料，总结一下。
<!-- more -->

Linux众多优良特性之一就是可以在运行时扩展内核的功能。而每块可以在运行时添加到（删除）内核的代码称为一个模块。可加载的内核模块包括设备驱动程序。这样使内核可以在不知道硬件如何工作的情况下和硬件进行交互。每个模块由目标代码组成（没有连接成一个完整可执行文件），可以动态连接到运行中的内核中。


## 重要数据结构

### 文件操作

`struct file_operations`结构或者其一个指针`fops`是可以将一个字符驱动连接到有编号得设备上。位于[`<linux/fs.h>`](https://elixir.bootlin.com/linux/latest/source/include/linux/fs.h#L1730)中。结构中得成员大部分负责系统调用实现。
```
struct file_operations {
  struct module *owner;
  loff_t (*llseek) (struct file *, loff_t, int);
  ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
  ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
  __poll_t (*poll) (struct file *, struct poll_table_struct *);
  long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
  long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
  int (*mmap) (struct file *, struct vm_area_struct *);
  int (*open) (struct inode *, struct file *);
  int (*flush) (struct file *, fl_owner_t id);
  int (*release) (struct inode *, struct file *);
  int (*lock) (struct file *, int, struct file_lock *);
};
```
一般的调用方式为:
```
static const struct file_operations fops = {
  .owner = THIS_MODULE,
  .llseek = llseek,
  .open = open,
  .read = read,
  .release = release,
  .write = write,
};
```
这个声明使用标准的C**标记式结构初始化**语法，这个语法是内核首选的，因为它使驱动在结构定义的改变之间更加可移植, 并且, 标记式初始化允许结构成员重新排序；在某种情况下，通过安放经常使用的成员的指针在相同硬件高速存储行中，提高了性能。

### 文件结构

`struct file`或其指针`filp`定义于`<linux/fs.h>`中，位于内核结构，不出现在用户程序中。文件结构代表一个打开的文件（它不特指设备驱动），由内核在open时创建，并传递给文件操作的任何函数，直到最后关闭，内核释放这个数据结构。

成员函数：
```
void *private_data;
```
open 系统调用设置这个指针为 NULL, 在为驱动调用 open 方法之前。 你可自由使用这个成员或者忽略它；你可以使用这个成员来指向分配的数据，但是接着你必须记住在内核销毁文件结构之前，在 release 方法中释放那个内存。 private_data是一个有用的资源，在系统调用间保留状态信息， 我们大部分例子模块都使用它.


### inode结构

`inode`结构由内核在内部用来表示文件，代表磁盘上的一个文件。`inode`不同于文件描述符的`struct file`文件结构，可能有代表单个文件的多个打开描述符的许多文件结构，但是它们都指向一个单个
inode 结构。

inode 结构包含大量关于文件的信息。作为一个通用的规则，这个结构只有 2 个成员对于编写驱动代码有用：
`dev_t i_rdev`：表示设备文件的节点，这个成员包含实际的设备号。
`struct cdev *i_cdev`：`struct cdev`是内核的内部结构，代表字符设备；这个成员包含一个指针，当节点指向是一个字符设备文件时，此域为指向这个inode结构的指针。

## 字符设备注册

内核在内部使用类型`struct cdev`的结构体代表字符设备，位于`<linux/cdev.h>`中。
```
struct cdev {   
    struct kobject kobj;                  //内嵌的内核对象kobject
    struct module *owner;                 //该字符设备所在的内核模块的对象指针
    const struct file_operations *ops;    //指向设备驱动程序文件操作表的指针  
    struct list_head list;                //用来将已经向内核注册的所有字符设备形成链表
    dev_t dev;                            //字符设备的设备号，由主设备号和次设备号构成
    unsigned int count;                   //隶属于同一主设备号的次设备号的个数
}; 
```

用下面的代码来初始化。
```
struct cdev *my_cdev = cdev_alloc();
my_cdev->ops = &my_fops;
```

`cdev_init`将struct cdev类型的结构体变量和file_operations结构体进行绑定。  
```
void cdev_init(struct cdev *, const struct file_operations *);
```

`cdev_alloc()`函数的功能是动态地分配cdev描述符并初始化kobject数组结构，在引用计数器变0时会自动释放该描述符。  
```
struct cdev *cdev_alloc(void);
```

`cdev_add()`函数功能是在设备驱动程序中注册一个cdev描述符。  
```
int cdev_add(struct cdev *, dev_t, unsigned);
```

`cdev_del()`删除cdev对象。  
```
void cdev_del(struct cdev *);
```

新设备的驱动程序采用的分配方法为：
```
int register_chrdev_region(dev_t from, unsigned count, const char *name)； 
//静态申请
```
+ from :要分配的设备编号范围的初始值, 这组连续设备号的起始设备号, 相当于register_chrdev() 中主设备号。
+ count: 连续编号范围，这组设备号的大小（也是次设备号的个数）
+ name: 编号相关联的设备名称. (/proc/devices); 本组设备的驱动名称

内核动态分配设备号：
```
int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, const char *name)；
```

+ dev：这个函数的第一个参数，是输出型参数，获得一个分配到的设备号。可以用MAJOR宏和MINOR宏，将主设备号和次设备号，提取打印出来，看是自动分配的是多少，方便我们在mknod创建设备文件时用到主设备号和次设备号。 例如 mknod /dev/xxx c 主设备号 次设备号
+ baseminor：次设备号的基准，从第几个次设备号开始分配。
+ count：次设备号的个数。
+ name: 驱动的名字。
+ 返回值：小于0，则错误，自动分配设备号错误。否则分配得到的设备号就被第一个参数带出来。


上述两种方法可以为驱动程序分配任意范围的设备号。

字符设备经典的注册方法是：
```
int register_chrdev(unsigned int major, const char *name, struct file_operations *fops);
```
这里`major`是感兴趣的主编号，`name`是驱动的名字（出现在`/proc/devices`），`fops`是缺省的`file_operations`结构。  
与注册配对的去除设备的方法是：  
```
int unregister_chrdev(unsigned int major, const char *name);
```
从主次编号来建立 `dev_t` 数据项的宏定义。  
```
dev_t MKDEV(unsigned int major, unsigned int minor);
```

**sample code**  

```

static struct cdev my_cdev[N_MINOR];    /* char device abstraction */
static struct class *my_class;          /* linux device model */
static int __init my_init(void)
{
    int i;
    dev_t curr_dev;
    dev_t dev_num;

    /* obtain major */
    dev_num = MKDEV(DRIVER_MAJOR, 0);

    /* Request the kernel for N_MINOR devices */
    alloc_chrdev_region(&dev_num, 0, N_MINORS, "my_driver");

    /* Create a class : appears at /sys/class */
    my_class = class_create(THIS_MODULE, "my_driver_class");

    /* Initialize and create each of the device(cdev) */
    for (i = 0; i < N_MINORS; i++) {

        /* Associate the cdev with a set of file_operations */
        cdev_init(&my_cdev[i], &fops);
        my_cdev[i].owner = THIS_MODULE;

        /* Build up the current device number. To be used further */
        curr_dev = MKDEV(MAJOR(dev_num), MINOR(dev_num) + i);

        /* Create a device node for this device. Look, the class is
         * being used here. The same class is associated with N_MINOR
         * devices. Once the function returns, device nodes will be
         * created as /dev/my_dev0, /dev/my_dev1,... You can also view
         * the devices under /sys/class/my_driver_class.
         */
        device_create(my_class, NULL, curr_dev, NULL, "my_dev%d", i);

        /* Now make the device live for the users to access */
        cdev_add(&my_cdev[i], curr_dev, 1); 
    }

    return 0;
}
```

### open函数

```
int (*open)(struct inode *inode, struct file *filp);
```

在大部分驱动中, open 应当 进行下面的工作:  
+ 检查设备特定的错误（例如设备没准备好, 或者类似的硬件错误）。
+ 如果它第一次打开, 初始化设备。  
+ 如果需要, 更新 f_op 指针。  
+ 分配并填充要放进 filp->private_data 的任何数据结构。  

### release函数

```
int (*release)(struct inode *inode, struct file *filp)
```

不是每个 `close` 系统调用引起调用 `release` 方法。 只有真正释放设备数据结 构的调用会调用这个方法。  
内核维持一个文件结构被使用多少次的计数。   
`fork` 和 `dup` 都不创建新文件(只有 `open` 这样); 它们只递增存在的结构中的计数。  
`close` 系统调用仅在文件结构计数掉到 `0` 时执行 `release` 方法， 这在结构被销毁时发生。   
`release` 方法和 `close` 系统调用之间的这种关系保证了驱动一次 `open` 只看到一次 `release` 。

### read/write函数

```
ssize_t read(struct file *filp, char __user *buff, size_t count, loff_t *offp);
ssize_t write(struct file *filp, const char __user *buff, size_t count, loff_t *offp);
```

filp 是文件指针，count是请求的传输数据大小。  
buff 参数指向持有被写入数据的缓存，或者放入新数据的空缓存。  
最后，offp 是一个指针指向一个 *long offset type* 对象，它指出用户正在存取的文件位置。返回值是一个 *signed size type*。 

### llseek
```
loff_t (*llseek) (struct file *, loff_t, int); 
```
`llseek` 方法用作改变文件中的当前读/写位置, 并且新位置作为(正的)返回值。 `loff_t` 参数是一个 `long offset`, 并且就算在 32 位平台上也至少 64 位宽。 若发生错误，返回一个负值。 如果这个函数指针是 `NULL`, `llseek` 调用会以潜在地无法预知的方式修改 file 结构中的位置计数器。

### ioctl

通过不同命令来对硬件进行控制。
函数原型：
```
long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
```
用户空间的 `ioctl` 系统调用的原型如下：

```
int ioctl(int d, int request, ...);
```
[大多数ioctl的实现都包含了一个switch语句来根据cmd参数选择对应的操作](http://www.linuxtcpipstack.com/820.html)



下面函数可以在用户空间和内核空间拷贝数据。
```
unsigned long copy_to_user(void __user \*to,const void \*from, unsigned long count);
unsigned long copy_from_user(void \*to, const void __user \*from, unsigned long count);

```
位于`<asm/uaccess.h>`。至于实际的设备方法, `read` 方法的任务是从设备拷贝数据到用户空间(使用 `copy_to_user`), 而 `write` 法必须从用户空间拷贝数据到设备(使用 `copy_from_user` )。

```
get_user(x, ptr)
put_user(x, ptr)
```
`x`是内核空间的变量，`ptr`是用户空间的指针。上述两个函数主要用于内核空间和用户空间完成一些简单类型变量（char、int、long等）的拷贝任务，对于一些复合类型的变量，比如数据结构或者数组类型，`get_user`和`put_user`函数还是无法胜任，这两个函数内部将对指针指向的对象长度进行检查。



## 调试

### printk
`printk` 允许你根据消息的严重程度对其分类，通过附加不同的记录级别或者优先级在消息上。常常用一个宏定义来指示记录级别。
比如
```
printk(KERN_INFO "Hello!\n");
```
记录宏定义扩展成一个字串, 在编译时与消息文本连接在一起;这就是为什么下面的在优先级和格式串之间没有逗号的原因。
有8种可能的记录等级。在`<linux/kernel.h>`里定义。按照严重等级递减顺序依次是：
```
KERN_EMERG
	用于紧急消息, 常常是那些崩溃前的消息.
KERN_ALERT
	需要立刻动作的情形.
KERN_CRIT
	严重情况, 常常与严重的硬件或者软件失效有关.
KERN_ERR
	用来报告错误情况; 设备驱动常常使用 KERN_ERR 来报告硬件故障.
KERN_WARNING
	有问题的情况的警告, 这些情况自己不会引起系统的严重问题.
KERN_NOTICE
	正常情况, 但是仍然值得注意. 在这个级别一些安全相关的情况会报告.
KERN_INFO
	信息型消息。在这个级别, 很多驱动在启动时打印它们发现的硬件的信息.
KERN_DEBUG
	用作调试消息。
```

整数范围0~7，越小表示优先级越高。这里面读取的方式有所不同。基于记录级别，内核可能打印消息到当前控制台，可能是一个文本模式终端，串口，或者是一台并口打印机，如果优先级小于整型值 console_loglevel，消息被递交给控制台，一次一行（除非提供一个新行结尾，否则什么都不发送）。
如果klogd 和 syslogd 都在系统中运行， 内核消息被追加到 /var/log/messages （或者另外根据你的 syslogd 配置处理），独立于 console_loglevel，如果 klogd 没有运行，你只有读 /proc/kmsg （用`dmsg` 命令最易做到 ）将消息取到用户空间。
当使用 klogd 时，你应当记住，它不会保存连续的同样的行；它只保留第一个这样的行，随后是，它收到的重复行数。
也可以通过文本文件 `/proc/sys/kernel/printk` 读写控制台记录级别。
```
$ cat /proc/sys/kernel/printk
4	4	1	7
```
这个文件有 4 个整型值: 当前记录级别4，适用没有明确记录级别的消息的缺省级别4，允许的最小记录级别1，以及启动时缺省记录级别7。 

### debugfs

`debugfs`是一种用于内核调试的虚拟文件系统，内核开发者通过debugfs和用户空间交换数据。

默认情况下，debugfs会被挂载在目录`/sys/kernel/debug` 之下，如果发行版里没有自动挂载，可以用如下命令手动完成。
```
mount -t debugfs none /your/debugfs/dir
```

创建和撤销目录及文件
```
struct dentry *debugfs_create_dir(const char *name, struct dentry *parent);
struct dentry *debugfs_create_file(const char *name, mode_t mode, 
        struct dentry *parent, void *data, 
        const struct file_operations *fops);
void debugfs_remove(struct dentry *dentry);
void debugfs_remove_recursive(struct dentry *dentry);
```
还可以创建单个文件以及BLOB文件。

### /proc

`/proc` 是一种伪文件系统（也即虚拟文件系统），存储的是当前内核运行状态的一系列特殊文件，用户可以通过这些文件查看有关系统硬件及当前正在运行进程的信息，甚至可以通过更改其中某些文件来改变内核的运行状态。 
大多数虚拟文件可以使用文件查看命令如cat、more或者less进行查看。

```
https://elixir.bootlin.com/linux/v3.7/source/include/linux/proc_fs.h#L152

struct proc_dir_entry *proc_mkdir(const char *name,
                                  struct proc_dir_entry *parent) 

struct proc_dir_entry *proc_create(const char *name, 
                                    umode_t mode, 
                                    struct proc_dir_entry *parent, 
                                    const struct file_operations *proc_fops);

struct proc_dir_entry *proc_create_data(const char *, umode_t,
                                         struct proc_dir_entry *,
                                         const struct file_operations *,
                                         void *);
void proc_remove(struct proc_dir_entry *);

void remove_proc_entry(const char *, struct proc_dir_entry *);
```
## 并发和竞争

### 自旋锁

作为互斥锁，自旋锁只有两个值：上锁和解锁。如果锁是可用的，上锁位被置为并且代码进入临界区；相反，如果这个锁已被获得，代码进入一个紧凑的循环中反复检查这个锁，直到变得可用。

自旋锁原语在`<linux/spinlock.h>`中。一个实际的锁有类型 `spinlock_t`。象任何其他数据结构，一个自旋锁必须初始化。 这个初始化可以在编译时完成`spinlock_t my_lock = SPIN_LOCK_UNLOCKED;`或运行时使用`void spin_lock_init(spinlock_t *lock);`。

在进入临界区钱，必须获得`lock`：
`void spin_lock(spinlock_t *lock);`


在获得自旋锁之前，禁止中断(只在本地处理器)；之前的中断状态保存在 `flags` 里:
`void spin_lock_irqsave(spinlock_t *lock, unsigned long flags); `

获取锁之前禁止软件中断，但是硬件中断留作打开的：
`void spin_lock_bh(spinlock_t *lock);`

当禁止本地中断时可以使用:
`void spin_lock_irq(spinlock_t *lock);`

为释放一个已获得的锁，传递它给:
```
void spin_unlock(spinlock_t *lock);
void spin_unlock_irqrestore(spinlock_t *lock, unsigned long flags);
void spin_unlock_irq(spinlock_t *lock);
void spin_unlock_bh(spinlock_t *lock);
```
每个 spin_unlock 变体恢复由对应的 spin_lock 函数锁做的工作。传递给spin_unlock_irqrestore 的 flags 参数必须是传递给 spin_lock_irqsave 的同一个变量。你必须也调用 spin_lock_irqsave 和 spin_unlock_irqrestore 在同一个函数里。


## 内存分配

内核通过`kmalloc`和`kfree`来分配和释放内存，位于`<linux/slab.h>`。

### kmalloc

这个函数快（除非它阻塞）并且不清零它获得的内存；分配的区仍然持有它原来的内容，分配的区也是在物理内存中连续。

```
void *kmalloc(size_t size, int flags);
```

第1个参数是要分配的块的大小；第2个参数，分配标志位于 `<linux/gfp.h>`，以几个方式控制`kmalloc`的行为，内部最终通过调用 `__get_free_pages` 来进行，它是 `GFP_` 前缀的来源。

- GFP_KERNEL

`GFP_KERNEL` 代表运行在内核空间的进程而进行的。
使用 GFP_KENRL意味着 kmalloc 能够使当前进程在少内存的情况下睡眠来等待一页。

- GFP_ATOMIC

用在中断处理和进程上下文之外的其他代码中分配内存，进程从不睡眠等待。

- GFP_USER

用来为用户空间页来分配内存，它可能睡眠。

- \__GFP_DMA

这个标志要求分配在能够 DMA 的内存区，跟平台相关。

- \__GFP_HIGHMEM

这个标志指示分配的内存可以位于高端内存，跟平台相关。


## 设备驱动程序

## 设备驱动程序模型

Linux设备模型提取了设备操作的共同属性，进行抽象，并将这部分共同的属性在内核中实现，而为需要新添加设备或驱动提供一般性的统一接口，这个框架称为设备驱动程序模型。

Linux设备模型学习分为：Linux设备底层模型，描述设备的底层层次实现（kobject）；Linux上层容器，包括总线类型（bus_type）、设备（device）和驱动（device_driver）。

### sysfs

`sysfs`是Linux一种特殊的文件系统，允许 **用户态**应用程序访问内核内部数据结构，并提供了内核数据结构的附加信息。
sysfs`文件系统的主要目的是展现设备驱动程序模型组件间的层次关系。`sysfs`被安装于`/sys`目录。
`/sys`目录描述了设备驱动模型的层次关系。
主要包括：

|设备|描述|
|---|---|
|block|所有块设备|
|devices | 系统所有设备（块设备特殊），对应struct device的层次结构|
| bus |系统中所有总线类型（指总线类型而不是总线设备，总线设备在devices下），bus的每个子目录都包含 <br/>    --devices：包含到devices目录中设备的软链接 <br/>--drivers：与bus类型匹配的驱动程序 <br/>|
|class |系统中设备类型（如声卡、网卡、显卡等）|
|fs | 一些文件系统，具体可参考filesystems /fuse.txt中例子|
|dev| 包含2个子目录<br>--char：字符设备链接，链接到devices目录，以<major>:<minor>命名<br>--block：块设备链接|

### kobject

`kojbect`是设备驱动程序模型的核心数据结构，每个kobject对应于sysfs文件系统中的一个目录，它的功能是提供引用计数和维持父子（parent）结构、平级（sibling）目录关系，许多kobject结构就构成了层次结构。

`kset`: 它用来对同类型对象提供一个包装集合，在内核数据结构上它也是由内嵌一个 kboject 实现，因而它同时也是一个 kobject (面向对象 OOP 概念中的继承关系。

### 设备模型的上层容器

上层容器包括总线类型（bus_type）、设备（device）和驱动（device_driver）。

每一种总线类型由`struct bus_type`对象描述。 `bus_type` 通过扫描 *设备链表*和 *驱动链表*，使用 `match`方法查找匹配的设备和驱动，然后将 `struct device` 中的 `driver` 设置为匹配的驱动，将 `struct device_driver` 中的 `device` 设置为匹配的设备，这就完成了将 **总线** 、 **设备** 和 **驱动** 3者之间的关联。

每个设备由一个`struct device`对象来描述。

内核提供了`device_create`函数在`sysfs`创建和注册 **设备**。
```
struct device *device_create( struct class *class, 
                              struct device *parent,
                              dev_t devt, 
                              void *drvdata, 
                              const char *fmt, 
                              ...)
- class：设备要注册到的struct class对象
- parent：新设备的父设备，如果没有就指定为NULL
- devt：主从设备号
- drvdata：在回调时添加到设备中的数据
- fmt：设备名字
```
相应的设备移除函数是 `device_destory`。
```
void device_destroy ( struct class *    class,
                      dev_t   devt);
```

每个驱动程序由`struct device_driver`对象描述。

每个类由一个`struct class`对象描述。所有的类对象都属于与`/sys/class`目录相对应的`class_subsys`子系统。同一类中的设备驱动程序可以对用户态应用程序提供相同的功能。

内核提供了`create_class`函数来创建一个类对象，这个类存在于 `sysfs` 下面，`class_destroy`函数来注销一个类对象。
```
struct class * class_create ( struct module *   owner,
  const char *    name);

```
+ owner: 指针，指向了拥有此 struct class 的模块。
+ name： 指针，指向了此类的字符串。

```
void class_destroy (  struct class *    cls);
```
+ cls: 指针，指向了要销毁的struct class 对象。

一旦创建好了这个类，再调用 `device_create()` 函数来在/dev目录下创建相应的设备节点。这样，加载模块的时候，用户空间中的udev会自动响应 `device_create()`函数，去 `/sys` 下寻找对应的类从而创建设备节点。

### 等待队列




### 参考
[Linux设备驱动模型](https://blog.csdn.net/xiahouzuoxin/article/details/8943863)

### 设备文件

设备文件分为字符设备文件和块设备文件。差异为：
- 块设备的数据可以被随机访问，典型的例子是硬盘、CD-ROM驱动器。
- 字符设备的数据或者不可以被随机访问，或者可以被随机访问，但是访问随机数据所需要的时间很大程度上依赖于数据在设备间的位置。

设备标识符由设备文件的类型（字符或块）和一对参数组成。第一个参数称为主设备号（major number），它标识了设备的类型。具有相同主设备号和类型的所有设备文件共享相同的文件操作集合，因为它们由同一个设备驱动程序处理的。第二个参数称为次设备号（minor number），它标识了主设备号相同的设备组中的一个特定设备。例如，由相同的磁盘控制器管理的一组磁盘具有相同的主设备号和不同的次设备号。

`mknod`系统调用用来创建设备文件，参数包括设备文件名、设备类型、主设备号及次设备号。设备文件通常包含在`/dev`目录中。主设备号对应的宏位于`include/linux/major.h`中。

为了解决设备号分配不足问题，Linux 2.6增加了设备号的编码大小，由原来的8位的次设备号改为20位的次设备号，主设备号的编码为12位。通常把这两个参数合并成一个32位的dev_t变量；`MAJOR`和`MINOR`宏可以从`dev_t`中提取出主设备号和次设备号，而`MKDEV`宏可以把主设备号和次设备号合并成一个`dev_t`值。定义为`typedef u_long dev_t;`

对于分配设备号和创建设备文件，静态的方法容易产生冲突并且移植性不好，因此如今更倾向于动态处理。




## 重要函数

### ioctl


通常用户态程序访问内核态资源要通过系统调用实现，传统的操作系统通常用这种方式给用户空间提供了上百个系统调用。因为大多数硬件设备只能够在内核空间内直接寻址,但是当访问非标准硬件设备这些系统调用显得不合适,有时候用户模式可能需要直接访问设备，比如，一个系统管理员可能要修改网卡的配置。现代操作系统提供了各种各样设备的支持，有一些设备可能没有被内核设计者考虑到，如此一来提供一个这样的系统调用来使用设备就变得不可能了。

为了解决这个问题，内核被设计成可扩展的，可以加入一个称为设备驱动的模块，驱动的代码允许在内核空间运行而且可以对设备直接寻址。一个`ioctl`接口是一个独立的系统调用，通过它用户空间可以跟设备驱动沟通。对设备驱动的请求是一个以设备和请求号码为参数的`ioctl`调用，如此内核就允许用户空间访问设备驱动进而访问设备而不需要了解具体的设备细节，同时也不需要一大堆针对不同设备的系统调用。

`ioctl`（input/output control）是一个专用于设备输入输出操作的系统调用，该调用传入一个跟设备有关的请求码，系统调用的功能完全取决于请求码。

#### 使用ioctl顺序

+ 在驱动中创建ioctl命令
+ 在驱动中写ioctl函数
+ 在用户态程序创建ioctl命令
+ 在用户态程序使用ioctl命令

#### 创建ioctl命令
32位的命令数字由4部分组成：
+ The Magic Number，魔数，是唯一得数字或字符，8 位宽(\_IOC_TYPEBITS)。
+ Command Number ，序(顺序)号. 它是 8 位(\_IOC_NRBITS)宽. r.
+ Argument type ，参数类型，14 位。
+ Direction of data transfer，数据传送的方向，2位。\_IOC_NONE(没有数据传输), \_IOC_READ, \_IOC_WRITE, 和 \_IOC_READ|\_IOC_WRITE (数据在 2 个方
向被传送)。

但是在创建ioctl命令时，使用宏定义生成。
```
#define "ioctl_command" _IOX("magic number","command number","argument type")
```
`_IOX` 可以取值:
+ `_IO`: \_IO(type,nr)，给没有参数的命令。
+ `_IOW`: \_IOW(type,nr,datatype)，给写数据 (copy_from_user)
+ `_IOR`: \_IOR(type, nre, datatype)，给从驱动中读数据的(copy_to_user)
+ `_IOWR`: \_IOWR(type,nr,datatype)，给双向传送。


头文件中定义的宏, 可在驱动中来解码这个宏: \_IOC_DIR(nr), \_IOC_TYPE(nr), \_IOC_NR(nr), 和 \_IOC_SIZE(nr) 。

比如字体驱动程序的IOCTL调用命令 SETFONT。
```
#define PRIN_MAGIC 'S'
#define SEQ_NO 1
#define SETFONT __IOW(PRIN_MAGIC, SEQ_NO, unsigned long)
```
在用户态调用方式为
```
char *font = "Arial";
ret_val = ioctl(fd, SETFONT, font); 
```
`font`是一个指针，它是一个被表示成 `unsigned long` 的地址，`_IOW` 表示只写数据，

#### 驱动中的ioctl命令
定义完IOCTL命令后，下一步就是要在驱动中完成ioctl函数。函数定义如下：
```
long etx_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case WR_VALUE:
            copy_from_user(&value ,(int32_t*) arg, sizeof(value));
            printk(KERN_INFO "Value = %d\n", value);
            break;
        case RD_VALUE:
            copy_to_user((int32_t*) arg, &value, sizeof(value));
            break;
        default:
            return -EINVAL;
    }
    return 0;
}
static struct file_operations fops =
{
  .owner = THIS_MODULE,
  .unlocked_ioctl = etx_ioctl,
};
```
ioctl函数参数
+ `file` : 文件指针，指向了应用程序传递的文件。
+ `cmd` :  从用户空间调用的ioctl命令。
+ `arg` : 用户空间传递的参数。

对于响应一个无效的 ioctl 命令，返回 `-ENIVAL("Invalid argument")` 。

#### 创建用户空间的ioctl命令

用户空间的ioctl命令和内核空间的ioctl命令一样，因此这些命令和参数需要在内核态和用户态间共享，一般放在各自的头文件中。


#### 用户空间调用ioctl系统调用

```
#include <sys/ioctl.h>
int ioctl(int fd, unsigned long request, ...);
```

`fd`是打开的文件描述符，`request`是用户程序对设备的控制命令码，后面省略号表示命令补充参数，如果存在第三个参数，那它是一个指针`void *` 类型。

成功返回0或者正数，出错返回-1，`errno`如下：

```
ERRORS：
       EBADF  fd is not a valid descriptor.

       EFAULT argp references an inaccessible memory area.

       EINVAL request or argp is not valid.

       ENOTTY fd is not associated with a character special device.

       ENOTTY The specified request does not apply to the kind of object that the descriptor fd references.
```

比如
```
int number=2;
ioctl(fd, WR_VALUE, (int32_t*) &number); 
```

[IOCTL Linux device driver](https://stackoverflow.com/questions/15807846/ioctl-linux-device-driver)
[Device Drivers, Part 9: I/O Control in Linux](https://opensourceforu.com/2011/08/io-control-in-linux/)
[Linux Device Driver Tutorial Part 8 – I/O Control in Linux IOCTL()](https://embetronicx.com/tutorials/linux/device-drivers/ioctl-tutorial-in-linux/)
内核模块[ioctl.c](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/kernel_module/ioctl.c)和[ioctl.h](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/kernel_module/ioctl.h)还有用[户态应用程序ioctl.c](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/kernel_module/user/ioctl.c)与[调用脚本](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/rootfs_overlay/ioctl.sh)

### list_head

Linux的列表`list_head`位于`include/linux/types.h`。
```
struct list_head {
    struct list_head *next, *prev;
};
```

`struct list_head`是双向链表。在Linux内核链表中，需要用链表组织起来的数据通常会包含一个struct list_head成员。
常用操作为：

- 声明和初始化
```
#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)

#define INIT_LIST_HEAD(ptr) do { \
    (ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)
```

- 插入

```
// 在表头插入
static inline void list_add(struct list_head *new, struct list_head *head);
// 在表尾插入
static inline void list_add_tail(struct list_head *new, struct list_head *head);
```

- 删除

```
static inline void list_del(struct list_head *entry);
```

- 迁移

```
static inline void list_move(struct list_head *list, struct list_head *head);
static inline void list_move_tail(struct list_head *list, struct list_head *head);
```

- 合并

整个列表合并
```
static inline void list_splice(struct list_head *list, struct list_head *head);
```

- 遍历

通过这个list_head成员访问到作为它的所有者的节点数据。
```
list_entry(nf_sockopts->next, struct nf_sockopt_ops, list);
```

这里有两个宏，`list_for_each`的`pos`是`(struct list_head \*)`，而`list_for_each_entry`的`pos`是数据项结构指针类型。
```
#define list_for_each(pos, head) \
	for (pos = (head)->next, prefetch(pos->next); pos != (head); \
        pos = pos->next, prefetch(pos->next))
#define list_for_each_entry(pos, head, member)
```

#### 参考
[深入分析 Linux 内核链表](https://www.ibm.com/developerworks/cn/linux/kernel/l-chain/)

### 工作队列work_queue

创建一个名为`my_work`的结构体变量。
```
struct work_struct my_work; 
```
初始化已经创建的my_work，其实就是往这个结构体变量中添加处理函数的入口地址和data的地址，通常在驱动的open函数中完成
```
INIT_WORK(&my_work,my_func,&data); 
```

将工作结构体变量添加入系统的共享工作队列，添加入队列的工作完成后会自动从队列中删除。
```
schedule_work(&my_work); 
```

### dma



初始化散列表项目。
```
void sg_init_one(struct scatterlist *sg, const void *buf, unsigned int buflen)
```

- `sg`：散列表
- `buf`：IO的虚拟地址
- `buflen`：IO长度



# 参考资料
[1] 深入理解Linux内核
[2] Linux设备驱动程序
[3] [编写Linux内核模块——第一部分：前言](http://www.infoq.com/cn/articles/linux-kernel-module-part01)
[4] [编写Linux内核模块——第二部分：字符设备](http://www.infoq.com/cn/articles/linux-kernel-module-part02)
[5] [标记化结构初始化语法---结构体成员前加小数点](https://blog.csdn.net/ixidof/article/details/7893680)
[6] [
sysfs、udev 和 它们背后的 Linux 统一设备模型](https://www.binss.me/blog/sysfs-udev-and-Linux-Unified-Device-Model/)
