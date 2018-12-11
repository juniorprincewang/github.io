---
title: Linux内核debugfs
date: 2018-12-11 10:04:56
tags:
- debugfs
categories:
- linux
---
`debugfs` 虚拟文件系统是一种内核空间与用户空间的接口，基于libfs库实现，专用于开发人员调试，便于向用户空间导出内核空间数据。
<!-- more -->

内核开发者经常需要向用户空间应用输出一些调试信息，在稳定的系统中可能根本不需要这些调试信息。
但是在开发过程中，为了搞清楚内核的行为，调试信息非常必要。
printk可能是用的最多的，但它并不是最好的，调试信息只是在开发中用于调试，而printk将一直输出，因此开发完毕后需要清除不必要的printk语句。
另外如果开发者希望用户空间应用能够改变内核行为时，printk就无法实现。
因此，需要一种新的机制，那只有在需要的时候使用，它在需要时通过在一个虚拟文件系统中创建一个或多个文件来向用户空间应用提供调试信息。

为了使得开发者更加容易使用这样的机制，Greg Kroah-Hartman开发了debugfs（在2.6.11中第一次引入），它是一个虚拟文件系统，专门用于输出调试信息，该文件系统非常小，很容易使用，可以在配置内核时选择是否构件到内核中，在不选择它的情况下，使用它提供的API的内核部分不需要做任何改动。

# 挂载debugfs文件系统

要使用debugfs，需要在内核编译配置中配置 `CONFIG_DEBUG_FS=y`选项，一般的发行版都会默认编译进了内核。通过下面命令查看
```
cat /boot/config-`uname -r` | grep CONFIG_DEBUG_FS
```
并且将其自动挂载默认的目录(`/sys/kernel/debug`)，也可手动挂载到其它位置：
```
mkdir /debugfs
mount -t debugfs none /debugfs
```
# 操作

## 创建目录和文件

使用debugfs的开发者首先需要在文件系统中创建一个目录，下面函数用于在debugfs文件系统下创建一个目录：
```
struct dentry *debugfs_create_dir(const char *name, struct dentry *parent);
```
+ `name`是要创建的目录名，
+ `parent` 指定创建目录的父目录的 `dentry`，如果为NULL，目录将创建在debugfs文件系统的根目录下。如果返回为-ENODEV，表示内核没有把debugfs编译到其中，如果返回为NULL，表示其他类型的创建失败，如果创建目录成功，返回指向该目录对应的dentry条目的指针。

下面函数用于在debugfs文件系统中创建一个文件：
```
struct dentry *debugfs_create_file(const char *name, mode_t mode,
                               struct dentry *parent, void *data,
                               struct file_operations *fops);
```
+ 参数name指定要创建的文件名，
+ 参数mode指定该文件的访问许可，
+ 参数parent指向该文件所在目录，
+ 参数data为该文件特定的一些数据，
+ 参数fops为实现在该文件上进行文件操作的 `file_operations` 结构指针。

## 导出基本的数据类型变量


当然，在一些情况下，开发者可能仅需要使用用户应用可以控制的变量来调试。
debugfs可以将内核中基本整数类型的变量导出为单个文件，在用户空间中可以直接对其读写(如使用cat、echo命令)，只要权限允许即可。
支持的类型有：`u8`, `u16`, `u32`, `u64`, `size_t`和` bool`。
其中 `bool` 类型在内核中要定义为 `u32` 类型，在用户空间中对应的文件内容则显示为 `Y` 或` N`。

debugfs提供的API为：
```
struct dentry *debugfs_create_u8(const char *name, mode_t mode, 
                                     struct dentry *parent, u8 *value);
struct dentry *debugfs_create_u16(const char *name, mode_t mode, 
                                      struct dentry *parent, u16 *value);
struct dentry *debugfs_create_u32(const char *name, mode_t mode, 
                                      struct dentry *parent, u32 *value);
struct dentry *debugfs_create_bool(const char *name, mode_t mode, 
										struct dentry *parent, u32 *value);
```
+ 参数name和mode指定文件名和访问许可，
+ 参数value为需要让用户应用控制的内核变量指针。

示例代码如下：
```
static struct dentry *root_d = debugfs_create_dir("exam_debugfs", NULL); //在debugfs根目录下创建新目录exam_debugfs，然会新建目录的目录项指针
static u8 var8;
debugfs_create_u8("var-u8", 0664, root_d, &var8); //在exam_debugfs中创建变量var8对应的文件，名为var-u8，权限为0664
static u32 varbool;
debugfs_create_bool("var-bool", 0664, root_d, &varbool); //bool变量
```

## 销毁目录和文件

当内核模块卸载时，Debugfs并不会自动清除该模块创建的目录或文件，因此对于创建的每一个文件或目录，开发者必须调用下面函数清除：
```
void debugfs_remove(struct dentry *dentry);
```
或者可调用 `debugfs_remove_recursive` 递归删除整个目录。
```
void debugfs_remove_recursive(struct dentry *dentry);
```
参数dentry为上面创建文件和目录的函数返回的dentry指针。


# 参考
1. [DebugFS Tutorial](https://github.com/chadversary/debugfs-tutorial)
2. [在 Linux 下用户空间与内核空间数据交换的方式，第 2 部分-procfs、seq_file、debugfs和relayfs](https://www.ibm.com/developerworks/cn/linux/l-kerns-usrs2/index.html)
3. [Linux内核空间-用户空间通信之debugfs](http://www.embeddedlinux.org.cn/emb-linux/file-system/201704/11-6516.html)
4. [debugfs.c](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/kernel_module/debugfs.c) 和 配套的脚本[debugfs.sh](https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/rootfs_overlay/debugfs.sh)