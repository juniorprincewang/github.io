---
title: androidx86 bootup
tags:
  - android
categories:
  - - linux
    - androidx86
date: 2023-09-08 09:48:19
---





Android-x86项目向Intel设备提供了 Android Board Support Package(BSP)，使用通用media来引导所有的intel 设备，即将引导顺序分成两个阶段。
第一阶段：启动一个最小限度的嵌入式Linux环境，以启用硬件设备。
第二阶段：通过 chroot 或者 switch_root 来切换到 Android system。

本文总结了Android-x86-7.1 的启动第一阶段。

<!-- more -->

# bootable newinstaller

第一阶段Androidx86使用了特别的ramdisk `initrd.img`，源码位于：`$AOSP/bootable/newinstaller`，源码构成：

- `boot`: 对于安装媒介的bootloader。Androidx86的镜像可以分成不同的格式（ISO，UEFI等）
- `editdisklbl`：用于编辑system image分区的host工具
- `initrd`：第一阶段boot的ramdisk
- `install`:Androidx86的 installer
- `Android.mk`:Makefile文件

通过命令 `make iso_img/usb_img/efi_img`来 build `newinstaller`，除了生成安装镜像外，还生成另外两个镜像：`initrd.img` 和 `install.img` 。

- initrd.img：第一阶段boot up的ramdisk image
- install.img：包含了Androidx86的installer

解压缩 img 命令可以通过`cpio`获得：
```
zcat ramdisk.img | cpio -id > /dev/null
```

修改完后重新压缩回img
```
find . | cpio -o -H newc | gzip > ../ramdisk.img
```

iso_img 生成的androidx86.iso 镜像解压缩后的目录：

    ├── initrd.img
    ├── install.img
    ├── isolinux
    │   ├── android-x86.png
    │   ├── boot.cat
    │   ├── isolinux.bin
    │   ├── isolinux.cfg
    │   ├── TRANS.TBL
    │   └── vesamenu.c32
    ├── kernel
    ├── ramdisk.img
    ├── system.sfs
    └── TRANS.TBL


首先是安装Androidx86 通过镜像中的 isolinux 引导安装。 
ISOLINUX其实是一个简单的Linux系统，对应于源码路径为 *bootable/newinstaller/boot/isolinux/*，包括引导文件 isolinux.bin、配置文件 *isolinux.cfg*、vesamenu启动界面文件和背景图片，boot.cat是创建生成文件。

- 引导程序 `isolinux.bin`： 这个文件是ISOLINUX的引导文件。相当于Linux系统中的grub程序。在系统启动时，先加载isolinux.bin来启动系统，isolinux.bin会根据配置文件isolinux.cfg选择不同的启动选项来启动系统。这个文件是一个二进制文件，在编译isolinux时可以得到。
- 配置引导项文件 `isolinux.cfg`: 用于isolinux.bin在引导时根据该配置文件的配置内容的不同，而选择不同的引导项来启动系统。
根据 isolinux.cfg 中的label信息，共有8个启动选项。

```sh
label livem
	menu label Live CD - ^Run OS_TITLE without installation
	kernel /kernel
	append initrd=/initrd.img CMDLINE quiet SRC= DATA=

label debug
	menu label Live CD - ^Debug mode
	kernel /kernel
	append initrd=/initrd.img CMDLINE DEBUG=2 SRC= DATA=

label install
	menu label ^Installation - Install OS_TITLE to harddisk
	kernel /kernel
	append initrd=/initrd.img CMDLINE INSTALL=1 DEBUG=
	
label nosetup
	menu label Live CD - No ^Setup Wizard
	kernel /kernel
	append initrd=/initrd.img CMDLINE quiet SETUPWIZARD=0 SRC= DATA=

label vesa
	menu label Live CD ^VESA mode - No GPU hardware acceleration
	kernel /kernel
	append initrd=/initrd.img CMDLINE nomodeset vga=ask SRC= DATA=

label auto_install
	menu label Auto_^Installation - Auto Install to specified harddisk
	kernel /kernel
	append initrd=/initrd.img CMDLINE AUTO_INSTALL=0 DEBUG=

label auto_update
	menu label Auto_^Update - Auto update OS_TITLE
	kernel /kernel
	append initrd=/initrd.img CMDLINE AUTO_INSTALL=update DEBUG=
```

label为 *install* 的启动选项制定了 kernel 路径以及内核参数append信息，initrd文件路径， 变量CMDLINE、INSTALL=1、DEBUG=空。
`SRC` 与 `DATA` 参数分别为后面的initrd启动引导传入 boot_image 路径和 data分区路径，分别对应了 `$SRC` 与 `$DATA` 变量。
ISOLINUX系统在使用isolinux.bin文件引导完成以后，就会调用一个内核来启动一个简单的Linux系统，这个简单的Linux系统就是由initrd.img完成的。

    initrd/
    ├── android
    ├── bin
    │   ├── busybox
    │   ├── ld-linux.so.2
    │   └── lndir
    ├── hd
    ├── init
    ├── initrd.img
    ├── iso
    ├── lib
    │   ├── ld-linux.so.2 -> /bin/ld-linux.so.2
    │   ├── libcrypt.so.1
    │   ├── libc.so.6
    │   ├── libdl.so.2
    │   ├── libm.so.6
    │   ├── libntfs-3g.so.31
    │   ├── libpthread.so.0
    │   └── librt.so.1
    ├── mnt
    ├── proc
    ├── sbin
    │   └── mount.ntfs-3g
    ├── scripts
    │   ├── 00-ver
    │   ├── 0-auto-detect
    │   ├── 1-install
    │   ├── 2-mount
    │   ├── 3-tslib
    │   └── 4-dpi
    ├── sfs
    ├── sys
    └── tmp

系统第一阶段的启动通过 `initrd/init` shell脚本文件启动。
在Linux系统启动时，加载完成内核以后，就开始调用该脚本了。建议将系统启动相关的内容放置在这里执行，比如外设挂载到指定分区，而将自己的脚本放置在可执行目录下［bin/sbin等］，在init脚本中调用该脚本。

# initrd/init

将 init 脚本拆分分析：

## 环境变量和LOG设置

```sh
#!/bin/busybox sh

PATH=/sbin:/bin:/system/bin:/system/xbin; export PATH

# auto installation
[ -n "$AUTO_INSTALL" ] && INSTALL=1

# configure debugging output
if [ -n "$DEBUG" -o -n "$INSTALL" ]; then
	LOG=/tmp/log
	set -x
else
	LOG=/dev/null
	test -e "$LOG" || busybox mknod $LOG c 1 3
fi
exec 2>> $LOG
```


1. busybox 中的sh执行此脚本。  
2. 设置 `$PATH` 环境变量。  
3. 如果kernel参数传入的 `$AUTO_INSTALL` 有值就设置 变量 `INSTALL`。  
4. 如果传入了 `$DEBUG` 有值或者 `$INSTALL` 有值，设置日志路径变量 `$LOG` 为 `/tmp/log`；否则设置 `$LOG`为 `/dev/null`，不存在就创建。  
5. 将错误输出stderr 也追加输出到 `$LOG`中。  

## 设置controlling tty，初始化 `/proc`、`/sys`、`/dev`等目录

```sh
# early boot
if test x"$HAS_CTTY" != x"Yes"; then
	# initialise /proc and /sys
	busybox mount -t proc proc /proc
	busybox mount -t sysfs sys /sys
	# let busybox install all applets as symlinks
	busybox --install -s
	# spawn shells on tty 2 and 3 if debug or installer
	if test -n "$DEBUG" || test -n "$INSTALL"; then
		# ensure they can open a controlling tty
		mknod /dev/tty c 5 0
		# create device nodes then spawn on them
		mknod /dev/tty2 c 4 2 && openvt
		mknod /dev/tty3 c 4 3 && openvt
	fi
	if test -z "$DEBUG" || test -n "$INSTALL"; then
		echo 0 0 0 0 > /proc/sys/kernel/printk
	fi
	# initialise /dev (first time)
	mkdir -p /dev/block
	echo /sbin/mdev > /proc/sys/kernel/hotplug
	mdev -s
	# re-run this script with a controlling tty
	exec env HAS_CTTY=Yes setsid cttyhack /bin/sh "$0" "$@"
fi
```
后面环境初始化需要在可控的tty下进行，此段脚本用于设置controlling tty。

1. 最开始是没有设置变量 `HAS_CTTY`
2. 初始化 `/proc` 和 `/sys` 目录，这里使用 busybox 中的 `mount` 命令
3. 让busybox将所有小程序作为符号链接安装
4. 如果debug 或者 installer，则在tty2 和 tty3 启动 shell
5. 如果不debug 或 进行installer，将kernel的printk控制台日志等级设置为最高0，即只有最高错误才输出。
6. 首次初始化 `/dev`，
    1. 创建目录 `/dev/block`，顺带创建 `/dev` 目录。
    2. 把/sbin/mdev写到/proc/sys/kernel/hotplug文件里。当有热插拔事件产生时，内核会调用/proc/sys/kernel/hotplug文件里指定的应用程序来处理热插拔事件。 
    3. `mdev -s`：系统启动时，通过执行`mdev -s` 扫描 /sys/class和/sys/block，在目录中查找dev文件。例如：/sys/class/tty/tty0/dev，输出它的内容为 `4:0`，即主设备号是4，次设备号是0，dev的上一级目录为设备名，这里是tty0。/sys/class/下的每个文件夹都代表
着一个子系统。
7. 重新运行init脚本`$0`与附带参数`$@`，本次附带了环境变量 `env HAS_CTTY=Yes`

[echo /sbin/mdev ＞ /proc/sys/kernel/hotplug 作用解析](https://blog.csdn.net/phmatthaus/article/details/107180696)

## 获取`BOOT_IMAGE`、`RAMDISK`和内核启动参数

接下来脚本就在可控tty中执行了。

```sh
echo -n Detecting Android-x86...

[ -z "$SRC" -a -n "$BOOT_IMAGE" ] && SRC=`dirname $BOOT_IMAGE`
[ -z "$RAMDISK" ] && RAMDISK=ramdisk.img || RAMDISK=${RAMDISK##/dev/}

for c in `cat /proc/cmdline`; do
	case $c in
		iso-scan/filename=*)
			SRC=iso
			eval `echo $c | cut -b1-3,18-`
			;;
		*)
			;;
	esac
done
```

1. 这串字符 `Detecting Android-x86...` 在虚拟机启动窗口出现。
2. 如果 `$SRC` 传入为空并且设置了 `$BOOT_IMAGE`，就将 `$SRC` 设置为 `$BOOT_IMAGE` 所在目录。
3. 如果 `$RAMDISK` 未设置，则设置其为 ramdisk.img；否则从 `$RAMDISK` 变量中删除 `/dev/`字符串。
4. `cat proc/cmdline` 查看 内核启动参数。可以看出这里不存在此变量 `iso-scan/filename`

从模拟器得出：  

> quiet nomodeset root=/dev/ram0 androidboot.selinux=permissive buildvariant=userdebug SRC=/androidx86

从 ubuntu 得到的kernel启动参数：

> BOOT_IMAGE=/boot/vmlinuz-5.15.0-50-generic root=UUID=xxxxxxxxxxxxxxxxxxxxxxx ro quiet splash

## 搜索启动ROOT目录

```sh
mount -t tmpfs tmpfs /android
cd /android
while :; do
	for device in ${ROOT:-/dev/[hmnsv][dmrv][0-9a-z]*}; do
		check_root $device && break 2
		mountpoint -q /mnt && umount /mnt
	done
	sleep 1
	echo -n .
done
```

1. 挂载 `tmpfs` 文件到 `/android` 目录，并进入 `/android` 目录操作。
2. 从环境变量 `$ROOT`搜索当前的设备，如果没有定义 `$ROOT`就搜索 `/dev/sda` 一类的设备。
3. 睡眠1秒，再输出 `.`。

## `try_mount`、`check_root`、`remount_rw` 函数

```sh
try_mount()
{
	RW=$1; shift
	if [ "${ROOT#*:/}" != "$ROOT" ]; then
		# for NFS roots, use nolock to avoid dependency to portmapper
		mount -o $RW,noatime,nolock $@
		return $?
	fi
	case $(blkid $1) in
		*TYPE=*ntfs*)
			mount.ntfs-3g -o rw,force $@
			;;
		*TYPE=*)
			mount -o $RW,noatime $@
			;;
		*)
			return 1
			;;
	esac
}
check_root()
{
	if [ "`dirname $1`" = "/dev" ]; then
		[ -e $1 ] || return 1
		blk=`basename $1`
		[ ! -e /dev/block/$blk ] && ln $1 /dev/block
		dev=/dev/block/$blk
	else
		dev=$1
	fi
	try_mount ro $dev /mnt || return 1
	if [ -n "$iso" -a -e /mnt/$iso ]; then
		mount --move /mnt /iso
		mkdir /mnt/iso
		mount -o loop /iso/$iso /mnt/iso
	fi
	if [ -e /mnt/$SRC/$RAMDISK ]; then
		zcat /mnt/$SRC/$RAMDISK | cpio -id > /dev/null
	elif [ -b /dev/$RAMDISK ]; then
		zcat /dev/$RAMDISK | cpio -id > /dev/null
	else
		return 1
	fi
	if [ -e /mnt/$SRC/system.sfs ]; then
		mount -o loop,noatime /mnt/$SRC/system.sfs /sfs
		mount -o loop,noatime /sfs/system.img system
	elif [ -e /mnt/$SRC/system.img ]; then
		remount_rw
		mount -o loop,noatime /mnt/$SRC/system.img system
	elif [ -s /mnt/$SRC/system/build.prop ]; then
		remount_rw
		mount --bind /mnt/$SRC/system system
	elif [ -z "$SRC" -a -s /mnt/build.prop ]; then
		mount --bind /mnt system
	else
		rm -rf *
		return 1
	fi
	mkdir -p mnt
	echo " found at $1"
	rm /sbin/mke2fs
	hash -r
}
remount_rw()
{
	# "foo" as mount source is given to workaround a Busybox bug with NFS
	# - as it's ignored anyways it shouldn't harm for other filesystems.
	mount -o remount,rw foo /mnt
}
```

1. `check_root` 传入的参数是 `$device` 设备，如果 `$device` 是 `/dev` 设备下面的，首先判断其存在，在判断是否存在 `/dev/block/$blk`，没有则创建硬链接。`$dev` 设置为此设备。
2. 将 `$dev` 载入 /mnt 目录。
    1. `try_mount` 对`mount` 命令做了层包装，加入了 `noatime` option。通过查询设备的TYPE类型，针对ntfs专门mount。
3. 如果设置了 `$iso` 变量，并且 `/mnt/$iso` 存在，则将 /mnt mount到 /iso。再将 /iso/$iso mount 到 /mnt/iso 上。
4. 解压缩 `$RAMDISK` 镜像中的内容到当前目录。
5. 如果发现了system.sfs，则mount system.sfs 到 /sfs，再从目录/sfs 中mount system.img 到 system。也就是说 system.sfs是对system.img 的又一层压缩。如果直接发现 system.img，则直接mount到system目录。剩下的判断逻辑是直接将build.prop所在目录重新bind到 system。
6. 在当前目录创建 mnt 目录。
7. 输出 `found at $1`，这里在虚拟机启动窗口输出的是 *found at /dev/sda1*。
8. 删除 `/sbin/mk2fs` ?
9. 清空命令表hash 缓存

## 创建`/`根目录等目录的链接、加载模块、挂载data、sdcard等分区

```sh
ln -s mnt/$SRC /src
ln -s android/system /
ln -s ../system/lib/firmware ../system/lib/modules /lib

if [ -n "$INSTALL" ]; then
	zcat /src/install.img | ( cd /; cpio -iud > /dev/null )
fi

if [ -x system/bin/ln -a -n "$BUSYBOX" ]; then
	mv -f /bin /lib .
	sed -i 's|\( PATH.*\)|\1:/bin|' init.environ.rc
	rm /sbin/modprobe
	busybox mv /sbin/* sbin
	rmdir /sbin
	ln -s android/bin android/lib android/sbin /
	hash -r
fi

# load scripts
for s in `ls /scripts/* /src/scripts/*`; do
	test -e "$s" && source $s
done

# A target should provide its detect_hardware function.
# On success, return 0 with the following values set.
# return 1 if it wants to use auto_detect
[ "$AUTO" != "1" ] && detect_hardware && FOUND=1

[ -n "$INSTALL" ] && do_install

load_modules
mount_data
mount_sdcard
setup_tslib
setup_dpi
post_detect

```

这里省略debug逻辑代码。  

1. 在 `/src` 中为 `mnt/$src` 创建软链接，为 `android/system` 在 根目录`/` 下创建软链接，为 上级目录下的lib/firmware和 lib/modules 在 /lib 中创建软链接。
2. 如果是 `$INSTALL`，则将 install.img 解压缩到根目录 `/`。
3. 如果 `/system/bin/ln` 存在且可执行，并且设置了 `$BUSYBOX`，操作忽略。
4. 将 /scripts 和 /src/scripts 中的脚本读取并在当前shell执行。
5. 如果传入了 `$INSTALL` 变量，则执行 do_install。只有安装模式才会调用，正常启动不会调用此函数。
`do_scripts` 有两处定义，在 initrd/scripts/1-install 和 install/scripts/1-install，由于解压缩后者会覆盖前者，因此会调用 install镜像中的函数。
6. 接下来调用了若干函数来挂载分区，加载模块。函数定义在 initrd/scripts的 0-auto-detect、2-mount、3-tslib、4-dpi。

```sh
if [ 0$DEBUG -gt 1 ]; then
	echo -e "\nUse Alt-F1/F2/F3 to switch between virtual consoles"
	echo -e "Type 'exit' to enter Android...\n"

	debug_shell debug-late
	SETUPWIZARD=${SETUPWIZARD:-0}
fi

[ "$SETUPWIZARD" = "0" ] && echo "ro.setupwizard.mode=DISABLED" >> default.prop

[ -n "$DEBUG" ] && SWITCH=${SWITCH:-chroot}

# We must disable mdev before switching to Android
# since it conflicts with Android's init
echo > /proc/sys/kernel/hotplug

export ANDROID_ROOT=/system

exec ${SWITCH:-switch_root} /android /init

# avoid kernel panic
while :; do
	echo
	echo '	Android-x86 console shell. Use only in emergencies.'
	echo
	debug_shell fatal-err
done
```

## 设置 `ANDROID_ROOT`执行 ramdisk.img 中的 init

1. 在启动Android前将 `mdev` 取消 
2. 设置环境变量 `ANDROID_ROOT`
3. 启动Android初始化程序，执行的是 ramdisk.img 中的 init 二进制文件。
4. 到此，第一阶段的启动工作完成。

# linux命令

## busybox

    BusyBox v1.22.1  (2021-03-22 17:51 +0800) multi-call binary.
    BusyBox is copyrighted by many authors between 1998-2012.
    Licensed under GPLv2. See source distribution for detailed
    copyright notices. Merged for bionic by tpruvot@github
    
    Usage: busybox [function [arguments]...]
       or: busybox --list[-full]
       or: busybox --install [-s] [DIR]
       or: function [arguments]...
    
            BusyBox is a multi-call binary that combines many common Unix
            utilities into a single executable.  Most people will create a
            link to busybox for each function they wish to use and BusyBox
            will act like whatever it was invoked as.
    
    Currently defined functions:
            [, [[, adjtimex, arp, ash, awk, base64, basename, bbconfig, blkid,
            blockdev, brctl, bunzip2, bzcat, bzip2, cal, cat, catv, chattr, chcon,
            chgrp, chmod, chown, chroot, chvt, clear, cmp, comm, cp, cpio, crond,
            crontab, cut, date, dc, dd, deallocvt, depmod, devmem, df, diff,
            dirname, dmesg, dnsd, dos2unix, du, echo, ed, egrep, env, expand, expr,
            false, fbset, fbsplash, fdisk, fgconsole, fgrep, find, findfs,
            flash_lock, flash_unlock, flashcp, flock, fold, free, freeramdisk,
            fstrim, fsync, ftpget, ftpput, fuser, getenforce, getopt, getsebool,
            grep, groups, gunzip, gzip, halt, head, hexdump, hwclock, id, ifconfig,
            inetd, insmod, install, ionice, iostat, ip, kill, killall, killall5,
            less, ln, losetup, ls, lsattr, lsmod, lsof, lsusb, lzcat, lzma, lzop,
            lzopcat, man, matchpathcon, md5sum, mesg, mkdir, mkdosfs, mke2fs,
            mkfifo, mkfs.ext2, mkfs.vfat, mknod, mkswap, mktemp, modinfo, modprobe,
            more, mount, mountpoint, mpstat, mv, nanddump, nandwrite, nbd-client,
            nc, netstat, nice, nmeter, nohup, nslookup, ntpd, od, openvt, patch,
            pgrep, pidof, ping, pipe_progress, pkill, pmap, poweroff, printenv,
            printf, ps, pstree, pwd, pwdx, rdate, rdev, readlink, realpath, reboot,
            renice, reset, resize, restorecon, rev, rm, rmdir, rmmod, route,
            run-parts, runcon, rx, sed, selinuxenabled, seq, sestatus, setconsole,
            setenforce, setfiles, setkeycodes, setsebool, setserial, setsid, sh,
            sha1sum, sha256sum, sha3sum, sha512sum, sleep, smemcap, sort, split,
            stat, strings, stty, sum, swapoff, swapon, switch_root, sync, sysctl,
            tac, tail, tar, taskset, tee, telnet, telnetd, test, tftp, tftpd, time,
            timeout, top, touch, tr, traceroute, true, ttysize, tune2fs, umount,
            uname, uncompress, unexpand, uniq, unix2dos, unlzma, unlzop, unxz,
            unzip, uptime, usleep, uudecode, uuencode, vi, watch, wc, wget, which,
            whoami, xargs, xz, xzcat, yes, zcat



## `bootable/newinstaller/initrd/init`

### `$?` 与 `&&` 或 `||`

`$?` (命令回传值)

`cmd1 && cmd2`: 

    1. 若 cmd1 运行完毕且正确运行($?=0)，则开始运行 cmd2。
    2. 若 cmd1 运行完毕且为错误 ($?≠0)，则 cmd2 不运行。

`cmd1 || cmd2`
    
    1. 若 cmd1 运行完毕且正确运行($?=0)，则 cmd2 不运行。
    2. 若 cmd1 运行完毕且为错误 ($?≠0)，则开始运行 cmd2。

例子：`ls /tmp/vbirding && echo "exist" || echo "not exist"`

### `chroot`

chroot - run command or interactive shell with special root directory，切换新的目录为跟目录 `/`，也可以继续在新目录执行命令。
```
chroot [OPTION] NEWROOT [COMMAND [ARG]...]
```

### `switch_root`

switch_root - switch to another filesystem as the root of the mount tree

```
switch_root newroot init [arg...]
```

其中 `newroot` 是实际的根文件系统的挂载目录，执行 `switch_root` 命令前需要挂载到系统中；`init` 是实际根文件系统的init程序的路径，一般是 `/sbin/init`.

### `mount`

```
mount [-t 文件系统类型] [-L Label名] [-o 额外选项]  [-n]  装置文件名  挂载点目录 
```

- `-t`  ：与 mkfs 的选项非常类似的，可以加上文件系统种类来指定欲挂载的类型。
      常见的 Linux 支持类型有：ext2, ext3, vfat, reiserfs, iso9660(光盘格式),
      nfs, cifs, smbfs(此三种为网络文件系统类型),tmpfs
- `-n`  ：在默认的情况下，系统会将实际挂载的情况实时写入 /etc/mtab 中，以利其他程序
      的运行。但在某些情况下(例如单人维护模式)为了避免问题，会刻意不写入。
      此时就得要使用这个 -n 的选项了。
- `-L`  ：系统除了利用装置文件名 (例如 /dev/hdc6) 之外，还可以利用文件系统的标头名称
      (Label)来进行挂载。最好为你的文件系统取一个独一无二的名称吧！
- `-o`  ：后面可以接一些挂载时额外加上的参数！比方说账号、密码、读写权限等：
      `ro`, `rw`:       挂载文件系统成为只读(ro) 或可擦写(rw)
      `async`, `sync`:  此文件系统是否使用同步写入 (sync) 或异步 (async) 的内存机制，请参考文件系统运行方式。默认为 async。
      `auto`, `noauto`: 允许此 partition 被以 mount -a 自动挂载(auto)
      `atime`, `noatime`：更新 inode access times
      `dev`, `nodev`:   是否允许此 partition 上，可创建装置文件？ dev 为可允许
      `suid`, `nosuid`: 是否允许此 partition 含有 suid/sgid 的文件格式？
      `exec`, `noexec`: 是否允许此 partition 上拥有可运行 binary 文件？
      `user`, `nouser`: 是否允许此 partition 让任何使用者运行 mount ？一般来说，
                    mount 仅有 root 可以进行，但下达 user 参数，则可让
                    一般 user 也能够对此 partition 进行 mount 。
      `defaults`:     默认值为：rw, suid, dev, exec, auto, nouser, and async
      `remount`:      重新挂载，这在系统出错，或重新升级参数时，很有用！

- `--move`一个挂载点到另一个挂载点：`mount --move ./bind/ /mnt`
- `--bind` 将源目录绑定到目的目录。

#### loop device

在类 UNIX 系统里，loop 设备是一种伪设备(pseudo-device)，或者也可以说是仿真设备。它能使我们像**块设备**一样访问一个文件。
loop block devices: `/dev/loop[0..N]`，`N`具体个数和内核配置有关，一般为8个。loop control device： `/dev/loop-cotrol`。
在使用之前，一个 loop 设备必须要和一个文件进行连接。这里的文件格式为CD 或 DVD 的 ISO 光盘镜像文件或者是软盘(硬盘)的 `*.img `镜像文件。通过这种 loop mount (回环mount)的方式，这些镜像文件就可以被 mount 到当前文件系统的一个目录下。

    

需要用到loop device的最常见的场景是mount一个ISO文件:

1. 创建空的磁盘镜像文件，这里创建一个1.44M的软盘：
```sh
dd if=/dev/zero of=floppy.img bs=512 count=2880
```

2. 使用 losetup将磁盘镜像文件虚拟成快设备：
```sh
losetup /dev/loop1 floppy.img
```
3. 挂载块设备：
```sh
mount /dev/loop0 /tmp
```
经过上面的三步之后，我们就可以通过/tmp目录，像访问真实快设备一样来访问磁盘镜像文件floppy.img。

卸载loop设备：
```sh
umount /tmp
losetup -d /dev/loop1
```

loop device另一种常用的用法是虚拟一个硬盘，比如我想尝试下btrfs这个文件系统，但系统中目前的所有分区都已经用了，里面都是有用的数据，不想格式化他们。

[Linux mount （第一部分）](https://segmentfault.com/a/1190000006878392)

### `losetup`

`losetup`: set up and control loop devices。

losetup命令 用来设置循环设备。循环设备可把文件虚拟成块设备，籍此来模拟整个文件系统，让用户得以将其视为硬盘驱动器，光驱或软驱等设备，并挂入当作目录来使用。

```
   Get info:

        losetup loopdev

        losetup -l [-a]

        losetup -j file [-o offset]

   Delete loop:

        losetup -d loopdev...

   Delete all used loop devices:

        losetup -D

   Print name of first unused loop device:

        losetup -f

   Set up a loop device:

        losetup [-o offset] [--sizelimit size]
                [-Pr] [--show] -f|loopdev file

   Resize loop device:

        losetup -c loopdev
```

    -a 显示所有循环设备的状态。
    -d 卸除设备。
    -e <加密选项> 启动加密编码 。
    -f 寻找第一个未使用的循环设备。
    -o <偏移量>设置数据偏移量，单位是字节。

```
x86_64:/ # losetup /dev/block/loop0
/dev/block/loop0: [0801]:868358 (/system/etc/houdini7_y.sfs)
x86_64:/ # losetup /dev/block/loop1
/dev/block/loop1: [0801]:934493 (/data/arm/houdini7_z.sfs)
```

### `mountpoint`

查看目录是否为挂载点。

+ `-q`	不打印任何信息
+ `-d`	打印文件系统的主设备号和次设备号
+ `-x`	打印块数设备的主设备号和次设备号

### `zcat`

`gzip`、`gunzip`、`zcat` 用于压缩、解压缩文件
```
gzip [ -acdfhlLnNrtvV19 ] [-S suffix] [ name ...  ]
gunzip [ -acfhlLnNrtvV ] [-S suffix] [ name ...  ]
zcat [ -fhLV ] [ name ...  ]
```

zcat 用于查看压缩文件的内容，而无需对其进行解压缩。 

### `hash`

linux系统下会有一个hash表，每个SHLL独立，当你新开一个SHELL的时候，这个hash表为空，每当你执行过一条命令时，hash表会记录下这条命令的路径，就相当于缓存一样。第一次执行命令shell解释器默认的会从 `$PATH` 路径下寻找该命令的路径，当你第二次使用该命令时，shell解释器首先会查看hash表，没有该命令才会去PATH路径下寻找。

`hash -r`　　//清除hash表，清除的是全部的

### `sleep`

```
sleep NUMBER[SUFFIX]
```
`SUFFIX` 包括 秒 `s` (default) ，分钟`m`，小时`h`，天`d`。

### `setsid`

`setsid` 是重新创建一个session，子进程从父进程继承了 S、进程组ID和打开的终端。子进程如果要脱离父进程，不受父进程控制，可以用此 `setsid` 命令。
使用 ：
```
setsid ping www.baidu.com
```

### `env`

env : run a program in a modified environment

语法：
```
env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]
```

### `blkid`

`blkid` 对设备上采用的文件系统类型进行查询。blkid主要用来对系统的块设备（包括交换分区）所使用的文件系统类型、LABEL、UUID等信息进行查询。

```
x86_64:/ # blkid
/dev/block/loop0: TYPE="squashfs"
/dev/block/loop1: TYPE="squashfs"
/dev/block/sda1: LABEL="Android-x86" UUID="033e8fc7-4cfe-9454-bc59-df7329ca862d" TYPE="ext4"
```

UUID是一个标帜系统中的存储设备的字符串，其目的是帮助使用者唯一的确定系统中的所有存储设备，不管它们是什么类型。它可以标识DVD驱动器，USB存储设备以及你系统中的硬盘设备等。
UUID是真正唯一标识符，设备名称并非总是不变。

`cat /proc/partitions`
查询设备分区和分区大小(KiB)。

### `pv`

pv命令 Pipe Viewer 的简称，由Andrew Wood 开发。意思是通过管道显示数据处理进度的信息。这些信息包括已经耗费的时间，完成的百分比(通过进度条显示)，当前的速度，全部传输的数据，以及估计剩余的时间。

```
# Debian 系的操作系统，如 Ubuntu
sudo apt-get install pv

# RedHat系的则这样：
yum install pv
```

# 参考

+ 自 《Android System Programming》 一书的 《Debugging the Boot Up Process Using a Customized ramdisk》一章的《The Android-x86 start up process》一节。
+ [android-x86 install安装流程注解](https://blog.csdn.net/renshuguo123723/article/details/121293263)
+ [Android x86镜像分析](https://www.cnblogs.com/jjxxjnzy/archive/2013/10/14/3368068.html)
+ [Android x86 镜像分析之二](https://www.cnblogs.com/jjxxjnzy/p/3369949.html)
+ [Android x86 镜像分析之三](https://www.cnblogs.com/jjxxjnzy/p/3370125.html)
+ [Android x86 镜像分析之四](https://www.cnblogs.com/jjxxjnzy/p/3374120.html)
+ [r11-x86分支的持续更新的 bootable_newinstaller](https://github.com/BlissRoms-x86/bootable_newinstaller)
+ [Android x86 Initrd 脚本的变量分析](https://github.com/dbergloev/android-initrd)
