---
title: kvm虚拟机磁盘扩容
date: 2018-04-13 15:02:18
tags:
- kvm
- qemu
categories:
- solutions
---

kvm虚拟机的磁盘空间告急，怎么有效进行扩容。

<!-- more -->

kvm的磁盘格式分为：raw磁盘格式和qcow2磁盘格式，扩充的思路如下:

假如对raw格式的 centos6.6.img 直接扩展：

```
qemu-img info centos6.6.img
```
	
	image: centos6.6.img
	file format: raw
	virtual size: 16G (17179869184 bytes)
	disk size: 16G

```
qemu-img resize centos6.6.img +10G
```

	image: centos6.6.img
	file format: raw
	virtual size: 26G (27917287424 bytes)
	disk size: 16G

开启虚拟机，然后查看磁盘空间。

```
df -h
```
	
	Filesystem            Size  Used Avail Use% Mounted on
	/dev/mapper/VolGroup-lv_root
	                       14G   13G  521M  97% /
	tmpfs                 940M   76K  940M   1% /dev/shm
	/dev/sda1             477M  150M  299M  34% /boot

```
fdisk -l
```

	Disk /dev/sda: 27.9 GB, 27917287424 bytes
	255 heads, 63 sectors/track, 3394 cylinders
	Units = cylinders of 16065 * 512 = 8225280 bytes
	Sector size (logical/physical): 512 bytes / 512 bytes
	I/O size (minimum/optimal): 512 bytes / 512 bytes
	Disk identifier: 0x0003d6ba

	   Device Boot      Start         End      Blocks   Id  System
	/dev/sda1   *           1          64      512000   83  Linux
	Partition 1 does not end on cylinder boundary.
	/dev/sda2              64        2089    16264192   8e  Linux LVM

	Disk /dev/mapper/VolGroup-lv_root: 14.9 GB, 14935916544 bytes
	255 heads, 63 sectors/track, 1815 cylinders
	Units = cylinders of 16065 * 512 = 8225280 bytes
	Sector size (logical/physical): 512 bytes / 512 bytes
	I/O size (minimum/optimal): 512 bytes / 512 bytes
	Disk identifier: 0x00000000


	Disk /dev/mapper/VolGroup-lv_swap: 1715 MB, 1715470336 bytes
	255 heads, 63 sectors/track, 208 cylinders
	Units = cylinders of 16065 * 512 = 8225280 bytes
	Sector size (logical/physical): 512 bytes / 512 bytes
	I/O size (minimum/optimal): 512 bytes / 512 bytes
	Disk identifier: 0x00000000

可见，实际的物理空间 `/dev/sd` 已经扩容为27.9GB，但是真正的使用空间还是16GB。

开始分区并保存。
```
[root@localhost Desktop]# fdisk /dev/sda

WARNING: DOS-compatible mode is deprecated. It's strongly recommended to
         switch off the mode (command 'c') and change display units to
         sectors (command 'u').

Command (m for help): n
Command action
   e   extended
   p   primary partition (1-4)
p
Partition number (1-4): 3
First cylinder (2089-3394, default 2089): 
Using default value 2089
Last cylinder, +cylinders or +size{K,M,G} (2089-3394, default 3394): 
Using default value 3394

Command (m for help): t
Partition number (1-4): 3
Hex code (type L to list codes): 8e
Changed system type of partition 3 to 8e (Linux LVM)

Command (m for help): w
The partition table has been altered!

Calling ioctl() to re-read partition table.

WARNING: Re-reading the partition table failed with error 16: Device or
resource busy.
The kernel still uses the old table. The new table will be used at
the next reboot or after you run partprobe(8) or kpartx(8)
Syncing disks.
```

重启该虚拟机，接下来开始创建物理卷，加入卷组，扩展逻辑卷。

```
[root@localhost Desktop]# pvs
  PV         VG       Fmt  Attr PSize  PFree
  /dev/sda2  VolGroup lvm2 a--  15.51g    0 
[root@localhost Desktop]# vgextend VolGroup /dev/sda
sda   sda1  sda2  sda3  
[root@localhost Desktop]# vgextend VolGroup /dev/sda3
  Physical volume "/dev/sda3" successfully created
  Volume group "VolGroup" successfully extended
[root@localhost Desktop]# lvextend -l +100%FREE /dev/VolGroup/lv_root 
  Size of logical volume VolGroup/lv_root changed from 13.91 GiB (3561
extents) to 23.91 GiB (6120 extents).
  Logical volume lv_root successfully resized
[root@localhost Desktop]# resize2fs /dev/VolGroup/lv_root 
resize2fs 1.41.12 (17-May-2010)
Filesystem at /dev/VolGroup/lv_root is mounted on /; on-line resizing required
old desc_blocks = 1, new_desc_blocks = 2
Performing an on-line resize of /dev/VolGroup/lv_root to 6266880 (4k) blocks.
The filesystem on /dev/VolGroup/lv_root is now 6266880 blocks long.

[root@localhost Desktop]# df -h
Filesystem            Size  Used Avail Use% Mounted on
/dev/mapper/VolGroup-lv_root
                       24G   13G   10G  56% /
tmpfs                 940M  224K  940M   1% /dev/shm
/dev/sda1             477M  150M  299M  34% /boot
```
成功扩容。


参考
[KVM虚拟化笔记（十）------kvm虚拟机扩充磁盘空间](http://blog.51cto.com/liqingbiao/1741244)
