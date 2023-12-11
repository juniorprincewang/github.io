---
title: androidx86中libhoudini安装
tags:
  - android
categories:
  - - linux
    - androidx86
date: 2023-12-11 20:11:38
---


Androidx86 中若要支持Native层的ARM指令，需要 libhoudini.so 作为NativeBridge 层将 ARM 指令转换成 x86指令。
本文总结32位/64位 libhoudini 的安装。

<!--more-->

# Pre Android 7.1

Androidx86 直到Android7.1提供的 libhoudini.so 都分成 3 种：

    houdini_7_x = (x86 arm translation)
    houdini_7_y = (x86_64 arm translation)
    houdini_7_z = (x86_64 arm64 translation)


libhoudini 文件分xyz三种，x是用32位x86指令集模拟arm32，y是用64位x86_64指令集模拟arm32，z是用64位x86_64指令集模拟arm64，但是 Androidx86 从8开始就没有z包。
因此，只能运行Androidx86 7.1才能跑arm64包。

x86_64 镜像的Androidx86 启用方法：

*   设置手机 `Settings>Apps Compatibility>Enable Native Bridge` 。
    *   打开这个选项的效果只是将系统属性 `persist.sys.nativebridge` 的值从false（0）改成了true（1）
*   Go to the Android console (Alt-F1 or install a terminal emulator)
*   Login as super user / root
*   运行脚本 `/system/bin/enable_nativebridge`

这么做可能会有问题，连接服务器failed wget不停循环。
需要下载的houdini.sfs 的链接访问过慢或失效，需要对url做些修改。

```sh
    if [ -z "$1" ]; then
        if [ "`uname -m`" = "x86_64" ]; then
            v=7_y
            url=http://goo.gl/SBU3is
        else
            v=7_x
            url=http://goo.gl/0IJs40
    else
    	v=7_z
    	url=http://goo.gl/FDrxVN
    fi
```

`/system/bin/enable_nativebridge` 脚本可以传一个参数 "64"，当不传参数时， `$1` 为空。
脚本的工作包括：

1.  下载32位或64位版本的houdini.sfs，并将其挂载到 `/system/lib$1/arm$1` 中，最终在 `/system/lib$1/` 中创建 libhoudini.so 。
2.  往目录 */proc/sys/fs/binfmt\_misc* 下的名为“register”的文件中写入了两串字符串，从而告诉Linux内核，所有使用ARM指令集的可执行和动态库ELF文件都用houdini程序打开，而所有ARM64指令集的可执行和动态库ELF文件都用houdini64程序打开（关于`binfmt_misc`的详细解释，可以参考[《Linux下如何指定某一类型程序用特定程序打开（通过binfmt\_misc）》）](https://blog.csdn.net/roland_sun/article/details/50062295)。

脚本先下载 `7_y` 版本，在下载 `7_z` 版本。\
脚本日志输出在 logcat 中，以 *houdini* 为 tag，最终输出


	houdini enabled
	houdini64 enabled


# After Androidx86 9

Androidx86 官方不支持 ARM64 转x86_64 指令，其他项目提供了方法：

- android9 从 ChromeOS recovery images 中提取出来 libhoudini [android_vendor_google_chromeos-x86](https://github.com/supremegamers/android_vendor_google_chromeos-x86)。
- Android-r11 从 Microsoft WSA 11 image 中提取出来的两个 Version (x86) = 11.0.1b_y.38765.m 与 Version (x86_64) = 11.0.1b_z.38765.m [Intel's libhoudini for intel/AMD x86 CPU, pulled from Microsoft's WSA 11 image](https://github.com/supremegamers/vendor_intel_proprietary_houdini)


对Androidx86 源码修改以支持libhoudini 的方法：
[android_vendor_google_chromeos-x86](https://github.com/supremegamers/android_vendor_google_chromeos-x86)
的 houdini 库对 Androidx86 p9 的补丁可以参考将 wsa11 houdini 替换
[Prepare for a new houdini repo to copy files](https://github.com/supremegamers/device_generic_common/commit/e4f3b23aa2042a27607e31d15367978e0fae29a2?diff=split)

修改 `device/generic/common` 目录下的

- `BoardConfig.mk` 

添加 sepolicy 与 board/native_bridge_arm_on_x86.mk
```
BOARD_SEPOLICY_DIRS += vendor/google/chromeos-x86/sepolicy

-include vendor/google/chromeos-x86/board/native_bridge_arm_on_x86.mk
```

- `device.mk`

添加 `target/houdini.mk` 与 `target/native_bridge_arm_on_x86.mk`。
```
ifneq ("$(wildcard vendor/google/chromeos-x86/*)","")
    $(call inherit-product-if-exists, vendor/google/chromeos-x86/target/houdini.mk)
    $(call inherit-product-if-exists, vendor/google/chromeos-x86/target/native_bridge_arm_on_x86.mk)
    PRODUCT_SYSTEM_DEFAULT_PROPERTIES += persist.sys.nativebridge=1
endif
```

修改 nativebridge ：

- `nativebridge/Android.mk`

如果使用 `vendor/google/chromeos-x86`，则不设置 `LOCAL_POST_INSTALL_CMD`。
```
 LOCAL_SHARED_LIBRARIES := libcutils libdl liblog
 LOCAL_C_INCLUDES := system/core/libnativebridge/include
 LOCAL_MULTILIB := both
-LOCAL_POST_INSTALL_CMD := $(hide) \
-    rm -rf $(TARGET_OUT)/*/{arm*,*houdini*} {$(TARGET_OUT),$(PRODUCT_OUT)}/vendor/{*/arm*,*/*houdini*}; \
-    mkdir -p $(TARGET_OUT)/{lib/arm,$(if $(filter true,$(TARGET_IS_64_BIT)),lib64/arm64)}; \
-    touch $(TARGET_OUT)/lib/libhoudini.so $(if $(filter true,$(TARGET_IS_64_BIT)),$(TARGET_OUT)/lib64/libhoudini.so)
+ifneq ("$(wildcard vendor/google/chromeos-x86/*)","")
+    include $(BUILD_SHARED_LIBRARY)
+else
+    LOCAL_POST_INSTALL_CMD := $(hide) \
+        rm -rf $(TARGET_OUT)/*/{arm*,*houdini*} {$(TARGET_OUT),$(PRODUCT_OUT)}/vendor/{*/arm*,*/*houdini*}; \
+        mkdir -p $(TARGET_OUT)/{lib/arm,$(if $(filter true,$(TARGET_IS_64_BIT)),lib64/arm64)}; \
+        touch $(TARGET_OUT)/lib/libhoudini.so $(if $(filter true,$(TARGET_IS_64_BIT)),$(TARGET_OUT)/lib64/libhoudini.so)
 
-include $(BUILD_SHARED_LIBRARY)
+    include $(BUILD_SHARED_LIBRARY)
+endif
```

- `nativebridge/nativebridge.mk`

修改 `PRODUCT_PROPERTY_OVERRIDES` 属性 `ro.dalvik.vm.native.bridge=libhoudini.so`
```
-ifneq ($(HOUDINI_PREINSTALL),intel)
+ifneq ("$(wildcard vendor/google/chromeos-x86/*)","")
+PRODUCT_PROPERTY_OVERRIDES := ro.dalvik.vm.native.bridge=libhoudini.so
+else ifneq ($(HOUDINI_PREINSTALL),intel)
 PRODUCT_DEFAULT_PROPERTY_OVERRIDES := ro.dalvik.vm.native.bridge=libnb.so
 
 PRODUCT_PACKAGES := libnb
```

- `init.x86.rc`

启动服务中注释掉 nativebridge 相关命令。

```
+#service nativebridge /system/bin/enable_nativebridge
+    #class main
+    #disabled
+    #oneshot
+    #seclabel u:r:zygote:s0

-on property:persist.sys.nativebridge=1
-    start nativebridge
+#on property:persist.sys.nativebridge=1
+    #start nativebridge
 
-on property:persist.sys.nativebridge=0
-    stop nativebridge
+#on property:persist.sys.nativebridge=0
+    #stop nativebridge
```


为使android property 支持 arm64 指令，满足以下查询：

	ro.product.cpu.abi=x86_64
	ro.product.cpu.abilist=x86_64,x86,armeabi-v7a,armeabi
	ro.product.cpu.abilist32=x86,armeabi-v7a,armeabi
	ro.product.cpu.abilist64=x86_64

需要重新覆盖写 ABI list，修改 `vendor/google/chromeos-x86/board/native_bridge_arm_on_x86.mk`


```
ifeq ($(TARGET_ARCH),x86_64)
    # TARGET_2ND_CPU_ABI2 := armeabi-v7a
    TARGET_CPU_ABI_LIST_64_BIT := $(TARGET_CPU_ABI) $(TARGET_CPU_ABI2) $(NATIVE_BRIDGE_ABI_LIST_64_BIT)
    TARGET_CPU_ABI_LIST_32_BIT := $(TARGET_2ND_CPU_ABI) $(TARGET_2ND_CPU_ABI2) $(NATIVE_BRIDGE_ABI_LIST_32_BIT)
    TARGET_CPU_ABI_LIST := $(TARGET_CPU_ABI) $(TARGET_CPU_ABI2) $(TARGET_2ND_CPU_ABI) $(TARGET_2ND_CPU_ABI2) $(NATIVE_BRIDGE_ABI_LIST_32_BIT) $(NATIVE_BRIDGE_ABI_LIST_64_BIT)
else
    TARGET_CPU_ABI2 := armeabi-v7a
    TARGET_CPU_ABI_LIST_32_BIT := $(TARGET_CPU_ABI) $(NATIVE_BRIDGE_ABI_LIST_32_BIT)
endif
```

# 参考

- [How to install libhoudini on a custom Android x86 rig](https://stackoverflow.com/questions/49634762/how-to-install-libhoudini-on-a-custom-android-x86-rig)
- [blog: Integrate Houdini to emulator](https://utzcoz.github.io/2020/03/15/Integrate-Houdini-to-emulator.html)
- [failed to download android-8.1-r1 houdini](https://groups.google.com/g/android-x86/c/_sSDez_JppQ/m/WVGy44QkBwAJ)
- [SurfaceGo Android系统折腾笔记](https://zhuanlan.zhihu.com/p/165988357)
- [如何打开Android X86对houdini的支持](https://blog.csdn.net/Roland_Sun/article/details/49735601)\
- [the default ARM translation layer for x86](https://github.com/Rprop/libhoudini)\
- [Android-X86集成houdini(Arm兼容工具)](https://www.jianshu.com/p/73198c3bfbb1)\
- [VirtualBox Android x86 踩坑记录](https://melty.land/blog/android-x86)
- [How to manually install Arm Native Bridge in android x86,手动安装Arm NB](https://github.com/SGNight/Arm-NativeBridge)
