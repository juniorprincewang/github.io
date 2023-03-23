---
title: ANGLE for Android
tags:
  - opengles
categories:
  - graphics
date: 2023-03-13 10:13:37
---


本文总结使用Google ANGLE库用作Android OpenGL ES driver。
<!--more -->


ANGLE 库项目仓库 <https://chromium.googlesource.com/angle/angle> 目前支持将OpenGL ES 3.0 转换成 Vulkan, desktop OpenGL, OpenGL ES, Direct3D 11, Metal实现，而 OpenGL ES 3.1 只支持转换成 Vulkan, desktop OpenGL, OpenGL ES实现，	OpenGL ES 3.2 转换成 Vulkan, desktop OpenGL, OpenGL ES 还在实现中。 Vulkan作为后端渲染器是支持Windows、Linux、Android等平台最多的实现。

下面总结在Android中用 ANGLE 库作为OpenGL ES driver步骤，**构建需要在 Linux 中**，最终构建包含 ANGLE 库的 ANGLE APK。
> Important note: Android builds currently require Linux.

# 编译构建ANGLE

安装 depot_tools

```
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
```

将 depot_tools 添加到 `PATH` 路径中。 (最好加入 `~/.bashrc` or `~/.zshrc`)

```
export PATH="$PATH:/path/to/depot_tools"
```

获取 angle 仓库代码，这里不[从 chromium 构建](https://android.googlesource.com/platform/packages/modules/ANGLE/+/61dfd992f57551663ba4e31cb4f9dabfef2db997/Readme.md)，直接下载 ANGLE 库。
```
git clone https://chromium.googlesource.com/angle/angle
```

生成  `.gclient` 文件

```
cd angle
python scripts/bootstrap.py

```

确认下 `.gclient` 文件内容需要有Android依赖 `target_os = [ 'android' ]`。
```
solutions = [
  { "name"        : '.',
    "url"         : 'https://chromium.googlesource.com/angle/angle.git',
    "deps_file"   : 'DEPS',
    "managed"     : False,
    "custom_deps" : {
    },
    "custom_vars": {},
  },
]
target_os = [ 'android' ]

```

下载所有源码和packages。
```
gclient sync
```

打开 Android Release 构建的 GN args 的文本输入框。
```
gn args out/Android
```

保存以下配置
```
target_os = "android"
target_cpu = "arm64"
is_component_build = false
is_debug = false
angle_assert_always_on = true   # Recommended for debugging. Turn off for performance.
```

Building ANGLE for Android
```
autoninja -C out/Android
```

# Using ANGLE as the Android OpenGL ES driver

自  Android 10 (Q) 起，可以将 ANGLE 作为 OpenGL ES driver。
**Important Note**: ANGLE需要在 Debuggable APPs 或者 Root权限 下调用。

Android APPs 可以一次一个、分组或全局选择加入ANGLE。App必须由 Java 运行时启动，因为库才能在安装的package中发现，这也意味着ANGLE不能被native 二进制或者SurfaceFlinger使用。

Building the ANGLE APK，貌似上一步已经生成好了，生成文件是 `out/Android/apks/AngleLibraries.apk` 。

```
autoninja -C out/Android angle_apks
```

通过adb安装ANGLE APK
```
adb install -r -d --force-queryable out/Android/apks/AngleLibraries.apk
```

验证安装成功，查询的包名： `org.chromium.angle`
```
adb shell pm path org.chromium.angle
```

对于 **debuggable app** 或者 **root users** 而言，选择 ANGLE 作为 OpenGL ES driver：

```
adb shell settings put global angle_debug_package org.chromium.angle
```

ANGLE driver 的可选项可通过查询 `angle_gl_driver_selection_values` 获得：

+ `angle` : Use ANGLE.
+ `native` : Use the native OpenGL ES driver.
+ `default` : Use the default driver. This allows the platform to decide which driver to use.

将ANGLE 设置为 当个 OpenGL ES app
```
adb shell settings put global angle_gl_driver_selection_pkgs <package name>
adb shell settings put global angle_gl_driver_selection_values <driver>
```

将ANGLE 设置多个 OpenGL ES app
```
adb shell settings put global angle_gl_driver_selection_pkgs <package name 1>,<package name 2>,<package name 3>,...
adb shell settings put global angle_gl_driver_selection_values <driver 1>,<driver 2>,<driver 3>,...
```

将 ANGLE 设置为 所有 OpenGL ES app 使用，只有 root user 才能设置。
Enable:
```
adb shell settings put global angle_gl_driver_all_angle 1
```
Disable:
```
adb shell settings put global angle_gl_driver_all_angle 0
```


检查是否设置成功:

```
logcat -d | grep ANGLE
```

app 成功载入 ANGLE 库：

    V GraphicsEnvironment: ANGLE developer option for <package name>: angle
    I GraphicsEnvironment: ANGLE package enabled: org.chromium.angle
    I ANGLE   : Version (2.1.0.f87fac56d22f), Renderer (Vulkan 1.1.87(Adreno (TM) 615 (0x06010501)))

实际启动日志，可以看到有一些warning：

    I ANGLE   : Version (2.1.20757 git hash: 9c29f84ce25e), Renderer (Vulkan 1.1.177 (Mali-G610 MC6 (0xA8670000)))
    W libEGL  : ANGLE Warn:Surface.cpp:421 (setSwapBehavior):        ! Unimplemented: setSwapBehavior(../../src/libANGLE/Surface.cpp:421)
    W ANGLE   : WARN: Surface.cpp:421 (setSwapBehavior):     ! Unimplemented: setSwapBehavior(../../src/libANGLE/Surface.cpp:421)
    W libEGL  : ANGLE Warn:Surface.cpp:421 (setSwapBehavior):        ! Unimplemented: setSwapBehavior(../../src/libANGLE/Surface.cpp:421)
    W ANGLE   : WARN: Surface.cpp:421 (setSwapBehavior):     ! Unimplemented: setSwapBehavior(../../src/libANGLE/Surface.cpp:421)
    
如果未设置成功，App 载入 ANGLE 库失败：

    V GraphicsEnvironment: ANGLE developer option for <package name>: angle
    E GraphicsEnvironment: Invalid number of ANGLE packages. Required: 1, Found: 0
    E GraphicsEnvironment: Failed to find ANGLE package.

缺少 ANGLE 库的日志：

    V GraphicsEnvironment: ANGLE Developer option for '<package name>' set to: 'angle'
    V GraphicsEnvironment: ANGLE developer option for <package name>: angle
    I GraphicsEnvironment: ANGLE debug package enabled: org.chromium.angle
    W GraphicsEnvironment: ANGLE debug package 'org.chromium.angle' not installed


Clean Up:
```
adb shell settings delete global angle_debug_package
adb shell settings delete global angle_gl_driver_all_angle
adb shell settings delete global angle_gl_driver_selection_pkgs
adb shell settings delete global angle_gl_driver_selection_values
```

参考：
+ [ANGLE for Android instructions](https://chromium.googlesource.com/angle/angle/+/HEAD/doc/DevSetupAndroid.md)