---
title: cmake-grammar
date: 2021-06-02 11:32:27
tags:
- cmake
categories:
---

总结CMakeLists.txt中的语法。  

<!--more-->



# demo

```
cmake_minimum_required(VERSION 3.9)
project(mylib VERSION 1.0.1 DESCRIPTION "mylib description")
include(GNUInstallDirs)
add_library(mylib SHARED src/mylib.c)
set_target_properties(mylib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER api/mylib.h)
configure_file(mylib.pc.in mylib.pc @ONLY)
target_include_directories(mylib PRIVATE .)
install(TARGETS mylib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/mylib.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
```

语法解析：  

[**cmake命令**](https://cmake.org/cmake/help/v3.19/manual/cmake-commands.7.html)可以分为脚本命令与项目命令。  

脚本命令包括一些命令流控制（if、else、elseif、endif、break、continue...）、循环（while、endwhile、foreach、endforeach...）、设置（set、set_property、set_directory_properties...）、cmake相关（cmake_language、cmake_minimum_required...）、查找（find_file、find_library、find_package、find_path、find_program...）等等。  

项目命令是涉及编译链接的命令。包括头文件目录、编译选项、编译生成目标文件、链接等。  

+ 指定cmake最小版本
    +  `cmake_minimum_required(VERSION 3.9)`
+ [`project`](https://cmake.org/cmake/help/v3.19/command/project.html):定义项目名称
    - `project(<PROJECT-NAME>  [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]] ...)`
    - 还可以设置版本VERSION 、项目描述 DESCRIPTION 、编程语言 LANGUAGES 等。
+ [`set`](https://cmake.org/cmake/help/v3.19/command/set.html):设置变量，包括normal、cache和environment。  
    - `set(<variable> <value>... [PARENT_SCOPE])`
+ `OPTION`：Provide an option that the user can optionally select.Provides an option for the user to select as ON or OFF. If no initial <value> is provided, OFF is used. 
    - `option(<variable> "<help_text>" [value])`
+ `string(TOUPPER <string> <output_variable>)`:转换成大写
+ [`configure_file`](https://cmake.org/cmake/help/v3.19/command/configure_file.html):Copy a file to another location and modify its contents.
    - `configure_file(<input> <output>...)`
+ [`inlcude_directories`](https://cmake.org/cmake/help/v3.19/command/include_directories.html)：向编译工程引入头文件目录。 
    - `include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])`
+ `add_subdirectory`:Add a subdirectory to the build.
    - `add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])`
+ [`target_link_libraries`](https://cmake.org/cmake/help/v3.19/command/target_link_libraries.html):当链接目标文件时指定相关的库或者flag。
    - `target_link_libraries(<target> ... <item>... ...)`
+ [`add_library`](https://cmake.org/cmake/help/v3.19/command/add_library.html#command:add_library): 使用指定的源文件添加或生成一个库文件，生成静态或者动态共享库。  
    - `add_library(<name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [source1] [source2 ...])`
+ [`add_executable`](https://cmake.org/cmake/help/v3.19/command/add_executable.html#command:add_executable):使用指定的源文件来**生成可执行文件**
    - `add_executable(<name> [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [source1] [source2 ...])`
+ [`target_include_directories`](https://cmake.org/cmake/help/v3.19/command/target_include_directories.html)：为目标文件编译指定头文件目录，目标文件必须是通过 `add_executable()` 和 `add_library()` 创建的。  
    * `target_include_directories(<target> [SYSTEM] [BEFORE] <INTERFACE|PUBLIC|PRIVATE> [items1...]`
+ `file(GLOB ...)`: 按指定格式搜索文件并将搜索结果存入变量中 `<variable>` 。
    - `file(GLOB <variable> [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS] [<globbing-expressions>...])`
+ [`install`](https://cmake.org/cmake/help/v3.19/command/install.html):在项目安装时候指定规则.
    - `TARGETS` ，安装的目的地
        + `LIBRARY` 指定库被当作library
        + `DESTINATION` 安装目录
    - `FILES` 指定了安装file的规则。用于头文件的安装  
+ [`set_target_properties`](https://cmake.org/cmake/help/v3.19/command/set_target_properties.html):指定properties指示如何编译  
    - `set_target_properties(target1 target2 ... PROPERTIES prop1 value1 prop2 value2 ...)`
    - targets的properties见<https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties>
        + shared library必须指定 [VERSION 和 SOVERSION](https://cmake.org/cmake/help/latest/prop_tgt/VERSION.html)
            - `VERSION`：指定build version
            - `SOVERSION`：指定API version
+ `find_package`:Finds and loads settings from an external project.

+ [CMake内部变量 cmake-variables(7)](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html)  
    * `CMAKE_BUILD_TYPE`: 指定项目的编译类型，Debug，Release等。
    * `PROJECT_BINARY_DIR`：运行cmake命令的目录，通常为 `${PROJECT_SOURCE_DIR}/build` 。
    * `PROJECT_SOURCE_DIR`：当前工程的根目录，`project()`命令所在的目录。
    * `CMAKE_CURRENT_BINARY_DIR`:target 编译目录。
    * `CMAKE_CURRENT_SOURCE_DIR`:当前处理的源文件所在目录。

# 参考资料

系统学习CMake资料：  

+ CMake Practice
+ [learning cmake](https://github.com/Akagi201/learning-cmake)  
+ [Introduction to CMake by Example](http://derekmolloy.ie/hello-world-introductions-to-cmake/)  

其他具体的参考资料：  

+ [How to create a shared library with cmake?](https://stackoverflow.com/questions/17511496/how-to-create-a-shared-library-with-cmake/45843676#45843676)  

每个subdirectory下面都有CMakeLists.txt。  

+ [cmake-add-subdirectory-vs-include](https://stackoverflow.com/a/48510440)  
+ [[CMake] Difference between ADD_SUBDIRECTORY and INCLUDE](https://cmake.org/pipermail/cmake/2007-November/017897.html)  