---
title: vscode debug 笔记
date: 2023-03-17 19:19:10
tags:
	- vscode
categories:
	- debug
---
vscode 调试C/C++/Rust笔记。
<!-- more -->

## launch.json

调试需要创建配置文件 launch.json，需要的 [`configurations` 字段](https://code.visualstudio.com/docs/editor/debugging#_launchjson-attributes)包括：

+ name
+ type
+ request
+ program
+ args
+ cwd
+ environment
+ setupCommands
+ ...

Rust调试有两种方式：
1. 设置 `program`

先通过 `cargo build` 生成好 *target/debug/forkserver_simple* 可执行文件。
调试器选择 Codelldb，这个插件要安装。
`type` 选择 `lldb`。
`program` 和 `cwd` 的路径通过使用变量 `${workspaceRoot}` 来衔接相对目录。
`args` 填写适当参数。
调试lib.rs会选择此方式。

2. 通过 `cargo` 命令，`bin` 和 `package` 是要调试的二进制程序，见下面第二条配置项。

```
"configurations": [
        {
            "name": "Debug execute forkserver_simple",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceRoot}/fuzzers/forkserver_simple/target/debug/forkserver_simple",
            "args": [ "target/release/program", "corpus/" ],
            "cwd": "${workspaceRoot}/fuzzers/forkserver_simple",
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug executable 'forkserver_simple'",
            "cargo": {
                "args": [
                    "build",
                    "--bin=forkserver_simple",
                    "--package=forkserver_simple"
                ],
                "filter": {
                    "name": "forkserver_simple",
                    "kind": "bin"
                }
            },
            "args": [
                "./target/release/program",
                "./corpus/",
                "-t",
                "1000",
                "-d",
            ],
            "env": {
                "AFL_DEBUG": "1",
            },
            "cwd": "${workspaceFolder}/fuzzers/forkserver_simple"
        },
    ]
```

更详细的文件参考官方 [VSCode Debugging](https://code.visualstudio.com/docs/editor/debugging)

## Environment

我这里碰到两种写法，具体要看vscode支持哪一种。

+ [`environment` 字段](https://stackoverflow.com/a/47446417)：值为向量类型，值里面元素是以`"name"` 为key，`"value"` 为值的字典。
```
"environment": [
	{"name": "DYLD_LIBRATY_PATH", "value": "/Users/x/boost_1_63_0/stage/lib/"}
]
```

+ `env` 字段，直接是 `环境变量名:变量值`：
```
"env": { "NODE_ENV": "development", }
```

## Debug child process via lldb

调试 `fork()` 出的子进程可以通过 console command 完成。
在程序创建子进程之前(`fork` 或者 `Command::new()` in Rust)之前，设置断点，在执行到这个断点的时候，在debug console里输入以下命令，debugger会追踪子进程代码。

```
settings set target.process.follow-fork-mode child
```

这里 `follow-fork-mode` 可以设置 `child` 和 `parent`。[llvm 中添加此命令[lldb] [client] Implement follow-fork-mode](https://github.com/llvm/llvm-project/commit/4a2a947317bf702178bf1af34dffd0d280d49970)

为方便，可以在 launch.json 中配置 启动命令`initCommands`的参数：
```
    "initCommands": [
        "settings set target.process.follow-fork-mode child"
    ]
```


为防止子进程再次创建子进程而debugger继续追踪，可以再进入子进程后设置回 `follow-fork-mode parent`:
```
settings set target.process.follow-fork-mode parent
```

[由浅入深CrosVM（五）—— Crosvm的开发和调试](https://www.owalle.com/2021/03/24/crosvm-develop-debug/) 介绍的console命令不必加 `-exec`，比如输出变量 `p variable`。
[Debugging with GDB fails if Launcher is used](https://github.com/AFLplusplus/LibAFL/issues/148)

## Pretty-printing

下面是C++ 代码的调试 launch.json，
为了让 vector 显示元素，需要加入 `setupCommands` 字段。详情见 [Unable to see elements of std::vector with gcc in VS code](https://stackoverflow.com/questions/56828562/unable-to-see-elements-of-stdvector-with-gcc-in-vs-code)
```
"configurations": [
    {
        "name": "(gdb) Launch",
        "type": "cppdbg",
        "request": "launch",
        "program": "${workspaceFolder}/example.exe",
        "args": [],
        "stopAtEntry": false,
        "cwd": "${workspaceFolder}",
        "environment": [],
        "externalConsole": true,
        "MIMode": "gdb",
        "miDebuggerPath": "C:\\MinGW\\bin\\gdb.exe",
        "preLaunchTask": "echo",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            }
        ],
    }
]
```

## Memory view

可以在 VARIABLES 窗口view中 将鼠标悬浮在变量上方，点击右侧显示有 **view binary data** 的二进制图标即可查看或编辑。
对于数组变量，需要点击首个元素元素的 **view binary data**。

[Does VS Code have a memory viewer and/or a disassembler for C++ extension?](https://stackoverflow.com/a/38616728)

## Disassembly

调试汇编代码： 在调试的源码中右键选择 **Open Disassembly View**。
[Visual Studio Code C++ July 2021 Update: Disassembly View, Macro Expansion and Windows ARM64 Debugging](https://devblogs.microsoft.com/cppblog/visual-studio-code-c-july-2021-update-disassembly-view-macro-expansion-and-windows-arm64-debugging/#disassembly-view)

## Macro

可以选中 宏 变量，在弹出的 lightbulb 中选择 **Inline macro**，即可在源码中展开宏。
