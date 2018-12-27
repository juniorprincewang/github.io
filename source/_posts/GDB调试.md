---
title: GDB调试
date: 2017-08-13 16:01:14
tags:
- gdb
- pwn
---

针对GDB总结的知识点，包括插件和命令。

<!--more-->

最好的学习方法是查看GDB的说明文档，采用命令`man gdb`查看启动项和参数。
进入gdb，采用命令`help`查看。
	List of classes of commands:

	aliases -- Aliases of other commands
	breakpoints -- Making program stop at certain points
	data -- Examining data
	files -- Specifying and examining files
	internals -- Maintenance commands
	obscure -- Obscure features
	running -- Running the program
	status -- Status inquiries
	support -- Support facilities
	tracepoints -- Tracing of program execution without stopping the program
	user-defined -- User-defined commands

	Type "help" followed by a class name for a list of commands in that class.
	Type "help all" for the list of all commands.
	Type "help" followed by command name for full documentation.
	Type "apropos word" to search for commands related to "word".
	Command name abbreviations are allowed if unambiguous.
gdb命令很多，这是gdb按照类别列出的命令。help命令只是例出gdb的命令种类，如果要看种类中的命令，可以使用`help`命令，如：`help breakpoints`，查看设置断点的所有命令。也可以直接`help`来查看命令的帮助。 


# 常用命令


## break

缩写为`b`。可以使用’行号‘、‘函数名称’、‘执行地址’等方式指定断点位置。
其中在函数名称前面加`*`符号表示将断点设置在‘由编译器生成的prolog代码处’。

	b <行号>
	b <函数名称>
	b *<函数名称>
	b *<代码地址>
	d [编号]
	d: Delete breakpoint的简写，删除指定编号的某个断点，或删除所有断点。断点编号从1开始递增。
```
(gdb) b 8
(gdb) b main
(gdb) b *main
(gdb) b *0x804835c
(gdb) d
(gdb) disable b 1 #禁止第一个断点
(gdb) enable b 1 # 允许使用第一个断点
```

## 程序运行参数
`set args `可指定运行时参数。（如：set args 10 20 30 40 50） 
`show args `命令可以查看设置好的运行参数。 

## run

运行调试的程序，缩写为`r`。
```
(gdb) r
```

## step & next

`step`: 执行一行源程序代码，如果此行代码中有函数调用，则进入该函数，相当于其它调试器中的`Step Into`(单步跟踪进入)，缩写为`s`；
`next`: 执行一行源程序代码，此行代码中的函数调用也一并执行，相当于其它调试器中的`Step Over`(单步跟踪)，缩写为`n`。
这两个命令必须在有源代码调试信息的情况下才可以使用（GCC编译时使用“-g”参数）。

## stepi & nexti

`stepi`，`nexti`与`step`,`next`功能相近，只不过是执行的是汇编指令。

## finish

`finish`继续执行程序，直到当前被调用的函数结束，如果该函数有返回值，把返回值也打印到控制台

## info

`i`是`info`的简写，用于显示各类信息，详情请查阅`help i`。

1. `i r`命令显示寄存器中的当前值———`i r`即`Infomation Register`。

显示任意一个指定的寄存器值：`i r eax`

2. `info b`

列出所有的断点。



## print

打印表达式的值，除了程序中的变量外，还可以是程序函数的调用，数据结构和其他它复杂对象，历史纪录的值（$是最后一个历史纪录变量，$num是倒数第num个历史纪录变量），人为数组。

## examine

简写`x`，用于查看内存地址的值`examine memory`。`x`命令的语法如下：

```
x/FMT ADDRESS
```
- `ADDRESS`是内存地址的表达式，比如0xff340112
- `FMT`由3个可选参数组成。分别为内存长度`count`,显示格式`format`,字节大小`size`。
	+ `format`: `o`表示8进制，`x`表示16进制，`d`表示10进制，`u`表示无符号16进制，`t`二进制,`f`浮点数，`c`字符，`i`指令，`a`地址，`s`字符串，`z`16进制，左侧补0对齐。
	+ `size`：`b`字节，`h`半字,`w`字,`g`8字节。
比如：命令：`x/3xh 0x54320` 表示，从内存地址0x54320读取内容，h表示以双字节为一个单位，3表示三个单位，x表示按十六进制显示。 

## backtrace

显示程序的调用栈信息，可以用`bt`缩写

## quit

退出GDB，缩写为`q`。

## attach

`attach process-id`: 在GDB状态下，开始调试一个正在运行的进程，其进程ID为process-id

## set 

`set variable`将值赋予变量

## whatis & ptype

识别数组或数据的类型，`ptype`比`whatis`功能更强，它可以提供一个结构的定义。



# 参考网站
[1] [GDB十分钟教程](http://blog.csdn.net/liigo/article/details/582231)
[2] [Using GDB to Develop Exploits - A Basic Run Through](https://www.exploit-db.com/papers/13205/)
[3] [比较全面的gdb调试命令](http://blog.csdn.net/dadalan/article/details/3758025)
[4] [1.gdb 调试利器](https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/gdb.html)