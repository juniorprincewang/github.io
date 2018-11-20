---
title: Linux od 命令
date: 2018-11-20 10:08:17
tags:
- od
categories:
- linux
---
linux中的`od`命令用于将文件内容以八进制或其他进制的编码方式显示。
<!-- more -->

`od` 命令用于输出文件的八进制、十六进制或其它格式编码的字节，通常用于显示或查看文件中不能直接显示在终端的字符。

# 参数

```
-A<RADIX>,--address-radix=RADIX：选择以何种基数表示地址偏移；偏移地址显示基数<RADIX>有：d for decimal, o for octal, x for hexadecimal or n for none。
-j<BYTES>,--skip-bytes=BYTES：跳过指定数目的字节；
-N,--read-bytes=BYTES：输出指定字节数；
-S<BYTES>, --strings[=BYTES]：输出长度不小于指定字节数的字符串；
-v,--output-duplicates：输出时不省略重复的数据； 
-w<BYTES>,--width=<BYTES>：设置每行显示的字节数，od默认每行显示16字节。如果选项--width不跟数字，默认显示32字节；
-t<TYPE>，--format=TYPE：指定输出格式，格式包括a、c、d、f、o、u和x，各含义如下：
  a：具名字符；
  c：可打印的字符或者反斜杠；
  d[SIZE]：十进制，正负数都包含，SIZE字节组成一个十进制整数；
  f[SIZE]：浮点，SIZE字节组成一个浮点数；
  o[SIZE]：八进制，SIZE字节组成一个八进制数；
  u[SIZE]：无符号十进制，只包含正数，SIZE字节组成一个无符号十进制整数；
  x[SIZE]：十六进制，SIZE字节为单位以十六进制输出，即输出时一列包含SIZE字节。

```
`<BYTES>` 可以是 `0x`或 `0X` 开头的十六进制数，也可以是其他计量单位开头的数：`b      512`,`KB     1000`, `K      1024`, `MB     1000*1000`, `M      1024*1024` 还有 `G`, `T`, `P`, `E`, `Z`, `Y`。

# 例子

+ 以十六进制输出，每列输出一字节。
```
od -tx1 testfile
```
+ 以十六进制显示的同时显示原字符。
```
# echo linux | od -t cx1
0000000   l   i   n   u   x  \n
         6c  69  6e  75  78  0a
0000006
```

# 参考文献
1. [Linux命令（2）——od命令](https://blog.csdn.net/K346K346/article/details/54177989)
2. [od命令](http://man.linuxde.net/od)