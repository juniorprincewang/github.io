---
title: Linux下查看二进制文件
date: 2018-05-14 11:50:44
tags:
- xxd
- vim
categories:
- linux
---
Linux 平台下想要查看二进制文件，可以通过 `xxd`、 `hexdump` 或者通过 `VIM` 与 `xxd` 结合使用。

<!-- more -->

# xxd

xxd命令为给定的标准输入或者文件做一次十六进制的输出，它也可以将十六进制输出转换为原来的二进制格式。

## 选项

> -b	用二进制显示一个bit，而不是十六进制
> -r	以十六进制作为输入，二进制作为输出
> -s [+][-]seek	从<seek>字节开始。+ -分别表示相对于文件的开头和结尾
> -seek offset	从offset数值开始显示
> -g 输出显示中以组为单位每组的字节数，默认为2。
> -c 每行显示的列数。
> -i 出为c包含文件的风格，如 0x7f
> -u 字节大写

# hexdump

一般用来查看“二进制”文件的十六进制编码，但实际上它能查看任何文件，而不只限于二进制文件。

## 选项

> `-n length 只格式化输入文件的前length个字节。`
> `-C 输出规范的十六进制和ASCII码。`
> -b 单字节八进制显示。
> -c 单字节字符显示。
> -d 双字节十进制显示。
> -o 双字节八进制显示。
> -x 双字节十六进制显示。
> `-s 从偏移量开始输出。`
> -e 指定格式字符串，格式字符串包含在一对单引号中，格式字符串形如：'a/b "format1" "format2"'。


# VIM

`vim -b a.out`

在用VIM打开二进制文件时，需要使用 `-b` 选项。 `-b` 选项是告诉 `vim` 打开的是一个二进制文件，不指定的话，会在后面加上 `0x0a` ，即一个换行符。

在命令行模式下，输入 `:%!xxd` 就可以将二进制文件通过管道传输给 `xxd` ，当然还可以添加选项，如 `:%!xxd -g 1` 。

# 参考

[1] [在Linux下使用vim配合xxd查看并编辑二进制文件](http://www.cnblogs.com/killkill/archive/2010/06/23/1763785.html)
[2] [hexdump命令](http://man.linuxde.net/hexdump)
[3] [linux 命令 xxd linux下查看二进制文件 ](http://fancyxinyu.blog.163.com/blog/static/18232136620111183019942/)