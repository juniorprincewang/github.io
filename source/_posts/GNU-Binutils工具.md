---
title: GNU Binutils工具
date: 2018-05-08 10:35:37
tags:
- linux
- objdump
- readelf
- nm
- c++filt
catogeries:
- linux
---

由于平时经常分析 `ELF` 格式的文件，经常用到 `objdump` 、 `readelf` 、 `nm` 程序，而它们都属于 `GNU Binutils`工具，所以在此简单总结各个程序的用法和参数。
<!--more-->


# binutils

`GNU Binary Utilities` 或 `binutils` 是一整套的编程语言工具程序，用来处理许多格式的目标文件。

|程序|作用|
|--|--|
|as	|汇编器|
|ld	| 连接器|
|gprof	| 性能分析工具程序|
|addr2line |	从目标文件的虚拟地址获取文件的行号或符号|
|ar	| 可以对静态库做创建、修改和取出的操作。|
|c++filt |	解码 C++ 的符号|
|dlltool |	创建Windows 动态库|
|gold	| 另一种连接器|
|nlmconv |	可以转换成NetWare Loadable Module目标文件格式|
|nm	| 显示目标文件内的符号|
|objcopy |	复制目标文件，过程中可以修改|
|objdump |	显示目标文件的相关信息，亦可反汇编|
|ranlib	| 产生静态库的索引|
|readelf |	显示ELF文件的内容|
|size	| 列出总体和section的大小|
|strings |	列出任何二进制档内的可显示字符串|
|strip	| 从目标文件中移除符号|
|windmc	| 产生Windows消息资源|
|windres |	Windows 资源档编译器|


# objdump
查看目标文件或可执行文件的构成工具。

## 选项

|选项|全称|描述|
|----|----|---|
|-a |--archive-headers | 显示档案库的成员信息,类似ls -l将lib*.a的信息列出。 |
|-b bfdname	|--target=bfdname 	|指定目标码格式。这不是必须的，objdump能自动识别许多格式，<br>比如： objdump -b oasys -m vax -h fu.o<br>显示fu.o的头部摘要信息，明确指出该文件是Vax系统下用Oasys编译器生成的目标文件。<br>objdump -i将给出这里可以指定的目标码格式列表。 	|
|-C 	|--demangle |	将底层的符号名解码成用户级名字，除了去掉所开头的下划线之外，还使得C++函数名以可理解的方式显示出来。 |
|-g 	|--debugging |显示调试信息。企图解析保存在文件中的调试信息并以C语言的语法显示出来。仅仅支持某些类型的调试信息。有些其他的格式被readelf -w支持。 |
|-e 	|--debugging-tags	| 类似-g选项，但是生成的信息是和ctags工具相兼容的格式。 |
|-d 	|--disassemble 		| 从objfile中反汇编那些特定指令机器码的section。 |
|-D 	|--disassemble-all 	| 与 -d 类似，但反汇编所有section. |
|-z 	| --prefix-addresses |反汇编的时候，显示每一行的完整地址。这是一种比较老的反汇编格式。 |
|-EB 	| --endian=big 		| 指定目标文件的大端 |
|-EL 	| --endian=little	|	指定目标文件的小端。 |
|-f 	| --file-headers 	| 显示objfile中每个文件的整体头部摘要信息。 |
|-h 	| --section-headers 或者 --headers |显示目标文件各个section的头部摘要信息。 |
|-H 	| --help 	| 简短的帮助信息。 |
|-i 	| --info 	| 显示对于 -b 或者 -m 选项可用的架构和目标格式列表。 |
|-j name| --section=name |	仅仅显示指定名称为name的section的信息 |
|-l 	| --line-numbers 	|用文件名和行号标注相应的目标代码，仅仅和-d、-D或者-r一起使用使用-ld和使用-d的区别不是很大。<br>在源码级调试的时候有用，要求编译时使用了-g之类的调试编译选项。 |
|-m machine | --architecture=machine |指定反汇编目标文件时使用的架构，当待反汇编文件本身没描述架构信息的时候(比如S-records)，这个选项很有用。<br>可以用-i选项列出这里能够指定的架构。 |
|-r 	| --reloc | 显示文件的重定位入口。如果和-d或者-D一起使用，重定位部分以反汇编后的格式显示出来。 |
|-R  	| --dynamic-reloc 	|显示文件的动态重定位入口，仅仅对于动态目标文件意义，比如某些共享库。 |
|-s 	| --full-contents 	| 显示指定section的完整内容。默认所有的非空section都会被显示。 |
|-S 	| --source 	|尽可能反汇编出源代码，尤其当编译的时候指定了-g这种调试参数时，效果比较明显。隐含了-d参数。 |
||--start-address=address |从指定地址开始显示数据，该选项影响-d、-r和-s选项的输出。 |
||--stop-address=address |显示数据直到指定地址为止，该项影响-d、-r和-s选项的输出。 |
|-t 	| --syms 	|	显示文件的符号表入口。类似于nm -s提供的信息 |
|-T 	| --dynamic-syms 	|显示文件的动态符号表入口，仅仅对动态目标文件意义，比如某些共享库。 |
|-x 	|	--all-headers 	| 显示所可用的头信息，包括符号表、重定位入口。-x 等价于-a -f -h -r -t 同时指定。 |

## 例子

显示目标文件各个段的头部摘要信息： 

```
objdump -h mytest.o 
```

反汇编目标文件的特定机器码段： 
```
objdump -d mytest.o 
```

显示文件的符号表入口，将底层符号解码并表示成用户级别: 
```
objdump -t -C mytest.o 
```

# readelf


`readelf` 用来显示一个或者多个elf格式的目标文件的信息。 这个程序和 `objdump` 提供的功能类似，但是它显示的信息更为具体，并且它不依赖BFD库。

`ELF`文件分三种：
- 可重定位文件:用户和其他目标文件一起创建可执行文件或者共享目标文件,例如lib\*.o文件。 
- 可执行文件：用于生成进程映像，载入内存执行,例如编译好的可执行文件a.out。 
- 共享目标文件：用于和其他共享目标文件或者可重定位文件一起生成elf目标文件或者和执行文件一起创建进程映像，例如lib\*.so文件。 

## 选项

|选项|全称|描述|
|----|----|---|
|-a 	| 	--all | 显示全部信息,等价于 -h -l -S -s -r -d -V -A -I. |
| -h | --file-header | 显示elf文件开始的文件头信息 |
| -l  | --program-headers  |--segments 显示程序头（段头）信息(如果有的话)。 |
|-S 	| --section-headers  | --sections 显示节头信息(如果有的话)。 |
|-g 	| --section-groups 	| 显示节组信息(如果有的话)。 |
| -t 	| --section-details  | 显示节的详细信息(-S的)。 |
|-s 	| --syms 或者  --symbols | 显示符号表段中的项（如果有的话）。 |
|-e 	| --headers | 显示全部头信息，等价于: -h -l -S |
|-n 	| --notes | 显示note段（内核注释）的信息。 |
| -r  	| --relocs | 	 显示可重定位段的信息。 |
|-d 	| --dynamic | 显示动态段的信息。 |
|-A 	| --arch-specific | 显示CPU构架信息。 |
|-D 	| --use-dynamic  | 使用动态段中的符号表显示符号，而不是使用符号段。 |
|-x <number or name> |--hex-dump=<number or name> | 以16进制方式显示指定段内内容。number指定段表中段的索引,或字符串指定文件中的段名。 |

## 例子

读取目标文件形式的elf文件头信息：
```
readelf -h myfile.o 
```

查看一个可执行的elf文件的节信息：
```
readelf -S main 
```

ELF文件的完整输出：
```
$readelf -all a.out
ELF Header:
  Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF32
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              EXEC (Executable file)
  Machine:                           Intel 80386
  Version:                           0x1
  Entry point address:               0x8048330
  Start of program headers:          52 (bytes into file)
  Start of section headers:          4412 (bytes into file)
  Flags:                             0x0
  Size of this header:               52 (bytes)
  Size of program headers:           32 (bytes)
  Number of program headers:         9
  Size of section headers:           40 (bytes)
  Number of section headers:         30
  Section header string table index: 27

Section Headers:
  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            00000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        08048154 000154 000013 00   A  0   0  1
  [ 2] .note.ABI-tag     NOTE            08048168 000168 000020 00   A  0   0  4
  [ 3] .note.gnu.build-i NOTE            08048188 000188 000024 00   A  0   0  4
  [ 4] .gnu.hash         GNU_HASH        080481ac 0001ac 000020 04   A  5   0  4
  [ 5] .dynsym           DYNSYM          080481cc 0001cc 000050 10   A  6   1  4
  [ 6] .dynstr           STRTAB          0804821c 00021c 00004c 00   A  0   0  1
  [ 7] .gnu.version      VERSYM          08048268 000268 00000a 02   A  5   0  2
  [ 8] .gnu.version_r    VERNEED         08048274 000274 000020 00   A  6   1  4
  [ 9] .rel.dyn          REL             08048294 000294 000008 08   A  5   0  4
  [10] .rel.plt          REL             0804829c 00029c 000018 08   A  5  12  4
  [11] .init             PROGBITS        080482b4 0002b4 00002e 00  AX  0   0  4
  [12] .plt              PROGBITS        080482f0 0002f0 000040 04  AX  0   0 16
  [13] .text             PROGBITS        08048330 000330 00018c 00  AX  0   0 16
  [14] .fini             PROGBITS        080484bc 0004bc 00001a 00  AX  0   0  4
  [15] .rodata           PROGBITS        080484d8 0004d8 000011 00   A  0   0  4
  [16] .eh_frame_hdr     PROGBITS        080484ec 0004ec 000034 00   A  0   0  4
  [17] .eh_frame         PROGBITS        08048520 000520 0000c4 00   A  0   0  4
  [18] .ctors            PROGBITS        08049f14 000f14 000008 00  WA  0   0  4
  [19] .dtors            PROGBITS        08049f1c 000f1c 000008 00  WA  0   0  4
  [20] .jcr              PROGBITS        08049f24 000f24 000004 00  WA  0   0  4
  [21] .dynamic          DYNAMIC         08049f28 000f28 0000c8 08  WA  6   0  4
  [22] .got              PROGBITS        08049ff0 000ff0 000004 04  WA  0   0  4
  [23] .got.plt          PROGBITS        08049ff4 000ff4 000018 04  WA  0   0  4
  [24] .data             PROGBITS        0804a00c 00100c 000008 00  WA  0   0  4
  [25] .bss              NOBITS          0804a014 001014 000008 00  WA  0   0  4
  [26] .comment          PROGBITS        00000000 001014 00002a 01  MS  0   0  1
  [27] .shstrtab         STRTAB          00000000 00103e 0000fc 00      0   0  1
  [28] .symtab           SYMTAB          00000000 0015ec 000410 10     29  45  4
  [29] .strtab           STRTAB          00000000 0019fc 0001f9 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings)
  I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
  O (extra OS processing required) o (OS specific), p (processor specific)

There are no section groups in this file.

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  PHDR           0x000034 0x08048034 0x08048034 0x00120 0x00120 R E 0x4
  INTERP         0x000154 0x08048154 0x08048154 0x00013 0x00013 R   0x1
      [Requesting program interpreter: /lib/ld-linux.so.2]
  LOAD           0x000000 0x08048000 0x08048000 0x005e4 0x005e4 R E 0x1000
  LOAD           0x000f14 0x08049f14 0x08049f14 0x00100 0x00108 RW  0x1000
  DYNAMIC        0x000f28 0x08049f28 0x08049f28 0x000c8 0x000c8 RW  0x4
  NOTE           0x000168 0x08048168 0x08048168 0x00044 0x00044 R   0x4
  GNU_EH_FRAME   0x0004ec 0x080484ec 0x080484ec 0x00034 0x00034 R   0x4
  GNU_STACK      0x000000 0x00000000 0x00000000 0x00000 0x00000 RW  0x4
  GNU_RELRO      0x000f14 0x08049f14 0x08049f14 0x000ec 0x000ec R   0x1

 Section to Segment mapping:
  Segment Sections...
   00
   01     .interp
   02     .interp .note.ABI-tag .note.gnu.build-id .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rel.dyn .rel.plt .init .plt .text .fini .rodata .eh_frame_hdr .eh_frame
   03     .ctors .dtors .jcr .dynamic .got .got.plt .data .bss
   04     .dynamic
   05     .note.ABI-tag .note.gnu.build-id
   06     .eh_frame_hdr
   07
   08     .ctors .dtors .jcr .dynamic .got

Dynamic section at offset 0xf28 contains 20 entries:
  Tag        Type                         Name/Value
 0x00000001 (NEEDED)                     Shared library: [libc.so.6]
 0x0000000c (INIT)                       0x80482b4
 0x0000000d (FINI)                       0x80484bc
 0x6ffffef5 (GNU_HASH)                   0x80481ac
 0x00000005 (STRTAB)                     0x804821c
 0x00000006 (SYMTAB)                     0x80481cc
 0x0000000a (STRSZ)                      76 (bytes)
 0x0000000b (SYMENT)                     16 (bytes)
 0x00000015 (DEBUG)                      0x0
 0x00000003 (PLTGOT)                     0x8049ff4
 0x00000002 (PLTRELSZ)                   24 (bytes)
 0x00000014 (PLTREL)                     REL
 0x00000017 (JMPREL)                     0x804829c
 0x00000011 (REL)                        0x8048294
 0x00000012 (RELSZ)                      8 (bytes)
 0x00000013 (RELENT)                     8 (bytes)
 0x6ffffffe (VERNEED)                    0x8048274
 0x6fffffff (VERNEEDNUM)                 1
 0x6ffffff0 (VERSYM)                     0x8048268
 0x00000000 (NULL)                       0x0

Relocation section '.rel.dyn' at offset 0x294 contains 1 entries:
 Offset     Info    Type            Sym.Value  Sym. Name
08049ff0  00000206 R_386_GLOB_DAT    00000000   __gmon_start__

Relocation section '.rel.plt' at offset 0x29c contains 3 entries:
 Offset     Info    Type            Sym.Value  Sym. Name
0804a000  00000107 R_386_JUMP_SLOT   00000000   printf
0804a004  00000207 R_386_JUMP_SLOT   00000000   __gmon_start__
0804a008  00000307 R_386_JUMP_SLOT   00000000   __libc_start_main

There are no unwind sections in this file.

Symbol table '.dynsym' contains 5 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
     0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 00000000     0 FUNC    GLOBAL DEFAULT  UND printf@GLIBC_2.0 (2)
     2: 00000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
     3: 00000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@GLIBC_2.0 (2)
     4: 080484dc     4 OBJECT  GLOBAL DEFAULT   15 _IO_stdin_used

Symbol table '.symtab' contains 65 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
     0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 08048154     0 SECTION LOCAL  DEFAULT    1
     2: 08048168     0 SECTION LOCAL  DEFAULT    2
     3: 08048188     0 SECTION LOCAL  DEFAULT    3
     4: 080481ac     0 SECTION LOCAL  DEFAULT    4
     5: 080481cc     0 SECTION LOCAL  DEFAULT    5
     6: 0804821c     0 SECTION LOCAL  DEFAULT    6
     7: 08048268     0 SECTION LOCAL  DEFAULT    7
     8: 08048274     0 SECTION LOCAL  DEFAULT    8
     9: 08048294     0 SECTION LOCAL  DEFAULT    9
    10: 0804829c     0 SECTION LOCAL  DEFAULT   10
    11: 080482b4     0 SECTION LOCAL  DEFAULT   11
    12: 080482f0     0 SECTION LOCAL  DEFAULT   12
    13: 08048330     0 SECTION LOCAL  DEFAULT   13
    14: 080484bc     0 SECTION LOCAL  DEFAULT   14
    15: 080484d8     0 SECTION LOCAL  DEFAULT   15
    16: 080484ec     0 SECTION LOCAL  DEFAULT   16
    17: 08048520     0 SECTION LOCAL  DEFAULT   17
    18: 08049f14     0 SECTION LOCAL  DEFAULT   18
    19: 08049f1c     0 SECTION LOCAL  DEFAULT   19
    20: 08049f24     0 SECTION LOCAL  DEFAULT   20
    21: 08049f28     0 SECTION LOCAL  DEFAULT   21
    22: 08049ff0     0 SECTION LOCAL  DEFAULT   22
    23: 08049ff4     0 SECTION LOCAL  DEFAULT   23
    24: 0804a00c     0 SECTION LOCAL  DEFAULT   24
    25: 0804a014     0 SECTION LOCAL  DEFAULT   25
    26: 00000000     0 SECTION LOCAL  DEFAULT   26
    27: 00000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
    28: 08049f14     0 OBJECT  LOCAL  DEFAULT   18 __CTOR_LIST__
    29: 08049f1c     0 OBJECT  LOCAL  DEFAULT   19 __DTOR_LIST__
    30: 08049f24     0 OBJECT  LOCAL  DEFAULT   20 __JCR_LIST__
    31: 08048360     0 FUNC    LOCAL  DEFAULT   13 __do_global_dtors_aux
    32: 0804a014     1 OBJECT  LOCAL  DEFAULT   25 completed.6086
    33: 0804a018     4 OBJECT  LOCAL  DEFAULT   25 dtor_idx.6088
    34: 080483c0     0 FUNC    LOCAL  DEFAULT   13 frame_dummy
    35: 00000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
    36: 08049f18     0 OBJECT  LOCAL  DEFAULT   18 __CTOR_END__
    37: 080485e0     0 OBJECT  LOCAL  DEFAULT   17 __FRAME_END__
    38: 08049f24     0 OBJECT  LOCAL  DEFAULT   20 __JCR_END__
    39: 08048490     0 FUNC    LOCAL  DEFAULT   13 __do_global_ctors_aux
    40: 00000000     0 FILE    LOCAL  DEFAULT  ABS a.c
    41: 08049f14     0 NOTYPE  LOCAL  DEFAULT   18 __init_array_end
    42: 08049f28     0 OBJECT  LOCAL  DEFAULT   21 _DYNAMIC
    43: 08049f14     0 NOTYPE  LOCAL  DEFAULT   18 __init_array_start
    44: 08049ff4     0 OBJECT  LOCAL  DEFAULT   23 _GLOBAL_OFFSET_TABLE_
    45: 08048480     2 FUNC    GLOBAL DEFAULT   13 __libc_csu_fini
    46: 08048482     0 FUNC    GLOBAL HIDDEN    13 __i686.get_pc_thunk.bx
    47: 0804a00c     0 NOTYPE  WEAK   DEFAULT   24 data_start
    48: 00000000     0 FUNC    GLOBAL DEFAULT  UND printf@@GLIBC_2.0
    49: 0804a014     0 NOTYPE  GLOBAL DEFAULT  ABS _edata
    50: 080484bc     0 FUNC    GLOBAL DEFAULT   14 _fini
    51: 08049f20     0 OBJECT  GLOBAL HIDDEN    19 __DTOR_END__
    52: 0804a00c     0 NOTYPE  GLOBAL DEFAULT   24 __data_start
    53: 00000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
    54: 0804a010     0 OBJECT  GLOBAL HIDDEN    24 __dso_handle
    55: 080484dc     4 OBJECT  GLOBAL DEFAULT   15 _IO_stdin_used
    56: 00000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@@GLIBC_
    57: 08048410    97 FUNC    GLOBAL DEFAULT   13 __libc_csu_init
    58: 0804a01c     0 NOTYPE  GLOBAL DEFAULT  ABS _end
    59: 08048330     0 FUNC    GLOBAL DEFAULT   13 _start
    60: 080484d8     4 OBJECT  GLOBAL DEFAULT   15 _fp_hw
    61: 0804a014     0 NOTYPE  GLOBAL DEFAULT  ABS __bss_start
    62: 080483e4    40 FUNC    GLOBAL DEFAULT   13 main
    63: 00000000     0 NOTYPE  WEAK   DEFAULT  UND _Jv_RegisterClasses
    64: 080482b4     0 FUNC    GLOBAL DEFAULT   11 _init

Histogram for `.gnu.hash' bucket list length (total of 2 buckets):
 Length  Number     % of total  Coverage
      0  1          ( 50.0%)
      1  1          ( 50.0%)    100.0%

Version symbols section '.gnu.version' contains 5 entries:
 Addr: 0000000008048268  Offset: 0x000268  Link: 5 (.dynsym)
  000:   0 (*local*)       2 (GLIBC_2.0)     0 (*local*)       2 (GLIBC_2.0)
  004:   1 (*global*)

Version needs section '.gnu.version_r' contains 1 entries:
 Addr: 0x0000000008048274  Offset: 0x000274  Link: 6 (.dynstr)
  000000: Version: 1  File: libc.so.6  Cnt: 1
  0x0010:   Name: GLIBC_2.0  Flags: none  Version: 2

Notes at offset 0x00000168 with length 0x00000020:
  Owner                 Data size   Description
  GNU                  0x00000010   NT_GNU_ABI_TAG (ABI version tag)
    OS: Linux, ABI: 2.6.15

Notes at offset 0x00000188 with length 0x00000024:
  Owner                 Data size   Description
  GNU                  0x00000014   NT_GNU_BUILD_ID (unique build ID bitstring)
    Build ID: 17fb9651029b6a8543bfafec9eea23bd16454e65
```

# nm

# strings

`strings` 命令在对象文件或二进制文件中查找可打印的字符串。字符串是4个或更多可打印字符的任意序列，以换行符或空字符结束。 strings命令对识别随机对象文件很有用。

# c++filt

