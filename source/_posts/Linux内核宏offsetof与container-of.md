---
title: Linux内核宏offsetof与container_of
date: 2018-11-20 21:38:54
tags:
- container_of
- offsetof
categories:
- linux
---
`offsetof` 宏是用来判断结构体中成员的偏移位置，`container_of`宏用来根据成员的地址来获取结构体的地址。
<!-- more -->

# offsetof宏
offsetof是返回结构体TYPE中MEMBER成员相对于结构体首地址的偏移量，以字节为单位。

使用offsetof宏需要包含 `stddef.h` 头文件，实例可以参考：<http://www.cplusplus.com/reference/cstddef/offsetof/>。

offsetof宏的定义如下：
```
#define offsetof(type, member) (size_t)&(((type*)0)->member)
```
巧妙之处在于将地址0强制转换为type类型的指针，从而定位到member在结构体中偏移位置。编译器认为0是一个有效的地址，从而认为0是type指针的起始地址。

# container_of宏

container_of的主要作用是根据一个结构体变量中的一个域成员变量的指针来获取指向整个结构体变量的指针。

使用container_of宏需要包含 `linux/kernel.h` 头文件，container_of宏的定义如下所示：
```
#define container_of(ptr, type, member) ({ \
     const typeof( ((type *)0)->member ) *__mptr = (ptr); \
     (type *)( (char *)__mptr - offsetof(type,member) );})    
```
container_of宏分为两部分，

第一部分： `const typeof( ((type *)0)->member ) *__mptr = (ptr);`

通过 `typeof` 定义一个 `member` 指针类型的指针变量 `__mptr` ，（即`__mptr`是指向`member`类型的指针），并将`__mptr`赋值为`ptr`。

第二部分： `(type *)( (char *)__mptr - offsetof(type,member) )`，通过`offsetof`宏计算出 `member` 在 `type` 中的偏移，然后用 `member` 的实际地址 `__mptr` 减去偏移，得到 `type` 的起始地址，即指向 `type`类型的指针。

用一个例子来说明：
```
struct numbers {
    int one;
    int two;
    int three;
} n;

int *ptr = &n.two;
struct numbers *n_ptr;
n_ptr = container_of(ptr, struct numbers, two);
```

# 例子

```
#include <stdio.h>
#include <stdlib.h>

#define NAME_STR_LEN  32

#define offsetof(type, member) (size_t)&(((type*)0)->member)

#define container_of(ptr, type, member) ({ \
        const typeof( ((type *)0)->member ) *__mptr = (ptr); \
        (type *)( (char *)__mptr - offsetof(type,member) );})

typedef struct student_info
{
    int  id;
    char name[NAME_STR_LEN];
    int  age;
}student_info;


int main()
{
    size_t off_set = 0;
    off_set = offsetof(student_info, id);
    printf("id offset: %u\n",off_set);
    off_set = offsetof(student_info, name);
    printf("name offset: %u\n",off_set);
    off_set = offsetof(student_info, age);
    printf("age offset: %u\n",off_set);
    student_info *stu = (student_info *)malloc(sizeof(student_info));
    stu->age = 10;
    student_info *ptr = container_of(&(stu->age), student_info, age);
    printf("age:%d\n", ptr->age);
    printf("stu address:%p\n", stu);
    printf("ptr address:%p\n", ptr);
    return 0;
}
```

运行的结果：

	id offset: 0
	name offset: 4
	age offset: 36
	age:10
	stu address:0x18c8420
	ptr address:0x18c8420



# 参考
1. [offsetof与container_of宏\[总结\]](https://www.cnblogs.com/Anker/p/3472271.html)
2. [C语言链表常用宏——offsetof和container_of](https://www.jianshu.com/p/e22e31257d9a)
3. [Understanding container_of macro in the Linux kernel](https://stackoverflow.com/questions/15832301/understanding-container-of-macro-in-the-linux-kernel)

