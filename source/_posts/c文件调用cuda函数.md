---
title: c文件调用cuda函数
date: 2018-04-11 17:30:30
tags:
- CUDA
categories:
- [GPU,CUDA]
---

经过无数次的折腾，终于成功在C文件中调用了cu文件里面定义的函数。
<!--more -->

`*.c` 文件可以用 `gcc` 或者 `g++` 编译， `*.cu` 文件需要用 `nvcc` 编译器编译，所以 

# .c文件调用.cu文件的函数

有三个文件， `b.h` 中声明了 `kernel_wrapper` 函数，在 `b.cu` 中实现， `a.c` 需要调用 `kernel_wrapper` 函数。

`b.h` 文件
```
#ifndef __B_H_
#define __B_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
void kernel_wrapper(int *a);
#endif 
```

`b.cu` 文件

```
#include "b.h"

__global__ void kernel(int *a)
{
	int tx = threadIdx.x; 
	switch( tx )
	{
		case 0:
			a[tx] = a[tx] + 2;
			break;
		case 1:
			a[tx] = a[tx] + 3;
			break;
	}
}
void kernel_wrapper(int *a)
{
	int *d_a;
	dim3 threads( 2, 1 );
	dim3 blocks( 1, 1 );
	cudaMalloc( (void **)&d_a, sizeof(int) * 2 );
	cudaMemcpy( d_a, a, sizeof(int) * 2, cudaMemcpyHostToDevice );
	kernel<<< blocks, threads >>>( d_a );
	cudaMemcpy( a, d_a, sizeof(int) * 2, cudaMemcpyDeviceToHost );
	printf( "Finish kernel wrapper\n" );
	cudaFree(d_a);
}
```

`a.c` 文件

```
#include "b.h"
int main(int argc, char *argv[])
{
	int *a = (int *)malloc(sizeof(int) * 2);
	a[0] = 2;
	a[1] = 3;
	printf( "a[0]: %d, a[1]: %d\n", a[0], a[1] );
	kernel_wrapper(a);
	printf( "a[0]: %d, a[1]: %d\n", a[0], a[1] );
	free(a);
	return 0;
} 
```


`Makefile` 文件：
```
run: a.o b.o
	gcc -L /usr/local/cuda/lib64 -o run a.o b.o -lcudart -lcuda

a.o: a.c b.h
	gcc -I /usr/local/cuda/include -c -o a.o a.c

b.o: b.cu b.h
	nvcc -c -o b.o b.cu
```

- `-I` 告诉编译器查找头文件的位置。
- `-L` 告诉链接器查找需要链接库的位置。
- `-l` 告诉链接器链接的库文件，通常的名字是不加 `lib`的，比如 `libcudart.so` 这里写 `cudart` 。


现在进行编译，输入 `make`，得到报错信息：

```
undefined reference to `kernel_wrapper'
```


## 解决办法

正如开头所说的， `nvcc` 使用 `C++` 编译器 `g++`，而 `c` 文件要链接由 `g++` 编译的库，因此会报错。

解决办法就是告诉编译器，函数以 `c` 的方式来编译封装接口，而函数中的 `C++` 语法还是用 `C++` 来编译。

```
// b.cu

extern "C" {
	#include "b.h"
}
...
extern "C" void kernel_wrapper(int *a)
{
	int *d_a;
	dim3 threads( 2, 1 );
	dim3 blocks( 1, 1 );
	cudaMalloc( (void **)&d_a, sizeof(int) * 2 );
	cudaMemcpy( d_a, a, sizeof(int) * 2, cudaMemcpyHostToDevice );
	kernel<<< blocks, threads >>>( d_a );
	cudaMemcpy( a, d_a, sizeof(int) * 2, cudaMemcpyDeviceToHost );
	printf( "Finish kernel wrapper\n" );
	cudaFree(d_a);
}
```

而且需要注意的是 CUDA 共享库需要在目标文件(\*.o)后使用。
```
gcc -L /usr/local/cuda/lib64 -o run a.o b.o -lcudart -lcuda
```

另一方法是使用 `g++` 或者 `nvcc` 来链接，还未尝试。

# .c文件调用.cu文件生成的.so库

还是上述修改过后的文件。

编译 `b.cu` 文件为 `libcudab.so` 动态链接库。

```
nvcc --shared --compiler-options "-fpic -shared" b.cu -o libcudab.so -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcublas 
```

再编译并链接 `a.c` 。
```
gcc -o main a.c -L. -lcudab -I /usr/local/cuda/include
```


# 可能遇到的错误

## error while loading shared libraries:XXX.so

1. 如果共享库安装到了 `/lib` 或者 `/usr/lib` 目录下，需要执行一下 `ldconfig` 命令。

`ldconfig` 命令的用途，主要是在默认搜寻目录(`/lib` 和 `/usr/lib` )以及动态库配置文件 `/etc/ld.so.conf `内所列的目录下，搜索出可共享的动态链接库(格式如 `lib*.so*` )，进而创建出动态装入程序( `ld.so` )所需的连接和缓存文件。缓存文件默认为 `/etc/ld.so.cache` ，此文件保存已排好序的动态链接库名字列表。


2. 如果共享库文件安装到了 `/usr/local/lib` （很多开源的共享库都会安装到该目录下）或其它 "非/lib或/usr/lib" 目录下, 那么在执行 `ldconfig` 命令前，还要把新共享库目录加入到共享库配置文件 `/etc/ld.so.conf` 中, 如下:

```
# cat /etc/ld.so.conf
include /etc/ld.so.conf.d/*.conf
# echo "/usr/local/lib" >> /etc/ld.so.conf
# ldconfig
```

3. 如果共享库文件安装到了其它 "非/lib或/usr/lib" 目录下，但是又不想在 `/etc/ld.so.conf` 中加路径（或者是没有权限加路径）。那可以 `export` 一个全局变量 `LD_LIBRARY_PATH` ，然后运行程序的时候就会去这个目录中找共享库.。


# 参考
[1] [Cuda C - Linker error - undefined reference
](https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference)
[2] [在.c文件中调用c++定义的函数](https://blog.csdn.net/wang11234514/article/details/24034969)
[3] [Linux下c和cuda混合编译，并生成动态链接库.so和使用](https://blog.csdn.net/u012816621/article/details/52334622)