---
title: CUDA API Remoting技术
date: 2018-05-14 16:28:02
tags:
- cuda
- gpu
categories:
- GPU
- GPU虚拟化
---

CUDA的虚拟化有一项技术为 `API Remoting`， 通俗点就是将编程API重定向，或者说远程过程调用。这是在接口层面上实现虚拟化, 采用对调用接口二次封
装的方法。 API 重定向虽然能够达到接近原生硬件的性能, 但是需要修改客户虚拟机中程序库。本文探究CUDA runtime API的重定向细节。

<!-- more -->


# GPU虚拟化

# API Remoting

`nvcc` 对CUDA库的默认的链接方式是静态链接。可以通过 `ldd` 查询，未发现关于 `libcudart.so`的动态链接库。
其实可以通过 `nvcc` 编译过程来发现端倪。

```
GPU$ nvcc --verbose thread.cu -o staticthread
```

> #$ nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use  -Wno-deprecated-gpu-targets to suppress warning).

- 读取环境变量

> #$ _SPACE_= 
> #$ _CUDART_=cudart
> #$ _HERE_=/usr/local/cuda-8.0/bin
> #$ _THERE_=/usr/local/cuda-8.0/bin
> #$ _TARGET_SIZE_=
> #$ _TARGET_DIR_=
> #$ _TARGET_SIZE_=64
> #$ TOP=/usr/local/cuda-8.0/bin/..
> #$ NVVMIR_LIBRARY_DIR=/usr/local/cuda-8.0/bin/../nvvm/libdevice
> #$ LD_LIBRARY_PATH=/usr/local/cuda-8.0/bin/../lib::/usr/local/cuda-8.0/lib64
> #$ PATH=/usr/local/cuda-8.0/bin/../open64/bin:/usr/local/cuda-8.0/bin/../nvvm/bin:/usr/local/cuda-8.0/bin:/home/max/bin:/usr/local/> sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-8.0/bin
> #$ INCLUDES="-I/usr/local/cuda-8.0/bin/..//include"  
> #$ LIBRARIES=  "-L/usr/local/cuda-8.0/bin/..//lib64/stubs" "-L/usr/local/cuda-8.0/bin/..//lib64"
> #$ CUDAFE_FLAGS=
> #$ PTXAS_FLAGS=

- 使用 `C++预处理器` 进行预处理，生成中间文件 `.cpp1.ii`

讲一些定义好的枚举变量（如cudaError）、struct、静态内联函数、extern c++和extern函数，还重新定义了std命名空间、函数模板等内容，写在main函数之前。

> #$ gcc -D__CUDA_ARCH__=200 -E -x c++           -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-8.0/bin/..//include"   -D"__CUDACC_VER__=80061" -D"__CUDACC_VER_BUILD__=61" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=8" -include "cuda_runtime.h" -m64 "thread.cu" > "/tmp/tmpxft_0000286a_00000000-9_thread.cpp1.ii" 

- 调用 `cudafe` 将 `.cpp1.ii` 分别执行在 `host` 和 `device` 上代码分离开，生成 `.cudafe1.gpu`和 `cudafe1.c` ，其中 `main` 函数在 `.cudafe1.c`文件中。

> #$ cudafe --allow_managed --m64 --gnu_version=50400 -tused --no_remove_unneeded_entities  --gen_c_file_name "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.c" --stub_file_name "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.gpu" --nv_arch "compute_20" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_0000286a_00000000-3_thread.module_id" --include_file_name "tmpxft_0000286a_00000000-2_thread.fatbin.c" "/tmp/tmpxft_0000286a_00000000-9_thread.cpp1.ii" 

- 预处理，由于不同架构gpu的计算能力不同，需要进行相应的处理，生成 `.cpp4.ii` 。

> #$ gcc -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-8.0/bin/..//include"   -D"__CUDACC_VER__=80061" -D"__CUDACC_VER_BUILD__=61" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=8" -include "cuda_runtime.h" -m64 "thread.cu" > "/tmp/tmpxft_0000286a_00000000-5_thread.cpp4.ii" 

> #$ cudafe++ --allow_managed --m64 --gnu_version=50400 --parse_templates  --gen_c_file_name "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.cpp" --stub_file_name "tmpxft_0000286a_00000000-4_thread.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_0000286a_00000000-3_thread.module_id" "/tmp/tmpxft_0000286a_00000000-5_thread.cpp4.ii" 

- 使用 `c预处理器` 进行预处理，生成中间文件 `.cpp2.i`

> #$ gcc -D__CUDA_ARCH__=200 -E -x c           -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDACC__ -D__NVCC__ -D__CUDANVVM__  -D__CUDA_FTZ=0 -D__CUDA_PREC_DIV=1 -D__CUDA_PREC_SQRT=1 "-I/usr/local/cuda-8.0/bin/..//include"   -m64 "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.gpu" > "/tmp/tmpxft_0000286a_00000000-11_thread.cpp2.i" 

- 调用 `cudafe` 将 `.cpp2.i` 分别执行在 `host` 和 `device` 上代码分离开，生成 `.cudafe2.gpu`和 `cudafe2.c` 。

> #$ cudafe -w --allow_managed --m64 --gnu_version=50400 --c  --gen_c_file_name "/tmp/tmpxft_0000286a_00000000-12_thread.cudafe2.c" --stub_file_name "/tmp/tmpxft_0000286a_00000000-12_thread.cudafe2.stub.c" --gen_device_file_name "/tmp/tmpxft_0000286a_00000000-12_thread.cudafe2.gpu" --nv_arch "compute_20" --module_id_file_name "/tmp/tmpxft_0000286a_00000000-3_thread.module_id" --include_file_name "tmpxft_0000286a_00000000-2_thread.fatbin.c" "/tmp/tmpxft_0000286a_00000000-11_thread.cpp2.i" 

- 使用 `c预处理器` 进行预处理，生成中间文件 `.cpp3.i`

> #$ gcc -D__CUDA_ARCH__=200 -E -x c           -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__CUDABE__ -D__CUDANVVM__ -D__USE_FAST_MATH__=0  -D__CUDA_FTZ=0 -D__CUDA_PREC_DIV=1 -D__CUDA_PREC_SQRT=1 "-I/usr/local/cuda-8.0/bin/..//include"   -m64 "/tmp/tmpxft_0000286a_00000000-12_thread.cudafe2.gpu" > "/tmp/tmpxft_0000286a_00000000-13_thread.cpp3.i" 

- 生成 `.ptx` 文件

> #$ cicc  -arch compute_20 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 -nvvmir-library "/usr/local/cuda-8.0/bin/../nvvm/libdevice/libdevice.compute_20.10.bc" --orig_src_file_name "thread.cu"  "/tmp/tmpxft_0000286a_00000000-13_thread.cpp3.i" -o "/tmp/tmpxft_0000286a_00000000-6_thread.ptx"

- `PTX` 离线编译，将代码编译成一个确定的计算能力和 `SM` 版本，对应的版本信息保存在 `.cubin` 中。

> #$ ptxas  -arch=sm_20 -m64  "/tmp/tmpxft_0000286a_00000000-6_thread.ptx"  -o "/tmp/tmpxft_0000286a_00000000-14_thread.sm_20.cubin" 

- 生成 `.fatbin.c`

> #$ fatbinary --create="/tmp/tmpxft_0000286a_00000000-2_thread.fatbin" -64 "--image=profile=sm_20,file=/tmp/tmpxft_0000286a_00000000-14_thread.sm_20.cubin" "--image=profile=compute_20,file=/tmp/tmpxft_0000286a_00000000-6_thread.ptx" --embedded-fatbin="/tmp/tmpxft_0000286a_00000000-2_thread.fatbin.c" --cuda


> #$ rm /tmp/tmpxft_0000286a_00000000-2_thread.fatbin
> #$ gcc -D__CUDA_ARCH__=200 -E -x c++           -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__USE_FAST_MATH__=0  -D__CUDA_FTZ=0 -D__CUDA_PREC_DIV=1 -D__CUDA_PREC_SQRT=1 "-I/usr/local/cuda-8.0/bin/..//include"   -m64 "/tmp/tmpxft_0000286a_00000000-4_thread.cudafe1.cpp" > "/tmp/tmpxft_0000286a_00000000-15_thread.ii" 
> #$ gcc -c -x c++ "-I/usr/local/cuda-8.0/bin/..//include"   -fpreprocessed -m64 -o "/tmp/tmpxft_0000286a_00000000-16_thread.o" "/tmp/tmpxft_0000286a_00000000-15_thread.ii" 
> #$ nvlink --arch=sm_20 --register-link-binaries="/tmp/tmpxft_0000286a_00000000-7_staticthread_dlink.reg.c" -m64   "-L/usr/local/cuda-8.0/bin/..//lib64/stubs" "-L/usr/local/cuda-8.0/bin/..//lib64" -cpu-arch=X86_64 "/tmp/tmpxft_0000286a_00000000-16_thread.o"  -o "/tmp/tmpxft_0000286a_00000000-17_staticthread_dlink.sm_20.cubin"
> #$ fatbinary --create="/tmp/tmpxft_0000286a_00000000-8_staticthread_dlink.fatbin" -64 -link "--image=profile=sm_20,file=/tmp/tmpxft_0000286a_00000000-17_staticthread_dlink.sm_20.cubin" --embedded-fatbin="/tmp/tmpxft_0000286a_00000000-8_staticthread_dlink.fatbin.c" 
> #$ rm /tmp/tmpxft_0000286a_00000000-8_staticthread_dlink.fatbin
> #$ gcc -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0000286a_00000000-8_staticthread_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0000286a_00000000-7_staticthread_dlink.reg.c\"" -I. "-I/usr/local/cuda-8.0/bin/..//include"   -D"__CUDACC_VER__=80061" -D"__CUDACC_VER_BUILD__=61" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=8" -m64 -o "/tmp/tmpxft_0000286a_00000000-18_staticthread_dlink.o" "/usr/local/cuda-8.0/bin/crt/link.stub" 

- `gcc` 链接所有目标文件

> #$ g++ -m64 -o "staticthread" -Wl,--start-group "/tmp/tmpxft_0000286a_00000000-18_staticthread_dlink.o" "/tmp/tmpxft_0000286a_00000000-16_thread.o"   "-L/usr/local/cuda-8.0/bin/..//lib64/stubs" "-L/usr/local/cuda-8.0/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group


注意最后一行 `#$ g++ -m64 -o "staticthread" -Wl,--start-group "/tmp/tmpxft_0000286a_00000000-18_staticthread_dlink.o" "/tmp/tmpxft_0000286a_00000000-16_thread.o"   "-L/usr/local/cuda-8.0/bin/..//lib64/stubs" "-L/usr/local/cuda-8.0/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group`，这里的链接的库 `cudadevrt`和 `cudart_static` 是位于 `/usr/local/cuda/lib64` 中的 `libcudadevrt.a` 和 `libcudart_static.a` 。


通过设置 cuda 的编译选项，使其调用库为动态调用： `--cudart=shared` 。 这样编译出来的二进制文件小很多。

```
-rwxrwxr-x  1 max max 569848 5月  15 09:39 staticthread*
-rwxrwxr-x  1 max max  19552 5月  14 14:19 thread*
```



## cuda kernel

github 上的[cudahook](https://github.com/nchong/cudahook)给出了cuda运行时API的钩子函数，利用了 `LD_PRELOAD` 和 `dlsym` 。

之前尝试失败，造成的原因就是 nvcc 静态编译。 修改成动态链接即可。

位于 `/usr/local/cuda/include/crt/host_runtime.h`中的
```
__cudaRegisterFunction
__cudaUnregisterFatBinary
__cudaRegisterFatBinary
__cudaInitModule	// 这个目前没用上
```
其他函数：
```
cudaMalloc
cudaConfigureCall
cudaLaunch
cudaFree
cudaSetupArgument
cudaMemcpy

```

## qCUDA

今天【2018，6，1】收获颇丰，先是找到一份 `API Remoting` 的源码[qCUDA: GPGPU Virtualization at a New API Remoting Method with Para-virtualization](https://github.com/coldfunction/qCUDA)。 

`qCUDA` 采用的是虚拟机上的GPU虚拟化，采用 `virtio` 作为传输通道，但是 `kernel` 函数没有进行传输，直接将其二进制的客户机物理地址转换（guest physical address, GPA）到宿主机虚拟地址（host virtual address，HVA），据说带宽效率达到的95%。

关键部分是在 `guest` 的驱动中，将用户态 `malloc` 后的内存，全部通过 `copy_from_user_safe` 拷贝到在内核中 `kmalloc` 申请的内存，之后再通过 `virt_to_phys` 将内核的虚拟地址转换成物理地址。


> virt_to_phys: The returned physical address is the physical (CPU) mapping for the memory address given. 
> It is only valid to use this function on addresses directly mapped or allocated via kmalloc. 
> It means It is used by the kernel to translate kernel virtual address (not user virtual address) to physical address


## CRCUDA

是的，今天【2018，6，1】收获第二件事就是找到了 `pause/resume` 的另一份代码：[Transparent checkpoint/restart library for CUDA application.
](https://github.com/tbrand/CRCUDA)。

【暂时没有研究】


## 查找二进制中的 `fatbin`部分

最关键的部分就是如何将包含 `kernel` 的 `GPU` 代码从二进制中找到并剥离出来。

还是在[what are the parameters for __cudaRegisterFatBinary and __cudaRegisterFunction functions?
](https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f/39453201)此问题下，良心答主给出了建设性意见。

答主提到： `__cuRegisterFatBinary ` 函数的唯一一个 `void *` 的指针参数，指向的是一个结构体：

```
struct {
    uint32_t magic; // Always 0x466243b1
    uint32_t seq;   // Sequence number of the cubin
    uint64_t ptr;   // The pointer to the real cubin
    uint64_t data_ptr;    // Some pointer related to the data segment
}
```
而这个结构体中的字段 `ptr` 指向的是真正的 fatBin文件，此fatBin文件按照 `fatBinary.h` 中的格式定义。 此文件中会有一些其他信息，如果接续搜索下去，会搜索到 `0x7F + 'ELF'` ，可以再此提取出 `cubin` 文件。

按照提示，我去做一些尝试！


在 `/usr/local/cuda/include/` 中找到 `fatBinary.h` 文件，并找到了 fat binary 头结构。
```
struct __align__(8) fatBinaryHeader        
{
	unsigned int 			magic;
	unsigned short         version;
	unsigned short         headerSize;
	unsigned long long int fatSize;
};
```

那么指向此结构体的指针在哪里呢？

这还要从 `__cudaRegisterFatBinary` 讲起来。它的函数声明为：
```
void** __cudaRegisterFatBinary(void *fatCubin);
```

这个函数传入的参数 `fatCubin` 指针是指向的是一个结构体，此结构体定义在 `/usr/local/cuda/include/fatBinaryCtl.h` 中。
```
/*
 * These defines are for the fatbin.c runtime wrapper
 */
#define FATBINC_MAGIC   0x466243B1
#define FATBINC_VERSION 1
#define FATBINC_LINK_VERSION 2
typedef struct {
	int magic;
	int version;
	const unsigned long long* data;
	void *filename_or_fatbins;  /* version 1: offline filename,
                               * version 2: array of prelinked fatbins */
} __fatBinC_Wrapper_t;
```
`__fatBinC_Wrapper_t` 第三个参数就是指向的真是的 fatCubin，而 fatCubin 的最开始的元数据是结构体 `struct fatBinaryHeader` 。

通过代码来验证：[空白]

```

```
可以通过 `nvcc` 编译生成 `fatbin` 文件，与 截获的文件比较。
```
nvcc --cudart=shared --fatbin -o test.fatbin test.cu
diff test.fatbin cut.fatbin
```




## 注意事项

### 编译问题

- 对 `dlopen` 未定义得引用

在编译时加入动态库 `ldl` 。
```
gcc -ldl ***
```

- `RTLD_NEXT` undeclared (first use in this function)

主要是 `RTLD_NEXT` 没有定义在 posix标准中，因此需要在代码的*最开始*加上 `#define _GNU_SOURCE` 。

通过 `man dlsym` 可以清晰得查看到。

```
SYNOPSIS
	#include <dlfcn.h>

	void *dlsym(void *handle, const char *symbol);

	#define _GNU_SOURCE
	#include <dlfcn.h>

	void *dlvsym(void *handle, char *symbol, char *version);

	Link with -ldl.
```

- error: conflicting types for错误原因

自己在写 `cudaMemcpy` 函数定义时，是按照CUDA Runtime API 文档写的，
```
cudaError_t cudaMemcpy(
		void* dst, 
		const void* src, 
		size_t count,  
		cudaMemcpyKind kind)
{
}
```

可是报错：

```
error: expected declaration specifiers or '...' before 'cudaMemcpyKind'
error: conflicting types for 'cudaMemcpy'
/usr/local/cuda/include/cuda_runtime_api.h:4130: note: previous declaration of 'cudaMemcpy' was here
```
原因包括很多，

	-- 没有函数声明，且函数定义在主函数之后；
	-- 头文件的被循环引用，在引用时考虑清楚包含顺序
	-- 头文件函数声明和函数定义参数不同

通过查看 `/usr/local/cuda/include/cuda_runtime_api.h:4130` 中声明的函数为：
```
cudaError_t cudaMemcpy(
		void* dst, 
		const void* src, 
		size_t count,  
		enum cudaMemcpyKind kind)
```
与自己定义的差了一个 `enum`，失声痛哭。

参考：[error: conflicting types for 错误原因及解决办法](http://blog.51cto.com/10901086/1903340)


# 参考
[1] [what are the parameters for __cudaRegisterFatBinary and __cudaRegisterFunction functions?](https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f)
[2] [cudahook](https://github.com/nchong/cudahook)