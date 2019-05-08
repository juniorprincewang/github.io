---
title: CUDA handbook
date: 2019-05-07 21:42:07
tags:
- CUDA
- loop unrolling
categories:
- [GPU,CUDA]
---

本篇博客介绍学习《CUDA Handbook》过程中的总结。

<!-- more -->

# loop unrolling

[CUDA循环展开](https://blog.csdn.net/u012417189/article/details/33313729)

global_write函数未展开版:
```c++
template <class T>
 __global__ void Global_write(T*out,T value,size_t N){
    for ( size_t i = blockIdx.x*blockDim.x+threadIdx.x;
        i < N;i += blockDim.x*gridDim.x ) {
    out[i] = value;
    }
}
```

global_write函数展开,展开n层   
```c++
template<class T, const int n>
__global__ void Global_write_unrolling(T* out, T value, size_t N){
    size_t i;
    for(i = n*blockDim.x*blockIdx.x + threadIdx.x;
        i < N - n*blockDim.x*blockIdx.x;
        i += n*gridDim.x*blockDim.x;) {
            for(int j = 0; j < n; i++){
                size_t index = i + j * blockDim.x;
                outp[index] = value;
            }
        }
        
    // 为了不在循环里加入控制语句，将最后一次循环单独的写
    for ( int j = 0; j < n; j++ ) {
        size_t index = i+j*blockDim.x;
        if ( index<N ) out[index] = value;
    }
}
```

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll) 也给出了 `#pragma unroll` 说明和使用方法。   
`#pragma unroll` 指导编译器优化循环，比如：  

```c++
// no argument specified, loop will be completely unrolled
#pragma unroll
for (int i = 0; i < 12; ++i) 
  p1[i] += p2[i]*2;
```
循环条件必须确定，不能是变量。见[What does #pragma unroll do exactly? Does it affect the number of threads?](https://stackoverflow.com/questions/22278631/what-does-pragma-unroll-do-exactly-does-it-affect-the-number-of-threads)。  



