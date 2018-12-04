---
title: CUDA kernel之多进程访问研究
date: 2018-10-19 14:40:13
tags:
- CUDA
- GPU
categories:
- GPU
---
探究多进程或者多线程并发执行多个CUDA Kernel。
<!-- more -->


[GPU 虚拟化相关技术研究综述](http://www.c-s-a.org.cn/csa/ch/reader/create_pdf.aspx?file_no=6096&flag=1&year_id=12&quarter_id=) 里面提到今后GPU的可能的研究方向其中一点：可抢占性。

	由于 GPU 核数较多, 抢占 GPU 需要保存大量的上下文信息, 开销较大, 所以目前市场上 GPU 都不支持抢占特性. 只用当前任务完成之后, GPU 才能被下个应用程序使用。 在 GPU 虚拟化的环境中, 多用户使用的场景会导致 GPU 进行频繁的任务切换, 可抢占的 GPU 能够防止恶意用户长期占用, 并且 能够实现用户优先级权限管理。



一个答案

	A CUDA context is a virtual execution space that holds the code and data owned by a host thread or process. Only one context can ever be active on a GPU with all current hardware.

	So to answer your first question, if you have seven separate threads or processes all trying to establish a context and run on the same GPU simultaneously, they will be serialised and any process waiting for access to the GPU will be blocked until the owner of the running context yields. There is, to the best of my knowledge, no time slicing and the scheduling heuristics are not documented and (I would suspect) not uniform from operating system to operating system.

	You would be better to launch a single worker thread holding a GPU context and use messaging from the other threads to push work onto the GPU. Alternatively there is a context migration facility available in the CUDA driver API, but that will only work with threads from the same process, and the migration mechanism has latency and host CPU overhead.

这里涉及到 CUDA 上下文（CUDA context）。

尝试建立context并且同时运行在同一GPU设备上的不同的线程或进程，它们会被串行化而且任何等待访问GPU的进程将会被阻塞直到运行的context的进程退出。
据“答主”了解，并没有文档来介绍时间分片还有调度算法。
答主建议最好先启动包含着GPU上下文的单 worker 线程，使用来自别的线程的消息来将工作推给GPU。或者，CUDA driver API有个上下文迁移工具，它也能与来自同一进程的线程配合，但是迁移机制有延迟，对CPU带来负荷。  



	CUDA activity from independent host processes will normally create independent CUDA contexts, one for each process. Thus, the CUDA activity launched from separate host processes will take place in separate CUDA contexts, on the same device.

独立主机进程的CUDA程序正常创建独立的CUDA上下文，每个进程一个CUDA context。从隔离主机进程启动的CUDA程序将在不同的CUDA上下文执行。

	CUDA activity in separate contexts will be serialized. The GPU will execute the activity from one process, and when that activity is idle, it can and will context-switch to another context to complete the CUDA activity launched from the other process. The detailed inter-context scheduling behavior is not specified. (Running multiple contexts on a single GPU also cannot normally violate basic GPU limits, such as memory availability for device allocations.)

在不同上下文的CUDA程序将被串行化。GPU将执行来自一个进程的程序，并且当此程序空闲时，它将上下文切换到另外的上下文来完成从另一个进程启动的CUDA程序。详细的上下文内部调度行为并不具体。（在单GPU上运行多上下文同样不能正常违背基本的GPU限制，比如设备分配时的内存获取）

	The "exception" to this case (serialization of GPU activity from independent host processes) would be the CUDA Multi-Process Server. In a nutshell, the MPS acts as a "funnel" to collect CUDA activity emanating from several host processes, and run that activity as if it emanated from a single host process. The principal benefit is to avoid the serialization of kernels which might otherwise be able to run concurrently. The canonical use-case would be for launching multiple MPI ranks that all intend to use a single GPU resource.

CUDA Multi-Process Server简称 MPS，它扮演着一个漏斗的角色，来收集来自几个host进程的CUDA程序，并运行它们就好像来自一个host进程。主要的好处是避免kernel的串行化。

	Note that the above description applies to GPUs which are in the "Default" compute mode. GPUs in "Exclusive Process" or "Exclusive Thread" compute modes will reject any attempts to create more than one process/context on a single device. In one of these modes, attempts by other processes to use a device already in use will result in a CUDA API reported failure. The compute mode is modifiable in some cases using the nvidia-smi utility.

GPU在 "Exclusive Process" 或者 "Exclusive Thread" 计算模式将不允许任何在单设备上创建超过一个进程或上下文的操作请求。在上述模式下，其他进程的任何使用被占用的设备的尝试将会造成CUDA API 调用失败。计算模式可以在某些情况下通过 nvidia-smi 工具修改。

# 参考

[Multiple processes launching CUDA kernels in parallel](https://stackoverflow.com/questions/14895034/multiple-processes-launching-cuda-kernels-in-parallel)
[Running more than one CUDA applications on one GPU](https://stackoverflow.com/questions/31643570/running-more-than-one-cuda-applications-on-one-gpu)
