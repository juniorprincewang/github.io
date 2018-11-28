---
title: Linux内核线程kthread
date: 2018-11-21 18:52:27
tags:
- kthread
categories:
- linux
---

介绍Linux内核线程的编程。
<!-- more -->

Linux内核是调度的基本单位。内核线程是直接由内核本身启动的进程。内核线程实际上是将内核函数委托给独立的进程，它与内核中的其他进程”并行”执行。内核线程经常被称之为内核守护进程。
他们执行下列任务：
+ 周期性地将修改的内存页与页来源块设备同步
+ 如果内存页很少使用，则写入交换区
+ 管理延时动作,　如２号进程接手内核进程的创建
+ 实现文件系统的事务日志
+ ...
内核线程主要有两种类型

+ 线程启动后一直等待，直至内核请求线程执行某一特定操作。
+ 线程启动后按周期性间隔运行，检测特定资源的使用，在用量超出或低于预置的限制时采取行动。

内核线程由内核自身生成，其特点在于它们在内核态执行，不能访问用户态地址空间。

# 内核线程描述符 `struct task_struct`

```
<linux/sched.h>

```
# 创建线程
## kthread_create

```
include <linux/kthread.h>

/**
 * kthread_create_on_node - create a kthread.
 * @threadfn: the function to run until signal_pending(current).
 * @data: data ptr for @threadfn.
 * @node: task and thread structures for the thread are allocated on this node
 * @namefmt: printf-style name for the thread.
 *
 * Description: This helper function creates and names a kernel
 * thread.  The thread will be stopped: use wake_up_process() to start
 * it.  See also kthread_run().  The new thread has SCHED_NORMAL policy and
 * is affine to all CPUs.
 *
 * If thread is going to be bound on a particular cpu, give its node
 * in @node, to get NUMA affinity for kthread stack, or else give NUMA_NO_NODE.
 * When woken, the thread will run @threadfn() with @data as its
 * argument. @threadfn() can either call do_exit() directly if it is a
 * standalone thread for which no one will call kthread_stop(), or
 * return when 'kthread_should_stop()' is true (which means
 * kthread_stop() has been called).  The return value should be zero
 * or a negative error number; it will be passed to kthread_stop().
 *
 * Returns a task_struct or ERR_PTR(-ENOMEM) or ERR_PTR(-EINTR).
 */
struct task_struct *kthread_create_on_node(int (*threadfn)(void *data),
					   void *data,
					   int node,
					   const char namefmt[], ...);

/**
 * kthread_create - create a kthread on the current node
 * @threadfn: the function to run in the thread
 * @data: data pointer for @threadfn()
 * @namefmt: printf-style format string for the thread name
 * @arg...: arguments for @namefmt.
 *
 * This macro will create a kthread on the current node, leaving it in
 * the stopped state.  This is just a helper for kthread_create_on_node();
 * see the documentation there for more details.
 */
#define kthread_create(threadfn, data, namefmt, arg...) \
	kthread_create_on_node(threadfn, data, NUMA_NO_NODE, namefmt, ##arg)
```

创建内核更常用的方法是辅助函数 `kthread_create`，该函数创建一个新的内核线程。最初线程是停止的，需要使用`wake_up_process` 启动它。
```
include <linux/sched.h>
int wake_up_process(struct task_struct *tsk);
```
## kthread_run
创建并唤醒一个线程。
```
/**
 * kthread_run - create and wake a thread.
 * @threadfn: the function to run until signal_pending(current).
 * @data: data ptr for @threadfn.
 * @namefmt: printf-style name for the thread.
 *
 * Description: Convenient wrapper for kthread_create() followed by
 * wake_up_process().  Returns the kthread or ERR_PTR(-ENOMEM).
 */
#define kthread_run(threadfn, data, namefmt, ...)			   \
({									   \
	struct task_struct *__k						   \
		= kthread_create(threadfn, data, namefmt, ## __VA_ARGS__); \
	if (!IS_ERR(__k))						   \
		wake_up_process(__k);					   \
	__k;								   \
})
```
# 终止线程
线程一旦启动起来后，会一直运行，除非该线程主动调用do_exit函数，或者其他的进程调用kthread_stop函数，结束线程的运行。

> @threadfn() can either call do_exit() directly if it is a
> * standalone thread for which no one will call kthread_stop(), or
> * return when 'kthread_should_stop()' is true (which means
> * kthread_stop() has been called).

## kthread_stop
```
int kthread_stop(struct task_struct *k);
```

设置 `kthread­>kthread_should_stop` ，并等待线程主动结束。

如果在调用 `kthread_stop` 前线程已结束，那么会导致进程crash。就需要`kthread_should_stop()` 来判断线程是否已经结束。
`kthread_should_stop()`返回 `should_stop` 标志。它用于创建的线程检查结束标志，并决定是否退出。线程完全可以在完成自己的工作后主动结束，不需等待 `should_stop`标志。


# 阻塞线程
阻塞线程一段预设的时间。
```
#include <linux/sched.h>
void schedule(void)
void schedule_timeout()
```

阻塞线程一段指定的时间。
```
#include <linux/delay.h>
void ssleep(unsigned int seconds)
```

# 样例

```
#include <linux/delay.h> /* usleep_range */
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");

static struct task_struct *kthread;

static int work_func(void *data)
{
	int i = 0;
	while (!kthread_should_stop()) {
		printk(KERN_INFO "%d\n", i);
		usleep_range(1000000, 1000001);
		i++;
		if (i == 10)
			i = 0;
	}
	return 0;
}

static int myinit(void)
{
	kthread = kthread_create(work_func, NULL, "mykthread");
	wake_up_process(kthread);
	return 0;
}

static void myexit(void)
{
	/* Waits for thread to return. */
	kthread_stop(kthread);
}

module_init(myinit);
module_exit(myexit);
```

# 参考
1. [Kernel threads made easy](https://lwn.net/Articles/65178/)
1. [Linux内核线程kernel thread详解--Linux进程的管理与调度（十）](https://blog.csdn.net/gatieme/article/details/51589205 )
3. [Proper way of handling threads in kernel?](https://stackoverflow.com/questions/10177641/proper-way-of-handling-threads-in-kernel)
4. [How to wait for a linux kernel thread (kthread)to exit?](https://stackoverflow.com/questions/4084708/how-to-wait-for-a-linux-kernel-thread-kthreadto-exit)
4. https://github.com/cirosantilli/linux-kernel-module-cheat/blob/6788a577c394a2fc512d8f3df0806d84dc09f355/kernel_module/kthreads.c
