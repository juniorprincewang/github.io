---
title: Linux设备驱动之workqueue
date: 2018-11-20 17:37:48
tags:
- workqueue
categories:
- linux
---

工作队列是一种将工作推后执行的形式，交由一个内核线程去执行在进程上下文执行，其不能访问用户空间。最重要特点的就是工作队列允许重新调度甚至是睡眠。
<!-- more -->

在内核代码中, 经常希望延缓部分工作到将来某个时间执行, 这样做的原因很多, 比如

+ 在持有锁时做大量(或者说费时的)工作不合适。
+ 希望将工作聚集以获取批处理的性能。
+ 调用了一个可能导致睡眠的函数使得在此时执行新调度非常不合适。
...
内核中提供了许多机制来提供延迟执行, 使用最多则是 workqueue。

工作队列（workqueue）是另外一种将工作推后执行的形式.工作队列可以把工作推后，交由一个内核线程去执行，也就是说，这个下半部分可以在进程上下文中执行。最重要的就是工作队列允许被重新调度甚至是睡眠。

对于使用者，基本上只需要做 3 件事情，依次为：

+ 创建工作队列 ( 如果使用内核默认的工作队列，连这一步都可以省略掉 )
+ 创、建工作项
+ 向工作队列中提交工作项

执行在进程上下文中，这样使得它可以睡眠，被调度及被抢占，在多核环境下的使用也非常友好。


# 数据结构

+ 工作：
所谓work就是异步执行的函数。用数据结构 `struct work_struct` 表示。

+ 工作队列： `struct workqueue_struct`


如果是多线程，Linux根据当前系统CPU的个数创建 `struct cpu_workqueue_struct`:

包含的头文件为 `<linux/workqueue.h>`

# 创建步骤

## 静态地创建work工作:

 静态地创建一个名为n，待执行函数为f，函数的参数为data的work_struct结构。
```
#define DECLARE_WORK(n, f)                    \ 
    struct work_struct n = __WORK_INITIALIZER(n, f)

#define DECLARE_DELAYED_WORK(n, f)                \ 
    struct delayed_work n = __DELAYED_WORK_INITIALIZER(n, f)
```
一般而言，work都是推迟到worker thread被调度的时刻，但是有时候，我们希望在指定的时间过去之后再调度worker thread来处理该work，这种类型的work被称作delayed work，DECLARE_DELAYED_WORK用来初始化delayed work，它的概念和普通work类似。

## 动态地创建work工作:

动态创建初始化的时候需要把work的指针传递给 `INIT_WORK` 。
```
INIT_WORK(struct work_struct work, work_func_t func); 
PREPARE_WORK(struct work_struct work, work_func_t func); 
INIT_DELAYED_WORK(struct delayed_work work, work_func_t func); 
PREPARE_DELAYED_WORK(struct delayed_work work, work_func_t func); 
```
## 清除或取消工作队列中的work工作

想清理特定的任务项目并阻塞任务， 直到任务完成为止， 可以调用 `flush_work` 来实现。 
指定工作队列中的所有任务能够通过调用 `flush_workqueue` 来完成。 这两种情形下，调用者阻塞直到操作完成为止。 
为了清理内核全局工作队列，可调用 `flush_scheduled_work`。
```
int flush_work( struct work_struct *work );
int flush_workqueue( struct workqueue_struct *wq );
void flush_scheduled_work( void );
```
还没有在处理程序当中执行的任务可以被取消。 调用 `cancel_work_sync` 将会终止队列中的任务或者阻塞任务直到回调结束（如果处理程序已经在处理该任务）。 如果任务被延迟，可以调用 `cancel_delayed_work_sync` 。

```
int cancel_work_sync( struct work_struct *work );
int cancel_delayed_work_sync( struct delayed_work *dwork );
```
最后，可以通过调用 `work_pending` 或者 `delayed_work_pending` 来确定任务项目是否在进行中。

```
work_pending( work );
delayed_work_pending( work );
```

## 创建销毁workqueue

+ 用于创建一个workqueue队列，为系统中的每个CPU都创建一个内核线程。
```
struct workqueue_struct *create_workqueue(const char *name); 
```
+ 用于创建workqueue，只创建一个内核线程。
```
struct workqueue_struct *create_singlethread_workqueue(const char *name);
```
+ 释放workqueue队列。
```
void destroy_workqueue(struct workqueue_struct *queue);
```

## 使用内核提供的共享列队

系统中包括若干的workqueue，最著名的workqueue就是系统缺省的的工作队列 `keventd_wq` 了，定义如下：
```
static struct workqueue_struct *keventd_wq __read_mostly;
```
+ 对工作进行调度，即把给定工作的处理函数提交给缺省的工作队列和工作线程。
```
      int schedule_work(struct work_struct *work);
```
+	确保没有工作队列入口在系统中任何地方运行。
```
      void flush_scheduled_work(void);
```
+	延时执行一个任务
```
      int schedule_delayed_work(struct delayed_struct *work, unsigned long delay);
```
+	从一个工作队列中去除入口;
```
      int cancel_delayed_work(struct delayed_struct *work);
```

## 使用自定义队列


+	将工作加入工作列队进行调度
```
int queue_work(struct workqueue_struct *wq, struct work_struct *work)
```

+	释放创建的工作列队资源
```
void destroy_workqueue(struct workqueue_struct *wq)
```

+	延时调用指定工作列队的工作
```
queue_delayed_work(struct workqueue_struct *wq, struct delay_struct *work, unsigned long delay)
```

+	取消指定工作列队的延时工作
```
cancel_delayed_work(struct delay_struct *work)
```

+	等待列队中的任务全部执行完毕。
```
void flush_workqueue(struct workqueue_struct *wq);
```

# 样例代码

```
/* https://github.com/cirosantilli/linux-kernel-module-cheat#workqueues */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/workqueue.h>

static struct workqueue_struct *queue;


static void work_func(struct work_struct *work)
{
	printk(KERN_INFO "worker\n");
}

DECLARE_WORK(work, work_func);

static int myinit(void)
{
	queue = create_workqueue("myworkqueue");
	queue_work(queue, &work);
	return 0;
}

static void myexit(void)
{
	destroy_workqueue(queue);
}

module_init(myinit)
module_exit(myexit)
MODULE_LICENSE("GPL");
```

# 参考
1. [4.8. Work Queues Understanding the Linux Kernel, 3rd Edition by Marco Cesati, Daniel P. Bovet](https://www.oreilly.com/library/view/understanding-the-linux/0596005652/ch04s08.html)
2. [Concurrency Managed Workqueue之（一）：workqueue的基本概念](http://www.wowotech.net/irq_subsystem/workqueue.html)
3. [linux工作队列](http://www.embeddedlinux.org.cn/emb-linux/system-development/201709/30-7472.html)
4. [Linux 的并发可管理工作队列机制探讨](https://www.ibm.com/developerworks/cn/linux/l-cn-cncrrc-mngd-wkq/index.html)
5. [工作队列(workqueue) create_workqueue/schedule_work/queue_work](https://blog.csdn.net/angle_birds/article/details/8448070)
6. [内核 API，第 2 部分：可延迟函数、内核微线程以及工作队列](https://www.ibm.com/developerworks/cn/linux/l-tasklets/index.html)