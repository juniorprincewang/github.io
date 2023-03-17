---
title: pthread condition使用
date: 2022-06-16 16:19:12
tags:
categories:
---

本文总结pthread condition的推荐使用方法。

<!--more-->

condition用于多线程之间同步。
涉及到的变量类型 `pthread_cond_t`，函数 `pthread_cond_init`、`pthread_cond_destroy`、 `pthread_cond_wait`、`pthread_cond_timedwait`、`pthread_cond_signal`、`pthread_cond_broadcast`。

这里推荐参考 [条件变量的陷阱与思考](https://www.cnblogs.com/liyuan989/p/4240271.html)。
使用c++类来阐述更方便一些，推荐使用condition的方式为：

```c
class ConditionCase
{
public:
    ConditionCase()
    {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond, NULL);
    }

    ~ConditionCase()
    {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }

    // invoke in thread 1:
    void wait()
    {
        pthread_mutex_lock(&mutex);
        while (!signal)
        {
            pthread_cond_wait(&cond, &mutex);
        }
        signal = false;
        pthread_mutex_unlock(&mutex);
    }

    // invoke in thread 2:
    void wakeup()
    {
        pthread_mutex_lock(&mutex);
        signal = true;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);
    }
private:
    pthread_mutex_t  mutex;
    pthread_cond_t   cond;
    bool signal;
};
```

通常 `pthread_cond_wait()` 会unlock mutex，然后是线程1睡眠，让出CPU，等待其他线程对condition发送signal，但是也需要重新获取mutex才可以。
`pthread_cond_signal` 不会unlock mutex，因此 waiting thread需要等到 signalling thread发出 condition signal并且unlock mutex才能醒过来后继续执行。
流程为：
1) TH1 locks the mutex 
2) TH1 unlocks the mutex (with pthread_cond) 
3) TH2 locks the mutex 
4) TH2 unlocks the mutex and sends the signal 
5) TH1 gets the mutex back 
6) TH1 unlocks the mutex

需要注意的是需要链接 `libpthread`，否则 `pthread_cond_wait()`不会阻塞等待，这是 glibc实现的愿意。参见[Why does pthread_cond_wait() not block when not linked with "-lpthread"?](https://stackoverflow.com/a/51163244)

参考：  
[understanding of pthread_cond_wait() and pthread_cond_signal()](https://stackoverflow.com/q/16522858)