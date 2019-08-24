---
title: C程序和代码执行时间
date: 2019-08-24 17:26:42
tags:
    - time
    - c
categories:
    - c
---

记录执行c代码和c程序的时间，benchmark执行时间。

<!-- more -->
# time

命令行命令： `time`

```
time yourscript.sh
```

+ [How to get execution time of a script effectively?](https://unix.stackexchange.com/questions/52313/how-to-get-execution-time-of-a-script-effectively)

# hyperfine

+ [A command-line benchmarking tool](https://github.com/sharkdp/hyperfine)

安装：

有各种平台的安装，这里在Ubuntu上的安装：  

```
wget https://github.com/sharkdp/hyperfine/releases/download/v1.6.0/hyperfine_1.6.0_amd64.deb
sudo dpkg -i hyperfine_1.6.0_amd64.deb
```

命令执行
```
hyperfine 'sleep 0.3'
```

这里默认执行benchmark10次，可以通过 *-m/--min-runs* 选项。  

```
hyperfine --min-runs 5 'sleep 0.2' 'sleep 3.2'
```

输出结果可以看到平均的执行时间。  

# SHELL

+ [Calculate average execution time of a program using Bash](https://stackoverflow.com/questions/54920113/calculate-average-execution-time-of-a-program-using-bash)

code snippet如下

```
avg_time() {
    #
    # usage: avg_time n command ...
    #
    n=$1; shift
    (($# > 0)) || return                   # bail if no command given
    for ((i = 0; i < n; i++)); do
        { time -p "$@" &>/dev/null; } 2>&1 # ignore the output of the command
                                           # but collect time's output in stdout
    done | awk '
        /real/ { real = real + $2; nr++ }
        /user/ { user = user + $2; nu++ }
        /sys/  { sys  = sys  + $2; ns++}
        END    {
                 if (nr>0) printf("real %f\n", real/nr);
                 if (nu>0) printf("user %f\n", user/nu);
                 if (ns>0) printf("sys %f\n",  sys/ns)
               }'
}

avg_time 5 sleep 1
```

运行此demo得到的结果如下：

    real 1.000000
    user 0.000000
    sys 0.000000

# code snippet

+ [Execution time of C program](https://stackoverflow.com/questions/5248915/execution-time-of-c-program)

执行代码的秒数：

```
#include <time.h>

clock_t begin = clock();

/* here, do your time-consuming job */

clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
```

执行代码的毫秒数：
```
#include <sys/time.h>

struct timeval  tv1, tv2;
gettimeofday(&tv1, NULL);
/* stuff to do! */
gettimeofday(&tv2, NULL);

printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));
```