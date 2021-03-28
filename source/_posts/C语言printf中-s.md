---
title: C语言printf中格式化字符串问题
date: 2018-11-21 15:05:55
tags:
    - printf
categories:
    - [c]
---

`printf` 的格式化参数`printf("%.*s", len, buffer)`。
<!-- more -->

在 `c++` 的官方文档中可以查到相关资料。

+ `.number`	: 
> For integer specifiers (d, i, o, u, x, X): precision specifies the minimum number of digits to be written. If the value to be written is shorter than this number, the result is padded with leading zeros. The value is not truncated even if the result is longer. A precision of 0 means that no character is written for the value 0.
For a, A, e, E, f and F specifiers: this is the number of digits to be printed after the decimal point (by default, this is 6).
For g and G specifiers: This is the maximum number of significant digits to be printed.
For s: this is the maximum number of characters to be printed. By default all characters are printed until the ending null character is encountered.
If the period is specified without an explicit value for precision, 0 is assumed.
	
+ `.*` :	
> The precision is not specified in the format string, but as an additional integer value argument preceding the argument that has to be formatted. 

翻译过来就是，`.number`和 `.*` 都表示输出精度。

比如

```
#include <stdio.h>

int main() {
    int precision = 8;
    int biggerPrecision = 16;
    const char *greetings = "Hello world";

    printf("|%.8s|\n", greetings);
    printf("|%.*s|\n", precision , greetings);
    printf("|%16s|\n", greetings);
    printf("|%*s|\n", biggerPrecision , greetings);

    return 0;
}
```
得到的输出结果为：
```
|Hello wo|
|Hello wo|
|     Hello world|
|     Hello world|
```

# 参考
1. [printf](http://www.cplusplus.com/reference/cstdio/printf/)
2. [What does “%.*s” mean in printf?](https://stackoverflow.com/questions/7899119/what-does-s-mean-in-printf)