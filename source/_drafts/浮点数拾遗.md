---
title: 浮点数拾遗
tags:
- float point
---
对浮点数做知识点整理，包括表示方式和与十进制转换。  
<!-- more -->

对浮点数的介绍，在《深入理解计算机系统》一书中的第二章 信息的表示和处理 就有详细介绍，只是我当时没有用心去读，再加上比较菜，完全没有深入理解。  
随着GPU双精度浮点数计算能力越来越强，用双精度浮点数计算成为了趋势，这方面的知识得补一补。  

# 表示形式

表示形式为：

$$ V=(-1)^{s} \times M \times 2^{E} $$

+ `s` 表示符号位，只有1比特；
+ M表示有效数字
+ E是指数或阶码。



IEEE754标准中规定float单精度浮点数在机器中表示用 1 位表示数字的符号，用 8 位来表示指数，用23 位来表示尾数，即小数部分。  

就单精度浮点来说：

```
S EEEEEEEE FFFFFFFFFFFFFFFFFFFFFFF
0 1      8 9                    31
```

对于double双精度浮点数，用 1 位表示符号，用 11 位表示指数，52 位表示尾数，其中指数域称为阶码。  
双精度浮点数：

```
S EEEEEEEEEEE FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
0 1        11 12                                                63
```

这里总结了单双精度浮点数的区别：

+ [What’s the difference between a single precision and double precision floating point operation?](https://stackoverflow.com/a/801146)

# 表示范围

+ 这里 E 既不是全0 又不是全1时。
阶码的值初始化有偏置bias，值为 $2^{k-1}-1$，单精度的值为$2^{8-1}-1 = 127$，双精度为 $2^{11-1}-1 = 1023$，实际上阶码的值为 $E = e-Bias$，由此产生的范围为*-126 ~ +127*和 *-1022 ~ +1023* 。  
尾码(Mantissa)小数为f，表示为 $0.f_{n-1}f_{n-2}...f_{0}$ ，而 $M=1+f$，这其中有一个隐含的以1开头的表示。 

+ 当阶码全为0  

这时候，阶码值E=1-bias；M=f，不包含隐含的开头1。  

+ 当阶码全为1

当尾码全为0时，s=0,得到正无穷；s=1，得到负无穷。

对于双精度浮点数而言：  

If E=2047 and F is nonzero, then V=NaN ("Not a number")
If E=2047 and F is zero and S is 1, then V=-Infinity
If E=2047 and F is zero and S is 0, then V=Infinity
If 0<E<2047 then V=(-1)**S * 2 ** (E-1023) * (1.F) where "1.F" is intended to represent the binary number created by prefixing F with an implicit leading 1 and a binary point.
If E=0 and F is nonzero, then V=(-1)**S * 2 ** (-1022) * (0.F) These are "unnormalized" values.
If E=0 and F is zero and S is 1, then V=-0
If E=0 and F is zero and S is 0, then V=0

为方便理解，需要跟十进制转换后加深理解，两个相互转换的方法见下方：

+ [Decimal to Floating-Point Conversions](http://sandbox.mc.edu/~bennet/cs110/flt/dtof.html)
+ [Float to Decimal Conversion](http://sandbox.mc.edu/~bennet/cs110/flt/ftod.html)

在线实时转换： <http://www.binaryconvert.com/result_double.html> 。

补充材料：
+ [Binary Fractions and Floating Point!](https://ryanstutorials.net/binary-tutorial/binary-floating-point.php)
