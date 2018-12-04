---
title: 密码学EXM?
date: 2017-08-06 10:41:42
tags:
- cryptography
- 国产密码
categories:
- security
---


搞安全怎么能离开密码学，虽然理论性那么强，虽然我啃不动。
<!-- more -->

![](../密码学EXM/exm.jpg)

凯撒和栅栏密码

Cipher Block

# base64/32/16编码

原来仅仅听过base64，后来还听到了base32, base16。其实原理都一样，base64、base32、base16可以分别编码转化8位字节为6位、5位、4位。这里重点介绍base64。
Base64常用于在通常处理文本数据的场合，表示、传输、存储一些二进制数据。包括MIME的email，email via MIME,在XML中存储复杂数据。

编码原理：Base64编码要求把3个8位字节转化为4个6位的字节，之后在6位的前面补两个0，形成8位一个字节的形式，6位2进制能表示的最大数是2的6次方是64，这也是为什么是64个字符(A-Z,a-z，0-9，+，/这64个编码字符，=号不属于编码字符，而是填充字符)的原因，这样就需要一张映射表。

python的base64模块用于base64/32/16编码和解码。
```
import base64
s="test"
t = base64.b64encode(s)
print t
print base64.b64decode(t)
```


RC4


## Diffie–Hellman key exchange [[2]](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)

Diffie–Hellman key exchange，迪菲-赫尔曼密钥交换，是一种安全协议。它能够让通信双方在没有对方任何预先信息的前提下通过不安全信道进行密钥交换。它是无认证的密钥交换协议。目的是创建一个可以用于公共信道上安全通信的共享秘密（shared secret）。

![Diffie-Hellman流程图](../密码学EXM/Diffie-Hellman-Schlüsselaustausch.svg)

1. 通信双方爱丽丝A和鲍勃B两人，再通信前约定好生成元g和质数p。（此g可以被攻击者捕获）
2. 爱丽丝A随机选择一个自然数a并且将g^a mod p发送给鲍勃B。
3. 鲍勃B随机选择一个自然数b并且将g^b mod p 发送给爱丽丝A。
4. 爱丽丝A计算(g^b mod p)^a mod p。
5. 鲍勃B计算(g^a mod p)^b mod p。
6. 爱丽丝A和鲍勃B最终得到了相同的值，协商出的群元素g^(ab)作为共享密钥。



## 分组密码工作模式

分组（block）密码的工作模式（mode of operation）允许使用同一个分组密码密钥对多于一块的数据进行加密，并保证其安全性。
常用模式有以下几块：

### 电子密码本（Electronic codebook，ECB）

### 密码块链接（CBC，Cipher-block chaining）

### 填充密码块链接 （PCBC，Propagating cipher-block chaining）

填充密码块链接 （PCBC，Propagating cipher-block chaining）或称为明文密码块链接（Plaintext cipher-block chaining）。

### 密文反馈（CFB，Cipher feedback）

### 输出反馈模式（Output feedback, OFB）


## AEAD(Authenticated Encryption with Associated Data)

# AES

AES作为DES的升级版本，是当今主流的对称加密算法。
AES包括加解密(encrypt/decrypt)和轮密钥生成(key shedule)。
加解密涉及四个操作：SubBytes(字节替换)、ShiftRows(行移位)、MixColumns(列混淆)、AddRoundKey(轮密钥加)。在最后一轮不进行MixColumns。
轮密钥生成：

1. 将128位种子密钥按照列进行排列，其中**w0**=k0 k1 k2 k3。

    
|w0 | w1| w2| w3|
|-- |--| --| ---|
|k0 |k4| k8| k12|
|k1 |k5| k9| k13|
|k2 |k6| k10| k14|
|k3 |k7| k11| k15|

2. 设j是整数并且j属于[4, 43]，若j%4=0,w[j]=w[j-4]⊕g(w[j-1]),否则w[j]=w[j-4]⊕w[j-1]。
3. 函数g(w)的操作为
    1. 将w循环左移8位。
    2. 分别对w的4个字节做S盒(S-Box)置换；
    3. 与32比特的常量（RC[j/4],0,0,0）进行异或。Rc={0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36}

AES加密算法的动态演示
<https://coolshell.cn/wp-content/uploads/2010/10/rijndael_ingles2004.swf>

# RSA

给定一个正整数m，以及两个整数a,b，如果a-b被m整除，则称a与b模m同余，记作a=b(mod m)，否则称a与b模m不同余，记作a!=b(mod m)。

## RSA算法流程

1. 随机生成两个素数: p、q；
2. 计算 m=(p-1)\*(q-1)，n=p\*q；
3. 随机取值e，使e与m互素；
4. 计算e对m的模逆，e*d=1(mod m)；
5. (e, n)为公钥，(d, n)为私钥。

## 公钥加密

```
C = M**e mod n
```

## 私钥解密
```
M = C**d mod n
```


# 国产密码算法

国产密码算法（国密算法）是指国家密码局认定的`国产商用密码算法`，在金融领域目前主要使用公开的SM2、SM3、SM4三类算法，分别是非对称算法、哈希算法和对称算法。 其中`SM`代表“商密”，即用于商用的、不涉及国家秘密的密码技术。

## SM2椭圆曲线公钥密码算

SM2椭圆曲线公钥密码算法是我国自主设计的公钥密码算法，包括SM2-1椭圆曲线数字签名算法，SM2-2椭圆曲线密钥交换协议，SM2-3椭圆曲线公钥加密算法，分别用于实现数字签名密钥协商和数据加密等功能。SM2算法与RSA算法不同的是，SM2算法是基于椭圆曲线上点群离散对数难题，相对于RSA算法，256位的SM2密码强度已经比2048位的RSA密码强度要高。

## SM3杂凑算法
SM3杂凑算法是我国自主设计的密码杂凑算法，适用于商用密码应用中的数字签名和验证消息认证码的生成与验证以及随机数的生成，可满足多种密码应用的安全需求。为了保证杂凑算法的安全性，其产生的杂凑值的长度不应太短，例如MD5输出128比特杂凑值，输出长度太短，影响其安全性SHA-1算法的输出长度为160比特，SM3算法的输出长度为256比特，因此SM3算法的安全性要高于MD5算法和SHA-1算法。

## SM4分组密码算法

SM4分组密码算法是我国自主设计的分组对称密码算法，用于实现数据的加密/解密运算，以保证数据和信息的机密性。要保证一个对称密码算法的安全性的基本条件是其具备足够的密钥长度，SM4算法与AES算法具有相同的密钥长度分组长度128比特，因此在安全性上高于3DES算法。

## 祖冲之序列密码算法



# 参考文献
1. [安全体系（一）—— DES算法详解](http://www.cnblogs.com/songwenlong/p/5944139.html)
2. [迪菲-赫尔曼密钥交换](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)
3. [分组密码工作模式](https://zh.wikipedia.org/wiki/%E5%88%86%E7%BB%84%E5%AF%86%E7%A0%81%E5%B7%A5%E4%BD%9C%E6%A8%A1%E5%BC%8F)
4. [密码算法详解——AES](http://www.cnblogs.com/luop/p/4334160.html)
5. [密码标准应用指南](http://www.gmbz.org.cn/upload/2018-03-24/1521879142922000396.pdf)


