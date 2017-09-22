---
title: 密码学EXM?
date: 2017-08-06 10:41:42
tags:
category:
- security
- 密码学
---


搞安全怎么能离开密码学，虽然理论性那么强，虽然我啃不动。
<!-- more -->

![](../密码学EXM/exm.jpg)

凯撒和栅栏密码

Cipher Block



RC4 DES AES


## Diffie–Hellman key exchange [[2]](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)

Diffie–Hellman key exchange，迪菲-赫尔曼密钥交换，是一种安全协议。它能够让通信双方在没有对方任何预先信息的前提下通过不安全信道进行密钥交换。它是无认证的密钥交换协议。目的是创建一个可以用于公共信道上安全通信的共享秘密（shared secret）。

![Diffie-Hellman流程图](../密码学EXM/400px-Diffie-Hellman-Schlüsselaustausch.svg.png)

1. 通信双方爱丽丝和鲍勃两人，再通信前约定好生成元g和质数p。（此g可以被攻击者捕获）
2. 爱丽丝随机选择一个自然数a并且将g^a mod p发送给鲍勃。
3. 鲍勃随机选择一个自然数b并且将g^b mod p 发送给爱丽丝。
4. 爱丽丝计算(g^b mod p)^a mod p。
5. 鲍勃计算(g^a mod p)^b mod p。
6. 爱丽丝和鲍勃最终得到了相同的值，协商出的群元素g^(ab)作为共享密钥。



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




## 参考文献
[1] [安全体系（一）—— DES算法详解](http://www.cnblogs.com/songwenlong/p/5944139.html)

[2] [迪菲-赫尔曼密钥交换](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)

[3] [分组密码工作模式](https://zh.wikipedia.org/wiki/%E5%88%86%E7%BB%84%E5%AF%86%E7%A0%81%E5%B7%A5%E4%BD%9C%E6%A8%A1%E5%BC%8F)


