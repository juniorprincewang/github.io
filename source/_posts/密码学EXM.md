---
title: 密码学EXM?
date: 2017-08-06 10:41:42
mathjax: true
tags:
- cryptography
- 国产密码
- ECDSA
- SM
- RSA
- AES
categories:
- [security,crypto]
---


搞安全怎么能离开密码学。武功再高，也怕菜刀。  
骐骥一跃，不能十步。
<!-- more -->

![](../密码学EXM/exm.jpg)


# tutorials  

+ [Practical Cryptography for Developers](https://cryptobook.nakov.com/)  
本书算是实用密码学实战，以python开发，涉及了 **hashes** (like SHA-3 and BLAKE2), **MAC codes** (like HMAC and GMAC), **key derivation functions** (like Scrypt, Argon2), **key agreement protocols** (like DHKE, ECDH), **symmetric ciphers** (like AES and ChaCha20, cipher block modes, authenticated encryption, AEAD, AES-GCM, ChaCha20-Poly1305), **asymmetric ciphers and public-key cryptosystems** (RSA, ECC, ECIES), **elliptic curve cryptography** (ECC, secp256k1, curve25519), **digital signatures** (ECDSA and EdDSA), **secure random numbers** (PRNG, CSRNG) and **quantum-safe cryptography** 。  


# 凯撒和栅栏密码

`Cipher Block` ： 分组密码
`nonce` : [Nonce](https://en.wikipedia.org/wiki/Cryptographic_nonce) 是一个在加密通信只能使用一次的数字。在认证协议中，它往往是一个随机或伪随机数，以避免重放攻击。  
Nonce也用于 *流密码* 以确保安全。如果需要使用相同的密钥加密一个以上的消息，就需要Nonce来确保不同的消息与该密钥加密的密钥流不同。

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


# stream cipher 

## RC4 (arch4)

不安全的流密码算法，已经被TLS弃用。 [rfc7465 Prohibiting RC4 Cipher Suites](https://tools.ietf.org/html/rfc7465)  


## Salsa20

一种新的流加密算法，由 Dan Bernstein 设计。根据内部轮数分为  Salsa20/12 和 Salsa20/8 。
基于 add-rotate-xor（ARX）操作。

优势：

用户可以在恒定时间内寻求输出流中的任何位置。它可以在现代x86处理器中提供约每4–14次循环周期一字节的速度，并具有合理的硬件性能。可以抵御侧信道攻击。

## ChaCha

也是由Dan Bernstein 设计的新型的流加密算法。 根据轮数不同分为：ChaCha8,ChaCha12,ChaCha20。

[Snuffle 2005: the Salsa20 encryption function](https://cr.yp.to/snuffle.html)

# KDF

# 填充模式

## 分组密码工作模式 mode of operation

分组（block）密码的工作模式（mode of operation）允许使用同一个分组密码密钥对多于一块的数据进行加密，并保证其安全性。
常用模式有以下几块：

### 电子密码本（Electronic codebook，ECB）

讲消息分成组，每组单独加密。

**缺点**  

+ Visual inspection of an encrypted stream  

本方法的缺点在于同样的明文块会被加密成相同的密文块；因此，它不能很好的隐藏数据模式。在某些场合，这种方法不能提供严格的数据保密性，因此并不推荐用于密码协议中。
+ Encryption oracle attack  
加密预言攻击，oracle是用于计算用的黑盒子，称为“预言机”。
比如对于 `C = ECB(k, m|S)` ，敌手就可以选择m长度为 len(block)-1 大小，那么整个块为 m|s0，敌手可以遍历最终匹配到s0，以此类推获得整个密文对应的明文。  

### 密码块链接（CBC，Cipher-block chaining）

### 计数器模式（CTR，Counter Mode）

### 填充密码块链接 （PCBC，Propagating cipher-block chaining）

填充密码块链接 （PCBC，Propagating cipher-block chaining）或称为明文密码块链接（Plaintext cipher-block chaining）。

### 密文反馈（CFB，Cipher feedback）

### 输出反馈模式（Output feedback, OFB）

### GCM

[SP 800-38D:Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC](https://csrc.nist.gov/publications/detail/sp/800-38d/final)  

## padding

### PKCS#5/PKCS#7 padding

PKCS是 Public Key Cryptography Standards 的简称。  

PKCS#5 是基于口令的加密标准，目前版本是 2.1。  
它将输入按照BlockSize=8字节进行分组，最后一组要填充成8字节。
加入口令长度为 $x$，则填充数据是 $8-(x%8)$，每个padding的字节值是 $8-(x%8)$；若口令长度恰好为8的整数倍，仍需要在后面增添一组，每个元素为0x08。  

这么做的目的：在解密时，根据密文的最后一位来确定填充字节数。因此如果原明文是8的整数倍，仍在末尾填充一组。    

[rfc8018 - PKCS #5: Password-Based Cryptography Specification Version 2.1](https://tools.ietf.org/html/rfc8018)


## Message Authentication Code(MAC)

### Hash-based Message Authentication Code(HMAC)

[rfc4418 UMAC: Message Authentication Code using Universal Hashing](https://www.ietf.org/rfc/rfc4418.txt)  


### AEAD(Authenticated Encryption with Associated Data)

# Block ciphers

分组密码算法

## AES

AES作为DES的升级版本，是当今主流的对称加密算法。
AES选取的分组长度为128比特，保持不变，而密钥长度可改变为128比特、192比特和256比特。
AES包括加解密(encrypt/decrypt)和轮密钥生成(key shedule)。
加解密涉及四个操作：SubBytes(字节替换)、ShiftRows(行移位)、MixColumns(列混淆)、AddRoundKey(轮密钥加)。在最后一轮不进行MixColumns。

可以参考 [FIPS 197，AES](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.197.pdf)。里面有详实的标准介绍。

### AES 加密

加密算法流程为：
```
Cipher(byte in[4*Nb], byte out[4*Nb], word w[Nb*(Nr+1)])
begin
	byte state[4,Nb]
	state = in
	AddRoundKey(state, w[0, Nb-1]) // See Sec. 5.1.4
	for round = 1 step 1 to Nr–1
		SubBytes(state) // See Sec. 5.1.1
		ShiftRows(state) // See Sec. 5.1.2
		MixColumns(state) // See Sec. 5.1.3
		AddRoundKey(state, w[round*Nb, (round+1)*Nb-1])
	end for
	SubBytes(state)
	ShiftRows(state)
	AddRoundKey(state, w[Nr*Nb, (Nr+1)*Nb-1])
	out = state
end
```
+ `State` 一个4行的矩阵，每行包括Nb个字节。用于行移位和列混淆。
+ `Nb` 组成 `State` 的列（一列4个字节，共32位）数量。这里取4。
+ `Nk` 表示密钥长度，32位字节的数量。对于128，192，256长度的密钥来说，Nk分别取4, 6, 8。
+ `Nr` 轮数量，对于128，192，256长度的密钥来说，Nr分别取10, 12, 14。

|密钥算法	|密钥长度Nk字节 | 分组长度Nb字节 | 轮数Nr |
|--------	|--------------|---------------|--------|
|AES-128	| 4 			|			 4 | 	10	|
|AES-192 	| 6				| 4				| 12	|
|AES-256 	| 8 			|4 				| 	14	|

#### SubBytes

SubBytes，将原 State中的每个字符转换成S-Box中对应下标的元素。 即 `State[i,j] = s_box[State[i,j]]` 。
```
uint8_t s_box[256] = {
	// 0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, // 0
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, // 1
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, // 2
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, // 3
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, // 4
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, // 5
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, // 6
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, // 7
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, // 8
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, // 9
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, // a
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, // b
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, // c
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, // d
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, // e
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};// f
```

#### ShiftRows 

ShiftRows，将State数组按照行 依次向左移位0字节，1字节，2字节，3字节。
```
Row0: s0  s4  s8  s12   <<< 0 byte
Row1: s1  s5  s9  s13   <<< 1 byte
Row2: s2  s6  s10 s14   <<< 2 bytes
Row3: s3  s7  s11 s15   <<< 3 bytes
```
#### MixColumns

MixColumns: 利用GF(2^8)域上算术特性的一个代替，同样用于提供算法的扩散性。
```
[02 03 01 01]   [s0  s4  s8  s12]
[01 02 03 01] . [s1  s5  s9  s13]
[01 01 02 03]   [s2  s6  s10 s14]
[03 01 01 02]   [s3  s7  s11 s15]
```
而此处的乘法和加法都是定义在GF(2^8)上的, 将某个字节所对应的值乘以2，其结果就是将该值的二进制位左移一位，如果原始值的最高位为1，则还需要将移位后的结果异或00011011。
乘法对加法满足分配率。
这里计算起来比较麻烦。但是如果用查表的话，速度会提升不少。
[有限域 GF(2^8) 上的乘法改用查表的方式实现](https://blog.csdn.net/lisonglisonglisong/article/details/41909813)
```
byte Mul_02[256] = {
	0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x10,0x12,0x14,0x16,0x18,0x1a,0x1c,0x1e,
	0x20,0x22,0x24,0x26,0x28,0x2a,0x2c,0x2e,0x30,0x32,0x34,0x36,0x38,0x3a,0x3c,0x3e,
	0x40,0x42,0x44,0x46,0x48,0x4a,0x4c,0x4e,0x50,0x52,0x54,0x56,0x58,0x5a,0x5c,0x5e,
	0x60,0x62,0x64,0x66,0x68,0x6a,0x6c,0x6e,0x70,0x72,0x74,0x76,0x78,0x7a,0x7c,0x7e,
	0x80,0x82,0x84,0x86,0x88,0x8a,0x8c,0x8e,0x90,0x92,0x94,0x96,0x98,0x9a,0x9c,0x9e,
	0xa0,0xa2,0xa4,0xa6,0xa8,0xaa,0xac,0xae,0xb0,0xb2,0xb4,0xb6,0xb8,0xba,0xbc,0xbe,
	0xc0,0xc2,0xc4,0xc6,0xc8,0xca,0xcc,0xce,0xd0,0xd2,0xd4,0xd6,0xd8,0xda,0xdc,0xde,
	0xe0,0xe2,0xe4,0xe6,0xe8,0xea,0xec,0xee,0xf0,0xf2,0xf4,0xf6,0xf8,0xfa,0xfc,0xfe,
	0x1b,0x19,0x1f,0x1d,0x13,0x11,0x17,0x15,0x0b,0x09,0x0f,0x0d,0x03,0x01,0x07,0x05,
	0x3b,0x39,0x3f,0x3d,0x33,0x31,0x37,0x35,0x2b,0x29,0x2f,0x2d,0x23,0x21,0x27,0x25,
	0x5b,0x59,0x5f,0x5d,0x53,0x51,0x57,0x55,0x4b,0x49,0x4f,0x4d,0x43,0x41,0x47,0x45,
	0x7b,0x79,0x7f,0x7d,0x73,0x71,0x77,0x75,0x6b,0x69,0x6f,0x6d,0x63,0x61,0x67,0x65,
	0x9b,0x99,0x9f,0x9d,0x93,0x91,0x97,0x95,0x8b,0x89,0x8f,0x8d,0x83,0x81,0x87,0x85,
	0xbb,0xb9,0xbf,0xbd,0xb3,0xb1,0xb7,0xb5,0xab,0xa9,0xaf,0xad,0xa3,0xa1,0xa7,0xa5,
	0xdb,0xd9,0xdf,0xdd,0xd3,0xd1,0xd7,0xd5,0xcb,0xc9,0xcf,0xcd,0xc3,0xc1,0xc7,0xc5,
	0xfb,0xf9,0xff,0xfd,0xf3,0xf1,0xf7,0xf5,0xeb,0xe9,0xef,0xed,0xe3,0xe1,0xe7,0xe5
};
 
byte Mul_03[256] = {
	0x00,0x03,0x06,0x05,0x0c,0x0f,0x0a,0x09,0x18,0x1b,0x1e,0x1d,0x14,0x17,0x12,0x11,
	0x30,0x33,0x36,0x35,0x3c,0x3f,0x3a,0x39,0x28,0x2b,0x2e,0x2d,0x24,0x27,0x22,0x21,
	0x60,0x63,0x66,0x65,0x6c,0x6f,0x6a,0x69,0x78,0x7b,0x7e,0x7d,0x74,0x77,0x72,0x71,
	0x50,0x53,0x56,0x55,0x5c,0x5f,0x5a,0x59,0x48,0x4b,0x4e,0x4d,0x44,0x47,0x42,0x41,
	0xc0,0xc3,0xc6,0xc5,0xcc,0xcf,0xca,0xc9,0xd8,0xdb,0xde,0xdd,0xd4,0xd7,0xd2,0xd1,
	0xf0,0xf3,0xf6,0xf5,0xfc,0xff,0xfa,0xf9,0xe8,0xeb,0xee,0xed,0xe4,0xe7,0xe2,0xe1,
	0xa0,0xa3,0xa6,0xa5,0xac,0xaf,0xaa,0xa9,0xb8,0xbb,0xbe,0xbd,0xb4,0xb7,0xb2,0xb1,
	0x90,0x93,0x96,0x95,0x9c,0x9f,0x9a,0x99,0x88,0x8b,0x8e,0x8d,0x84,0x87,0x82,0x81,
	0x9b,0x98,0x9d,0x9e,0x97,0x94,0x91,0x92,0x83,0x80,0x85,0x86,0x8f,0x8c,0x89,0x8a,
	0xab,0xa8,0xad,0xae,0xa7,0xa4,0xa1,0xa2,0xb3,0xb0,0xb5,0xb6,0xbf,0xbc,0xb9,0xba,
	0xfb,0xf8,0xfd,0xfe,0xf7,0xf4,0xf1,0xf2,0xe3,0xe0,0xe5,0xe6,0xef,0xec,0xe9,0xea,
	0xcb,0xc8,0xcd,0xce,0xc7,0xc4,0xc1,0xc2,0xd3,0xd0,0xd5,0xd6,0xdf,0xdc,0xd9,0xda,
	0x5b,0x58,0x5d,0x5e,0x57,0x54,0x51,0x52,0x43,0x40,0x45,0x46,0x4f,0x4c,0x49,0x4a,
	0x6b,0x68,0x6d,0x6e,0x67,0x64,0x61,0x62,0x73,0x70,0x75,0x76,0x7f,0x7c,0x79,0x7a,
	0x3b,0x38,0x3d,0x3e,0x37,0x34,0x31,0x32,0x23,0x20,0x25,0x26,0x2f,0x2c,0x29,0x2a,
	0x0b,0x08,0x0d,0x0e,0x07,0x04,0x01,0x02,0x13,0x10,0x15,0x16,0x1f,0x1c,0x19,0x1a
};
```
#### AddRoundKey

AddRoundKey, 将State和密钥进行XOR。

轮密钥生成：
```
KeyExpansion(byte key[4*Nk], word w[Nb*(Nr+1)], Nk)
begin
	word temp
	i = 0
	while (i < Nk)
		w[i] = word(key[4*i], key[4*i+1], key[4*i+2], key[4*i+3])
		i = i+1
	end while
	i = Nk
	while (i < Nb * (Nr+1)]
		temp = w[i-1]
		if (i mod Nk = 0)
			temp = SubWord(RotWord(temp)) xor Rcon[i/Nk]
		else if (Nk > 6 and i mod Nk = 4)
			temp = SubWord(temp)
		end if
		w[i] = w[i-Nk] xor temp
		i = i + 1
	end while
end
```
1. 将128位种子密钥按照列进行排列，其中**w0**=k0 k1 k2 k3。

    
|w0 | w1| w2| w3|
|-- |--| --| ---|
|k0 |k4| k8| k12|
|k1 |k5| k9| k13|
|k2 |k6| k10| k14|
|k3 |k7| k11| k15|

2. 设j是整数并且j属于[4, 43]，若j%4=0,w[j]=w[j-4]⊕g(w[j-1]),否则w[j]=w[j-4]⊕w[j-1]。
	w[j]是前一列w[j-1]与上一轮w[j-Nb]异或的结果，如果是首列j%4==0，那么需要对它前一列w[j-1]做g(w)处理。
3. 函数g(w)的操作为
    1. 将w循环左移8位。（仅对w循环）
    2. 分别对w的4个字节做S盒(S-Box)置换；
    3. 与32比特的常量（RC[j/4],0,0,0）进行异或。Rc={0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36}

AES加密算法的动态演示
<https://coolshell.cn/wp-content/uploads/2010/10/rijndael_ingles2004.swf>
### AES 解密

解密的话也需要四个步骤：InvShiftRows(逆行移位), InvSubBytes(逆字节替换),InvMixColumns(逆列混淆),和 AddRoundKey(轮密钥加)。
但是解密的顺序略有不同。 `w` 为轮密钥。


```
InvCipher(byte in[4*Nb], byte out[4*Nb], word w[Nb*(Nr+1)])
begin
	byte state[4,Nb]
	state = in
	AddRoundKey(state, w[Nr*Nb, (Nr+1)*Nb-1]) // See Sec. 5.1.4
	for round = Nr-1 step -1 downto 1
		InvShiftRows(state) // See Sec. 5.3.1
		InvSubBytes(state) // See Sec. 5.3.2
		AddRoundKey(state, w[round*Nb, (round+1)*Nb-1])
		InvMixColumns(state) // See Sec. 5.3.3
	end for
	InvShiftRows(state)
	InvSubBytes(state)
	AddRoundKey(state, w[0, Nb-1])
	out = state
end
```
#### InvShiftRows

InvShiftRows 只是将 State序列按照 行号， 进行逆向向右依次移动0个字节、1个字节、2个字节、3个字节。
```
		state
Row0: s0  s4  s8  s12   >>> 0 byte
Row1: s1  s5  s9  s13   >>> 1 byte
Row2: s2  s6  s10 s14   >>> 2 bytes
Row3: s3  s7  s11 s15   >>> 3 bytes
```

#### InvSubBytes

InvSubBytes 字节替换用到的逆序S-Box为：
```
	// 0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb, // 0
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, // 1
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e, // 2
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25, // 3
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, // 4
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, // 5
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, // 6
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, // 7
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, // 8
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, // 9
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, // a
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, // b
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, // c
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, // d
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61, // e
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};// f
```

#### InvMixColumns

InvMixColumns 是 MixColumns的逆序，需要用到的矩阵相乘系数为
```
[0e 0b 0d 09]   [s0  s4  s8  s12]
[09 0e 0b 0d] . [s1  s5  s9  s13]
[0d 09 0e 0b]   [s2  s6  s10 s14]
[0b 0d 09 0e]   [s3  s7  s11 s15]
```
[有限域 GF(2^8) 上的乘法改用查表的方式实现](https://blog.csdn.net/lisonglisonglisong/article/details/41909813)
```
byte Mul_09[256] = {
	0x00,0x09,0x12,0x1b,0x24,0x2d,0x36,0x3f,0x48,0x41,0x5a,0x53,0x6c,0x65,0x7e,0x77,
	0x90,0x99,0x82,0x8b,0xb4,0xbd,0xa6,0xaf,0xd8,0xd1,0xca,0xc3,0xfc,0xf5,0xee,0xe7,
	0x3b,0x32,0x29,0x20,0x1f,0x16,0x0d,0x04,0x73,0x7a,0x61,0x68,0x57,0x5e,0x45,0x4c,
	0xab,0xa2,0xb9,0xb0,0x8f,0x86,0x9d,0x94,0xe3,0xea,0xf1,0xf8,0xc7,0xce,0xd5,0xdc,
	0x76,0x7f,0x64,0x6d,0x52,0x5b,0x40,0x49,0x3e,0x37,0x2c,0x25,0x1a,0x13,0x08,0x01,
	0xe6,0xef,0xf4,0xfd,0xc2,0xcb,0xd0,0xd9,0xae,0xa7,0xbc,0xb5,0x8a,0x83,0x98,0x91,
	0x4d,0x44,0x5f,0x56,0x69,0x60,0x7b,0x72,0x05,0x0c,0x17,0x1e,0x21,0x28,0x33,0x3a,
	0xdd,0xd4,0xcf,0xc6,0xf9,0xf0,0xeb,0xe2,0x95,0x9c,0x87,0x8e,0xb1,0xb8,0xa3,0xaa,
	0xec,0xe5,0xfe,0xf7,0xc8,0xc1,0xda,0xd3,0xa4,0xad,0xb6,0xbf,0x80,0x89,0x92,0x9b,
	0x7c,0x75,0x6e,0x67,0x58,0x51,0x4a,0x43,0x34,0x3d,0x26,0x2f,0x10,0x19,0x02,0x0b,
	0xd7,0xde,0xc5,0xcc,0xf3,0xfa,0xe1,0xe8,0x9f,0x96,0x8d,0x84,0xbb,0xb2,0xa9,0xa0,
	0x47,0x4e,0x55,0x5c,0x63,0x6a,0x71,0x78,0x0f,0x06,0x1d,0x14,0x2b,0x22,0x39,0x30,
	0x9a,0x93,0x88,0x81,0xbe,0xb7,0xac,0xa5,0xd2,0xdb,0xc0,0xc9,0xf6,0xff,0xe4,0xed,
	0x0a,0x03,0x18,0x11,0x2e,0x27,0x3c,0x35,0x42,0x4b,0x50,0x59,0x66,0x6f,0x74,0x7d,
	0xa1,0xa8,0xb3,0xba,0x85,0x8c,0x97,0x9e,0xe9,0xe0,0xfb,0xf2,0xcd,0xc4,0xdf,0xd6,
	0x31,0x38,0x23,0x2a,0x15,0x1c,0x07,0x0e,0x79,0x70,0x6b,0x62,0x5d,0x54,0x4f,0x46
};
 
byte Mul_0b[256] = {
	0x00,0x0b,0x16,0x1d,0x2c,0x27,0x3a,0x31,0x58,0x53,0x4e,0x45,0x74,0x7f,0x62,0x69,
	0xb0,0xbb,0xa6,0xad,0x9c,0x97,0x8a,0x81,0xe8,0xe3,0xfe,0xf5,0xc4,0xcf,0xd2,0xd9,
	0x7b,0x70,0x6d,0x66,0x57,0x5c,0x41,0x4a,0x23,0x28,0x35,0x3e,0x0f,0x04,0x19,0x12,
	0xcb,0xc0,0xdd,0xd6,0xe7,0xec,0xf1,0xfa,0x93,0x98,0x85,0x8e,0xbf,0xb4,0xa9,0xa2,
	0xf6,0xfd,0xe0,0xeb,0xda,0xd1,0xcc,0xc7,0xae,0xa5,0xb8,0xb3,0x82,0x89,0x94,0x9f,
	0x46,0x4d,0x50,0x5b,0x6a,0x61,0x7c,0x77,0x1e,0x15,0x08,0x03,0x32,0x39,0x24,0x2f,
	0x8d,0x86,0x9b,0x90,0xa1,0xaa,0xb7,0xbc,0xd5,0xde,0xc3,0xc8,0xf9,0xf2,0xef,0xe4,
	0x3d,0x36,0x2b,0x20,0x11,0x1a,0x07,0x0c,0x65,0x6e,0x73,0x78,0x49,0x42,0x5f,0x54,
	0xf7,0xfc,0xe1,0xea,0xdb,0xd0,0xcd,0xc6,0xaf,0xa4,0xb9,0xb2,0x83,0x88,0x95,0x9e,
	0x47,0x4c,0x51,0x5a,0x6b,0x60,0x7d,0x76,0x1f,0x14,0x09,0x02,0x33,0x38,0x25,0x2e,
	0x8c,0x87,0x9a,0x91,0xa0,0xab,0xb6,0xbd,0xd4,0xdf,0xc2,0xc9,0xf8,0xf3,0xee,0xe5,
	0x3c,0x37,0x2a,0x21,0x10,0x1b,0x06,0x0d,0x64,0x6f,0x72,0x79,0x48,0x43,0x5e,0x55,
	0x01,0x0a,0x17,0x1c,0x2d,0x26,0x3b,0x30,0x59,0x52,0x4f,0x44,0x75,0x7e,0x63,0x68,
	0xb1,0xba,0xa7,0xac,0x9d,0x96,0x8b,0x80,0xe9,0xe2,0xff,0xf4,0xc5,0xce,0xd3,0xd8,
	0x7a,0x71,0x6c,0x67,0x56,0x5d,0x40,0x4b,0x22,0x29,0x34,0x3f,0x0e,0x05,0x18,0x13,
	0xca,0xc1,0xdc,0xd7,0xe6,0xed,0xf0,0xfb,0x92,0x99,0x84,0x8f,0xbe,0xb5,0xa8,0xa3
};
 
byte Mul_0d[256] = {
	0x00,0x0d,0x1a,0x17,0x34,0x39,0x2e,0x23,0x68,0x65,0x72,0x7f,0x5c,0x51,0x46,0x4b,
	0xd0,0xdd,0xca,0xc7,0xe4,0xe9,0xfe,0xf3,0xb8,0xb5,0xa2,0xaf,0x8c,0x81,0x96,0x9b,
	0xbb,0xb6,0xa1,0xac,0x8f,0x82,0x95,0x98,0xd3,0xde,0xc9,0xc4,0xe7,0xea,0xfd,0xf0,
	0x6b,0x66,0x71,0x7c,0x5f,0x52,0x45,0x48,0x03,0x0e,0x19,0x14,0x37,0x3a,0x2d,0x20,
	0x6d,0x60,0x77,0x7a,0x59,0x54,0x43,0x4e,0x05,0x08,0x1f,0x12,0x31,0x3c,0x2b,0x26,
	0xbd,0xb0,0xa7,0xaa,0x89,0x84,0x93,0x9e,0xd5,0xd8,0xcf,0xc2,0xe1,0xec,0xfb,0xf6,
	0xd6,0xdb,0xcc,0xc1,0xe2,0xef,0xf8,0xf5,0xbe,0xb3,0xa4,0xa9,0x8a,0x87,0x90,0x9d,
	0x06,0x0b,0x1c,0x11,0x32,0x3f,0x28,0x25,0x6e,0x63,0x74,0x79,0x5a,0x57,0x40,0x4d,
	0xda,0xd7,0xc0,0xcd,0xee,0xe3,0xf4,0xf9,0xb2,0xbf,0xa8,0xa5,0x86,0x8b,0x9c,0x91,
	0x0a,0x07,0x10,0x1d,0x3e,0x33,0x24,0x29,0x62,0x6f,0x78,0x75,0x56,0x5b,0x4c,0x41,
	0x61,0x6c,0x7b,0x76,0x55,0x58,0x4f,0x42,0x09,0x04,0x13,0x1e,0x3d,0x30,0x27,0x2a,
	0xb1,0xbc,0xab,0xa6,0x85,0x88,0x9f,0x92,0xd9,0xd4,0xc3,0xce,0xed,0xe0,0xf7,0xfa,
	0xb7,0xba,0xad,0xa0,0x83,0x8e,0x99,0x94,0xdf,0xd2,0xc5,0xc8,0xeb,0xe6,0xf1,0xfc,
	0x67,0x6a,0x7d,0x70,0x53,0x5e,0x49,0x44,0x0f,0x02,0x15,0x18,0x3b,0x36,0x21,0x2c,
	0x0c,0x01,0x16,0x1b,0x38,0x35,0x22,0x2f,0x64,0x69,0x7e,0x73,0x50,0x5d,0x4a,0x47,
	0xdc,0xd1,0xc6,0xcb,0xe8,0xe5,0xf2,0xff,0xb4,0xb9,0xae,0xa3,0x80,0x8d,0x9a,0x97
};
 
byte Mul_0e[256] = {
	0x00,0x0e,0x1c,0x12,0x38,0x36,0x24,0x2a,0x70,0x7e,0x6c,0x62,0x48,0x46,0x54,0x5a,
	0xe0,0xee,0xfc,0xf2,0xd8,0xd6,0xc4,0xca,0x90,0x9e,0x8c,0x82,0xa8,0xa6,0xb4,0xba,
	0xdb,0xd5,0xc7,0xc9,0xe3,0xed,0xff,0xf1,0xab,0xa5,0xb7,0xb9,0x93,0x9d,0x8f,0x81,
	0x3b,0x35,0x27,0x29,0x03,0x0d,0x1f,0x11,0x4b,0x45,0x57,0x59,0x73,0x7d,0x6f,0x61,
	0xad,0xa3,0xb1,0xbf,0x95,0x9b,0x89,0x87,0xdd,0xd3,0xc1,0xcf,0xe5,0xeb,0xf9,0xf7,
	0x4d,0x43,0x51,0x5f,0x75,0x7b,0x69,0x67,0x3d,0x33,0x21,0x2f,0x05,0x0b,0x19,0x17,
	0x76,0x78,0x6a,0x64,0x4e,0x40,0x52,0x5c,0x06,0x08,0x1a,0x14,0x3e,0x30,0x22,0x2c,
	0x96,0x98,0x8a,0x84,0xae,0xa0,0xb2,0xbc,0xe6,0xe8,0xfa,0xf4,0xde,0xd0,0xc2,0xcc,
	0x41,0x4f,0x5d,0x53,0x79,0x77,0x65,0x6b,0x31,0x3f,0x2d,0x23,0x09,0x07,0x15,0x1b,
	0xa1,0xaf,0xbd,0xb3,0x99,0x97,0x85,0x8b,0xd1,0xdf,0xcd,0xc3,0xe9,0xe7,0xf5,0xfb,
	0x9a,0x94,0x86,0x88,0xa2,0xac,0xbe,0xb0,0xea,0xe4,0xf6,0xf8,0xd2,0xdc,0xce,0xc0,
	0x7a,0x74,0x66,0x68,0x42,0x4c,0x5e,0x50,0x0a,0x04,0x16,0x18,0x32,0x3c,0x2e,0x20,
	0xec,0xe2,0xf0,0xfe,0xd4,0xda,0xc8,0xc6,0x9c,0x92,0x80,0x8e,0xa4,0xaa,0xb8,0xb6,
	0x0c,0x02,0x10,0x1e,0x34,0x3a,0x28,0x26,0x7c,0x72,0x60,0x6e,0x44,0x4a,0x58,0x56,
	0x37,0x39,0x2b,0x25,0x0f,0x01,0x13,0x1d,0x47,0x49,0x5b,0x55,0x7f,0x71,0x63,0x6d,
	0xd7,0xd9,0xcb,0xc5,0xef,0xe1,0xf3,0xfd,0xa7,0xa9,0xbb,0xb5,0x9f,0x91,0x83,0x8d
};

```

解密的AddRoundKey 与 加密的相同，只是将State和密钥做XOR操作。

### 参考
[aes算法实现](https://github.com/openluopworld/aes_128/blob/master/aes.c)
[aes算法实现](https://github.com/dhuertas/AES/blob/master/aes.c)




# RSA

给定一个正整数m，以及两个整数a,b，如果a-b被m整除，则称a与b模m同余，记作 $ a=b \pmod {m} $，否则称a与b模m不同余，记作 $ a \neq b \pmod {m} $。

**欧拉函数** 

意义是求跟某个数互素，且小于这个数的元素的个数。设数n，那么 $\phi(n)=|Z_n^\*|$ 。  
与n互素且小于n的任意一个数，在计算模n的幂次的时候，等于1的那个最小的幂次。  
即 $ gcd(a,n)=1 $，那么$ a^{\phi(n)}=1 \pmod {n} $。

## RSA算法流程

1. 随机生成等二进制长度的两个素数: $p$、$q$；
2. 计算 $\phi(n)=(p-1)\*(q-1)$，$n=p\*q$；
3. 随机取值$e$，使$e$与 $\phi(n)$ 互素；
4. 计算$e$对 $\phi(n)$ 的模逆，$e*d=1\pmod {\phi(n)}$；
5. $(e, n)$为公钥，$(d, n)$为私钥。

+ 公钥加密

$$ C = M^e \pmod {n} $$

+ 私钥解密

$$ M = C^d \pmod {n} $$

+ RSA signatures  

[RSA Signatures](https://cryptobook.nakov.com/digital-signatures/rsa-signatures)  

1. 计算消息的hash: $h=hash(msg)$
2. 用私钥 $d$ 加密消息hash: $s=h^d \pmod {n}$

+ RSA signatures verification  

1. 计算消息的hash: $h=hash(msg)$
2. 用公钥 $e$ 解密消息hash: $h'=s^e \pmod {n}$
3. 比较 $h$ 与 $h'$ 是否相等

## Modular Exponentiation

RSA 的操作主要是模幂运算，这里有 *Repeated squaring*，*Sliding window*，*Chinese Remainder Theorem (CRT)*，*Montgomery multiplication*，*Karatsuba multiplication* 等。  


### 滑动窗口：sliding windows


Input: $M$; $e$; $n$。
Output: $C = M^e \pmod {n}$.
1. Compute and store $M^w \pmod {n}$ for all $w = 3, 5, 7, ... , 2^d - 1$。
2. Decompose $e$ into zero and nonzero windows $F(i)$ of length $L(F(i))$，for $i = 0, 1, 2, ... , p - 1$。
3. $C := M^{F(p-1)} \pmod {n}$
4. for i = p - 2 down to 0
	1. $C := C^{2^{L(F(i))}} \pmod {n}$
    2. if (F(i) != 0), then $C := C * M^{F(i)} \pmod {n}$
5. return C


### CRT calculation  

+ [Implementation of RSA Algorithm with Chinese Remainder Theorem for Modulus N 1024 Bit and 4096 Bit](https://www.cscjournals.org/manuscript/Journals/IJCSS/Volume10/Issue5/IJCSS-1289.pdf)

简介中国剩余定理（Chinese Remainder Theorem，CRT）：  

p和q是互相独立的大素数，n为p*q，对于任意(m1, m2), (0<=m1< p, 0<=m2< p)
必然存在一个唯一的m ,0<=m< n
使得
$$m1 = m \pmod {p}$$
$$m2 = m \pmod {q}$$

所以换句话说，给定一个(m1,m2)，其满足上述等式的m必定唯一存在。


通过中国剩余定理计算RSA。  
需要的参数：  
+ Modulus ($n=pq$)
+ Public exponent ($e$，通常为3, 17 or 65537)
+ Private exponent ($d=e^{−1}\pmod {\phi(n)}$)
+ First prime ($p$)
+ Second prime ($q$)
+ First exponent, used for Chinese remainder theorem ($d_P=d\pmod{p−1}$)
+ Second exponent, used for CRT ($d_Q=d\pmod{q−1}$)
+ Coefficient, used for CRT ($q_{inv}=q^{−1}\pmod{p}$)

公钥为 $(e, n)$，私钥为 $(d_P, d_Q, q_{inv}, p, q)$。

+ 公钥加密

$$ C = M^e \pmod {n} $$

+ 私钥解密

这里和常规方法不同。  

1. 计算 $C_p = C \pmod{p}$, $C_q = C \pmod{q}$
2. 计算 $x_1={C_p}^{d_P}\pmod{p}$ , $x_2={C_q}^{d_Q}\pmod{q}$
3. 计算 $h=q_{inv} \times (x_1-x_2)\pmod{p}$
4. 明文 $M = x_2+ {h}\times{q}$


### Montgomery Multiplication



## padding

### PKCS1

### OAEP

OAEP (optimal asymmetric encryption padding)

### PSS

[RSA Algorithom 详尽的介绍](https://www.di-mgt.com.au/rsa_alg.html)

# Paillier同态加密算法(Paillier Homomorphic Encryption)

Paillier加密系统是概率公钥加密系统。基于复合剩余类的困难问题。该加密算法是一种同态加密，满足加法和数乘同态。


## 密钥生成：

+ 随机选择两个大素数 $p$ , $q$ ，即满足 $ gcd(pq,(p−1)(q−1))=1 $，这个属性是为了保证两个质数长度相等, $gcd()$用于计算两个数的最大公约数 。
+ 计算 $n=pq$ , $ \lambda=\mathrm{lcm}(p-1)(q-1)$， $lcm()$ 用于计算最小公倍数。
+ 随机选择一个整数 $g$ , $ g\in \mathbb{Z}_{n^2}^*$ ， 且满足 $gcd(L(g^{\lambda }{\bmod  n}^{2}),n)=1$
+ $\mu =(L(g^{\lambda }{\bmod  n}^{2}))^{-1}{\bmod  n} $，这里 $L$ 被定义为 $L(x)=\frac{x-1}{n} $.

$\mathbb{Z}\_{n^2}$ 为小于 $n^2$ 的整数集合，而$\mathbb{Z}\_{n^2}^\*$ 为 $\mathbb{Z}\_{n^2}$ 中与 $n^2$ 互质的整数的集合。

**公钥**：$(n, g)$
**私钥**：$(\lambda, \mu)$ 。

## 加密过程：

+ m 是要被加密的明文，在这里 $0 \leq m \lt n$。
+ 随机选择一个整数 $r$, $0 \lt r \lt n $, $r \in \mathbb{Z}^*_{n^2}$, 与$n$ 互质，即 $\mathrm{gcd}(r,n)=1$。  
+ 计算密文：$c=g^{m}\cdot r^{n}{\bmod  n}^{2}$

对于任意明文 $m \in \mathbb{Z}\_{n}$，随机选取的整数 $r$ 不同，得到的密文就不同，但是解密后可以还原出相同的明文 $m$ ，从而保证了m密文的语义安全。

## 解密过程：

+ $c$ 是要解密的密文，$c \in \mathbb{Z}^*_{n^2}$
+ 计算明文：$m=L(c^{\lambda }{\bmod  n}^{2})\cdot \mu {\bmod  n}$

## 证明

$$m=L(c^{\lambda }{\bmod  n}^{2})\cdot \mu {\bmod  n}
=\frac{L(c^{\lambda }{\bmod  n}^{2})}{L(g^{\lambda }{\bmod  n}^{2})} {\bmod n}$$

Carmichael’s theorem:
$$c^{\lambda}=(g^m \cdot ⋅r^n )^{λ} = g^{mλ} \cdot r^{nλ} =g^{mλ} $$

多项式的幂次项:
$$(1+n)^x ≡1+nx \bmod {n^2}$$
 
$$g^{mλ} =((1+n)^α\cdot β^n )^{λm} =(1+n)^{αλm} \cdot β^{nλm} ≡(1+αλmn) \bmod {n^2}$$

应用 $L()$ 函数

$$\frac{L(c^{\lambda }{\bmod  n}^{2})}{L(g^{\lambda }{\bmod  n}^{2})} {\bmod n} = \frac{L(1+αλmn)}{L(1+αλn)} {\bmod n}=\frac{αλmn}{αλn} {\bmod n} = m$$

## 同态的性质

### 同态加法

+ 两个密文的乘机将解密为对应的明文之和

$$ D(E(m\_{1},r\_{1})\cdot E(m\_{2},r\_{2}){\bmod  n}^{2})=m\_{1}+m\_{2}{\bmod  n} $$


+ 一个密文与以 $g$为底、明文为幂的数相乘将解密为对应明文之和 

$$ D(E(m\_{1},r\_{1})\cdot g^{m\_{2}}{\bmod  n}^{2})=m\_{1}+m\_{2}{\bmod  n} $$

### 同态乘法

+ 密文的明文幂将倍解密为对应明文的乘积

$$ D(E(m\_{1},r\_{1})^{m\_{2}}{\bmod  n}^{2})=m\_{1}m\_{2}{\bmod  n} $$
$$ D(E(m\_{2},r\_{2})^{m\_{1}}{\bmod  n}^{2})=m\_{1}m\_{2}{\bmod  n}$$
更一般地，
$$ D(E(m\_{1},r\_{1})^{k}{\bmod  n}^{2})=km\_{1}{\bmod  n}$$

参考：
[Paillier cryptosystem](https://en.wikipedia.org/wiki/Paillier_cryptosystem#Key_generation)  
[Paillier算法详解及Java实现](https://blog.csdn.net/qq_41199831/article/details/81096625)  
[Paillier Cryptosystem](https://blog.csdn.net/caorui_nk/article/details/83305709)  

## 百万富翁安全比富问题


安全多方计算起源于百万富翁问题（比谁更富有但不泄露财产），即两方安全函数计算（two-party secure function evaluation，2P-SFE）。

安全多方计算在针对无可信第三方情况下，可让多个数据所有者在联合的数据上进行协同计算以提取数据的价值，而不泄露每个数据所有者的原始数据。在安全多方计算里节点通过隐私计算协议完成加密运算，核心思想是不让其他节点看到保密信息，确保在计算过程中对输入的数据保密，在不暴露明文的前提下完成某种运算。  
安全多方计算在云数据安全和隐私保护方面开始应用。

回到百万富翁比富问题，比较常见的是利用同态加密来实现。  
同样假设两个富翁Alice与Bob，财富分别是$a$ 和 $b$，安全比较过程如下：

第一步：Bob生成两个非常大的随机正整数 $x$ 和 $y$ ，但是并不公开只有他自己知道；

第二步：Alice生成一对属于自己的密钥(公钥是$pub$，私钥是$pri$)，用公钥加密自己的财富的到 $E(a)$ ，并将它和公钥一起公布出去；

第三步：Bob得到Alice公布出来的数据以后，首先用Alice公钥计算出 $E(b⋅x+y)$，然后用Paillier算法的同态属性计算出 $E(a⋅x+y)=E(a)x⋅E(y)$，并将这两个结果也公布出去；

第四步：Alice得到Bob公布出来的计算结果以后，用自己的私钥分别反解出 $A=a⋅x+y$ 和 $B=b⋅x+y$ 的值。Alice虽然对 $x$ 、 $y$ 和 $b$ 一无所知，但她只要比较 $A$ 和 $B$ 的大小就行了。而对于Bob来说，他对 $A$ 、 $B$ 和 $a$ 也是一无所知，如果他也想要知道相对大小，要么Alice告诉他，要么把角色对换重新执行一遍协议即可。

Python代码可以参考[两个百万富翁如何安全比富](https://www.dploop.org/2017-10-18-zhihu66376147/)



# 零知识证明 Zero-knowledge proof


零知识证明(Zero—Knowledge Proof)，指的是证明者能够在不向验证者提供任何有用的信息的情况下，使验证者相信某个论断是正确的。零知识证明实质上是一种涉及两方或更多方的协议。证明者向验证者证明并使其相信自己知道或拥有某一消息，但证明过程不能向验证者泄漏任何关于被证明消息的信息。大量事实证明，零知识证明在密码学中非常有用。如果能够将零知识证明用于验证，将可以有效解决许多问题。

在密码应用中，Peggy想要向Victor证明她知道在给定群中的给定值的离散对数。比如，对于给定值 $y$ ，素数 $p$ ，生成元 $g$ ，她想证明她知道满足 $g^{x}{\bmod {p}}=y$ 的 $x$ ，而不泄露 $x$ 。 Victor想要确定她是否知道 $x$ 的过程如下。 

1. Peggy第一次计算 $g^{x}{\bmod {p}}=y$ 并将 $y$ 传给 Victor。
2. Peggy选择随机数 $r$ ，并计算 $ C=g^{r}{\bmod {p}} $ 再将计算结果传给 Victor。
3. Victor 向 Peggy请求 $ (x+r){\bmod {(p-1)}} $ ，并且验证 $ (C\cdot y){\bmod {p}}\equiv g^{(x+r){\bmod {(p-1)}}}{\bmod {p}} $ 。
4. Victor 重复向Peggy 请求随机数并作验证。

<https://en.wikipedia.org/wiki/Zero-knowledge_proof>

# 离散对数体系（Discrete Logarithm）

实现离散对数体制的最常用的群是有限域的乘法群的循环子群和椭圆曲线群的循环子群。  

## 困难问题  


+ 离散对数问题discrete logarithm problem  

给定素数 $p$ 和正整数 $g$ ，知道 $g^x \pmod{p}$ 的值，求 $x$ 。  

+ 椭圆曲线上的离散对数问题 elliptic curve discrete logarithm problem  

k为正整数，P 是椭圆曲线上的点，已知 $P^k$ 和 $P$ ，计算 $k=\log_{P}{P^k}$ 。

[Elliptic-Curve Discrete Logarithm Problem (ECDLP)](https://cryptobook.nakov.com/asymmetric-key-ciphers/elliptic-curve-cryptography-ecc#elliptic-curve-discrete-logarithm-problem-ecdlp)  

[离散对数和椭圆曲线加密原理](https://blog.csdn.net/qmickecs/article/details/76585303)  

## D-H

第一个离散对数体制是Diffie-Hellman于1976年提出的密钥协商协议。1984年，ElGamal提出了离散对数公钥加密方案和离散对数签名方案。以后，人们相机提出了离散对数公钥密码的各种变种。
下面介绍基本的ElGamal公钥加密方案和密钥签名方案（DSA）。

`模除`（又称模数、取模操作、取模运算等，英语： `modulo` 有时也称作 `modulus`）得到的是一个数除以另一个数的余数。

Diffie–Hellman key exchange[[2]](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)，迪菲-赫尔曼密钥交换，是一种安全协议。它能够让通信双方在没有对方任何预先信息的前提下通过不安全信道进行密钥交换。它是无认证的密钥交换协议。目的是创建一个可以用于公共信道上安全通信的共享秘密（shared secret）。


![Diffie-Hellman流程图](../密码学EXM/Diffie-Hellman-Schlüsselaustausch.svg)

1. 通信双方爱丽丝A和鲍勃B两人，再通信前约定好生成元g和质数p。（此g可以被攻击者捕获）
2. 爱丽丝A随机选择一个自然数a并且将g^a mod p发送给鲍勃B。
3. 鲍勃B随机选择一个自然数b并且将g^b mod p 发送给爱丽丝A。
4. 爱丽丝A计算(g^b mod p)^a mod p。
5. 鲍勃B计算(g^a mod p)^b mod p。
6. 爱丽丝A和鲍勃B最终得到了相同的值，协商出的群元素g^(ab)作为共享密钥。


## 基于椭圆曲线的DH密钥交换（ECDH）  

ECDH跟DH的流程基本是一致的。

1. 爱丽丝A 和 鲍勃B 约定使用某条椭圆曲线（包括曲线参数，有限域参数以及基点P等）
2. 爱丽丝A 生成私钥 x，计算 $x∗P$ 作为公钥公布出去
3. 鲍勃B 生成私钥 $y$，计算 $y∗P$ 作为公钥公布出去
4. 爱丽丝A 得知 $y∗P$ 后，计算  
$s=x∗(y∗P)=xy∗P$  
5. 鲍勃B 得到 x∗P 后，计算  
$s=y∗(x∗P)=yx∗P$    
6. 双方都得到了相同的密钥的 s，交换完毕

D-H具体实现分为 *基于离散对数* 和 *基于椭圆曲线离散对数* 两种。 两种方法的密钥安全等级如下，对于同样的安全等级，椭圆曲线密钥长度比离散对数密钥长度要小得多。 

|Security level in bits | Discrete log key bits | Elliptic curve key bits |
|-----------------------|	--------------------|-------------------------|
|56 | 512 |112|
|80 | 1024 |160|
|112 | 2048 |224|
|128 | 3072 |256|
|256 | 15360 |512|

D-H容易遭受  man-in-the-middle，即MITM（中间人）攻击。因为消息没有认证。

## ElGamal加密算法

定义可以参见<https://ctf-wiki.github.io/ctf-wiki/crypto/signature/elgamal/>

非常详细的ElGamal加密的教程，给了循环组的例子。
<https://ritter.vg/security_adventures_elgamal.html>

ElGamal加密算法是一个基于 Diffie-Hellman 密钥交换的非对称加密算法。
ElGamal加密算法由三部分组成：密钥生成、加密和解密。

### 密钥生成
密钥生成的步骤如下：

+ Alice利用生成元  ${g}$ 产生一个大素数 $q$,即$g$是$q$的本原根，阶循环群 $G$的有效描述，该循环群的阶为 $q-1$。该循环群需要满足一定的安全性质。[ [本原根的概念对应模q乘法群(需循环群)中的生成元。]]
+ Alice从 $\lbrace1,\ldots ,q-1\rbrace$中随机选择一个 $x$。
+ Alice计算 $h:=g^{x}$。
+ Alice公开 $h$,以及 $G,q,g$的描述作为其公钥，并保留 $x$ 作为其私钥。私钥必须保密。

### 加密

使用Alice的公钥 $(G,q,g,h)$向她加密一条消息 $m$ 的加密算法工作方式如下：

+ Bob从 $\lbrace1,\ldots ,q-1\rbrace$ 随机选择一个 $y$，然后计算 $c_{1}:=g^{y}$。
+ Bob计算共享秘密 $s:=h^{y}$。
+ Bob把他要发送的秘密消息 $m$ 映射为 $G$ 上的一个元素 $m'$。
+ Bob计算 $c_{2}:=m'\cdot s$。
+ Bob将密文 $(c\_{1},c_{2})=(g^{y},m'\cdot h^{y})=(g^{y},m'\cdot (g^{x})^{y})$发送给Alice。

值得注意的是，如果一个人知道了 $m'$，那么它很容易就能知道 $h^{y}$的值。因此对每一条信息都产生一个新的 $y$ 可以提高安全性。所以 $y$ 也被称作临时密钥。

### 解密

利用私钥 $x$ 对密文 $(c\_{1},c_{2})$进行解密的算法工作方式如下：

+ Alice计算共享秘密 $s:=c\_{1}{}^{x}$
然后计算 $m':=c\_{2}\cdot s^{-1}$，并将其映射回明文 $m$，其中 $s^{-1}$ 是 $s$ 在群 $G$ 上的逆元。（例如：如果 $G$ 是整数模n乘法群的一个子群，那么逆元就是模逆元）。
解密算法是能够正确解密出明文的，因为
$c\_{2}\cdot s^{-1}=m'\cdot h^{y}\cdot (g^{xy})^{-1}=m'\cdot g^{xy}\cdot g^{-xy}=m'.$

同样参考 [ElGamal加密算法](https://www.jianshu.com/p/cd36ae7dca47)

## ElGamal签名算法

ElGamal 来说，其签名方案与相应的加密方案具有很大区别。

**补充**： 在同余理论中，模 n 的互质同余类组成一个乘法群，称为整数模 n 乘法群。
> In modular arithmetic, the integers coprime (relatively prime) to n from the set {0,1,... ,n-1} of n non-negative integers form a group under multiplication modulo n, called the multiplicative group of integers modulo n.

**补充**：
扩展欧几里得算法（英语：Extended Euclidean algorithm）是欧几里得算法（又叫辗转相除法）的扩展。已知整数a、b，扩展欧几里得算法可以在求得a、b的最大公约数的同时，能找到整数x、y（其中一个很可能是负数），使它们满足贝祖等式

$ ax + by = \gcd(a, b)$

### 密钥生成

1. 选取一个足够大的素数 $p$（十进制位数不低于 160），以便于在$Z_p$上求解离散对数问题是困难的。
2. 选取整数模 $p$ 乘法群$Z_{p}^{*}$ 的生成元 $g$。
3. 随机选取密钥 $x$，满足 $1 < x < p − 2$，计算 $y = g^x \bmod p$ 。

其中私钥为 ${x}$，公钥为 ${p,g,y}$ 。

### 签名

如果A 要对消息 $m$ 进行签名 $sig_d(m,k)=(r,s)$ ，过程为：
1.  选取随机数 $k$ ，满足 $1 < k < p − 1$ ，并且 $gcd(k,p-1)=1$。
2. 计算  $r\,\equiv \,g^{k}{\pmod {p}}$
3. 利用扩展欧几里得公式 $m \, \equiv \, x r + s k \pmod{p-1}$，计算 $s\,\equiv \,(m-xr)k^{-1}{\pmod {p-1}}$ 。
4. 如果 $s=0$ ， 重新计算。

对 $m$ 的签名结果为 $(r,s)$ 。

### 验证
B拿到消息和消息的签名结果验证阶段：
如果 $g^m\, \equiv \, y^{r}r^{s} {\pmod {p}}$ ，那么验证成功，否则验证失败。

由于 $m \, \equiv \, x r + s k \pmod{p-1}$
$$
\begin{align}
g^{m} & \equiv g^{xr} g^{ks} \\
& \equiv (g^{x})^r (g^{k})^s \\
& \equiv (y)^r (r)^s \pmod p.\\
\end{align}$$



## DSA -Digital Signature Algorithm
Digital Signature Algorithm (DSA)是Schnorr和ElGamal签名算法的变种，被美国NIST作为DSS(DigitalSignature Standard)。 专门用于签名和验签。
DSA是基于整数有限域离散对数难题的，其安全性与RSA相比差不多。

### 密钥生成 

密钥生成有两个阶段。  
第一阶段，是公开的参数信息。
1. 选择一个合适的哈希函数 $H$，目前一般选择 SHA1，当前也可以选择强度更高的哈希函数 如 SHA2。
2. 选择密钥的长度 $L$ 和 $N$，这两个值决定了签名的安全程度。在最初的 DSS（Digital Signature Standard ）中建议 $L$ 必须为 64 的倍数，并且 $512 ≤ L ≤ 1024$，当然，也可以更大。 $N$ 必须不大于哈希函数 $H$ 输出的长度。FIPS 186-3 给出了一些建议的 L 和 N 的取值例子：(1024, 160)， (2048, 224)， (2048, 256)，以及 (3,072, 256)。
3. 选择 $N$ 比特的素数 $q$ , $N$ 长度小于或等于哈希函数输出长度。
4. 选择 $L$ 比特的素数 $p$，使得 $p-1$ 是 $q$ 的倍数。
5. 选择 $g$ ，其模$p$ 的乘阶为 $q$ ，意味着 $q$ 是满足  $g^q=1\pmod p$ 最小的正整数，即 $ord_p(g)=p$。 即 $g$ 在模 $p$ 的意义下，其指数次幂可以生成具有 $q$ 个元素的子群。这里，我们可以通过计算 $g = h^ {\frac {p − 1} {q} }\pmod {p}$ 来得到 $g$，其中 $1 < h < p − 1$ 。 大部分的 $h$ 选择会导致可使用的 $g$ ，通常 $h=2$ 。

$(p, g, q)$会在不同的系统间公开。
第二阶段，计算公钥和私钥。
选择私钥 $x$，使其满足 $0 < x < q$ ，计算 $y ≡ g^x mod p$ 。  
公钥为 $(p, q, g, y)$ 。

### 签名

1. 选择随机整数数 k 作为*临时密钥*， $ 0 < k < q $。 
2. 计算 $r ≡ (g^k \pmod {p} ) \pmod {q} $
3. 计算 $s ≡ (H(m) + x r ) k^{−1} \pmod {q}$ 。

签名结果为 $(r,s)$。**需要注意的是，这里与 Elgamal 很重要的不同是这里使用了哈希函数对消息进行了哈希处理**。

可以利用扩展欧几里得算法计算 模逆 $k^{−1} \pmod {q}$ ，或者使用费马小定理。

由于签名者 既不知道 私钥 $x$ ，又不知道随机数 $k$ ，在验证 $s ≡ (H(m) + x r ) k^{−1} \pmod {q}$ 时，需要将其转换成 $ k ≡ (H(m) + x r ) s^{−1} \pmod {q} $ 。  
两边作 $g$ 的幂指数，得到 $ g^k ≡ g^{H(m)k^{-1}}y^{rs^{-1}} \pmod {p}$ 。 所以，验签者可以计算等式右边，等式左边是 $r$ ，那么可以判断等式是否成立。

### 验证

1. 先判断 $ 0 < r <q $ 或者 $ 0 < s < q $ 是否满足条件，如果不满足，则不验签。
2. 计算 $w=s^{-1}{\bmod {\,}}q $
3. 计算 $ u_{1}=H\left(m\right)\cdot w\,{\bmod {\,}}q $
4. 计算 $ u_{2}=r\cdot w\,{\bmod {\,}}q $
5. 计算 $ v=\left(g^{u\_{1}}y^{u\_{2}}{\bmod {\,}}p\right){\bmod {\,}}q $

如果 $ v = r $ ， 那么签名有效。

### 正确性证明



签名者计算 $$ s ≡ (H(m) + x r ) k^{−1} \pmod {q}$$ ，
可得
$$
\begin{align}
k & \equiv H(m)s^{-1}+xrs^{-1}\\
  & \equiv H(m)w + xrw \pmod{q}
\end{align}
$$

费马小定理 $ g^q ≡ h^{p − 1} ≡ 1 \pmod {p} $ ，且 $ g >1 $ , $q$ 是质数， 因此 $g$ 有 $q \pmod{p}$ 阶。  
$$
\begin{align}
g^{k}&\equiv g^{H(m)w}g^{xrw}\\
	&\equiv g^{H(m)w}y^{rw}\\
	&\equiv g^{u\_{1}}y^{u\_{2}}{\pmod {p}}
\end{align}
$$

DSA的正确性可从下式得出：
$$
\begin{align}
r&=(g^{k}{\bmod {\,}}p){\bmod {\,}}q\\
&=(g^{u\_{1}}y^{u\_{2}}{\bmod {\,}}p){\bmod {\,}}q\\
&=v
\end{align}
$$

## ECDSA

随机数很重要！
> The ECDSA digital signature has a drawback compared to RSA in that it requires a good source of entropy.   
Without proper randomness, the private key could be revealed.  
A flaw in the random number generator on Android allowed hackers to find the ECDSA private key used to protect the bitcoin wallets of several people in early 2013.   
Sony's Playstation implementation of ECDSA had a similar vulnerability.   
A good source of random numbers is needed on the machine making the signatures. Dual_EC_DRBG is not recommended.

from [A (Relatively Easy To Understand) Primer on Elliptic Curve Cryptography](https://blog.cloudflare.com/a-relatively-easy-to-understand-primer-on-elliptic-curve-cryptography/)  

# Elliptic Curve Cryptography (ECC)  

[Elliptic Curve Cryptography (ECC)](https://cryptobook.nakov.com/asymmetric-key-ciphers/elliptic-curve-cryptography-ecc)  

基于椭圆曲线的密码算法包括：

1. ECC签名算法，比如ECDSA和EdDSA  

2. ECC加密算法，比如ECIES integrated encryption scheme and EEECC (EC-based ElGamal).

3. ECC密钥协商，比如ECDH、X25519。    


ECC算法可以选取不同的椭圆曲线，根据曲线的不同，安全等级、密钥长度、计算速度也不同。  
比如 `secp256k1` 和 `Curve25519` 。   
一般，ECC私钥长度为 256 bits，但是也分曲线。比如 192-bit (curve secp192r1), 233-bit (curve sect233k1), 224-bit (curve secp224k1), 256-bit (curves secp256k1 and Curve25519), 283-bit (curve sect283k1), 384-bit (curves p384 and secp384r1), 409-bit (curve sect409r1), 414-bit (curve Curve41417), 448-bit (curve Curve448-Goldilocks), 511-bit (curve M-511), 521-bit (curve P-521), 571-bit (curve sect571k1)。  

椭圆曲线的函数表示：  $y^2 = x^3 + a*x + b$  
例如，对于 secp256k1，$y^2 = x^3 + 7$，a=0,b=7。  


椭圆曲线上的操作包括点加（ EC point addition），点乘（EC point multiplication）。  

椭圆曲线的几点要素：  
+ Еlliptic curve (EC) over finite field $𝔽_p$
+ $G$ == generator point (fixed constant, a base point on the EC)
+ $k$ == private key (integer)
+ $P$ == public key (point)  

私钥是 一个整数，公钥是一个椭圆曲线上的点(EC point)，$P = k * G$。  

有限域上的椭圆曲线的点构成了循环群，因此定义曲线的阶数 `order` 为EC全部的点。  
定义无穷远点为 任一点乘以 0 得到的点。  
但是有些曲线会生成若干$h$循环子群，每个子群的阶数为$r$，因此整个群的阶数为 $n=h*r$。  


+ Curve25519  

$y^2 = x^3 + 486662x^2 + x$


# 国产密码算法

国产密码算法（国密算法）是指国家密码局认定的`国产商用密码算法`，在金融领域目前主要使用公开的SM2、SM3、SM4三类算法，分别是非对称算法、哈希算法和对称算法。 其中`SM`代表“商密”，即用于商用的、不涉及国家秘密的密码技术。

## SM2椭圆曲线公钥密码算

SM2椭圆曲线公钥密码算法是我国自主设计的公钥密码算法，包括 **SM2-1椭圆曲线数字签名算法**，**SM2-2椭圆曲线密钥交换协议**，**SM2-3椭圆曲线公钥加密算法**，分别用于实现数字签名密钥协商和数据加密等功能。SM2算法与RSA算法不同的是，SM2算法是基于椭圆曲线上点群离散对数难题，相对于RSA算法，256位的SM2密码强度已经比2048位的RSA密码强度要高。  

SM2 $F_p-256$ 椭圆曲线选取：  
$$y^2 = x^3 + ax + b$$
曲线参数：  
```
p=FFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF 00000000 FFFFFFFF FFFFFFFF
a=FFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF 00000000 FFFFFFFF FFFFFFFC
b=28E9FA9E 9D9F5E34 4D5A9E4B CF6509A7 F39789F5 15AB8F92 DDBCBD41 4D940E93
n=FFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF 7203DF6B 21C6052B 53BBF409 39D54123
Gx=32C4AE2C 1F198119 5F990446 6A39C994 8FE30BBF F2660BE1 715A4589 334C74C7
Gy=BC3736A2 F4F6779C 59BDCEE3 6B692153 D0A9877C C62A4740 02DF32E5 2139F0A0
```

[OpenSSL SM2代码](https://github.com/openssl/openssl/tree/master/crypto/sm2)  

### SM2-椭圆曲线数字签名算法

### SM2-椭圆曲线密钥交换协议

### SM2-椭圆曲线公钥加密算法

## SM3杂凑算法
SM3杂凑算法是我国自主设计的密码杂凑算法，适用于商用密码应用中的数字签名和验证消息认证码的生成与验证以及随机数的生成，可满足多种密码应用的安全需求。为了保证杂凑算法的安全性，其产生的杂凑值的长度不应太短，例如MD5输出128比特杂凑值，输出长度太短，影响其安全性SHA-1算法的输出长度为160比特，SM3算法的消息分组长度是512比特，输出长度为256比特，因此SM3算法的安全性要高于MD5算法和SHA-1算法。

整个算法的执行过程可以概括成四个步骤：消息填充、迭代压缩、输出结果。  

### 消息填充

由于分组长度为512比特，把数据长度填充至512位的倍数。  
首先在数据末尾填充一个比特`1`，而后在后面填充k个0，k满足(n+1+k) mod 512 = 448的最小正整数。这是为了保证最后填充的64位存储真实有效的原始数据长度，

这里存在着这种情况，如果最后一分组的有效数据长度超过448，那么需要再创建一个512的分组。

### 迭代压缩  

将消息按照512分组进行迭代压缩。 每轮的迭代过程为：  
$$V^{(i+1)} = CF(V^{(i)}, B^{(i)})$$

其中 $CF$ 是压缩函数，$V^{(0)}$ 为256比特初始值$IV$，$B^{(i)}$ 为填充后的消息分组，迭代压缩的结果为$V^{(n)}$。  

对于压缩函数计算过程，SM3没有直接使用原始消息，而是SM3使用 **消息扩展** 得到的消息字进行运算。  

压缩函数的初值IV被放在A、B、C、D、E、F、G、H八个32位变量中，需要进行64轮迭代，每轮的输出再作为下一轮压缩函数时的初值。

将得到的A、B、C、D、E、F、G、H八个变量拼接输出，就是SM3算法的输出。

kernel中sm3实现在 *crypto/sm3_generic.c*

## SM4分组密码算法

SM4分组密码算法是我国自主设计的分组对称密码算法，用于实现数据的加密/解密运算，以保证数据和信息的机密性。
要保证一个对称密码算法的安全性的基本条件是其具备足够的密钥长度，SM4算法与AES算法具有相同的密钥长度/分组长度128比特，因此在安全性上高于3DES算法。

SM4加密算法与密钥扩展算法都采用 32 轮非线性迭代结构。数据解密和数据加密的算法结构相同，
只是轮密钥的使用顺序相反，解密轮密钥是加密轮密钥的逆序。  

该分组密码算法实现相对简单，也同样分为密钥扩展算法、加密和解密算法。
字节按照大端序处理。 

+ 每一轮的轮函数 $F$：  

$$𝐹(𝑋_0, 𝑋_2, 𝑋_3, 𝑋_4, 𝑟𝑘) = 𝑋_0\bigoplus𝑇(𝑋_2 \bigoplus 𝑋_3 \bigoplus 𝑋_4 \bigoplus 𝑟𝑘)$$
其中，输入 $(𝑋_0, 𝑋_2, 𝑋_3, 𝑋_4)$ 为4个32bit的字， $rk$ 为当前32bit的轮密钥，$T$ 为可逆变换，由字节替换(SBox)和循环向左移位异或两部组成。 

$$T(x) = L(\tau(x))$$
其中， $\tau(x)$ 为非线性变换，将 32bit 的 $x$ 的每个字节使用 SBox 替换。
$L(x)= x \bigoplus (x<<<2) \bigoplus (x<<<10) \bigoplus (x<<<18) \bigoplus (x<<<24)$ 

循环左移可参见宏  
```c
#define ROTL(x, shift)	(((x)<<(shift&(32-1))) | ((x)>>(32-(shift&(32-1)))))
```

### 密钥扩展算法  

密钥扩展算法需要辅助参数 $FK$ 和 $CK$ ，

加密密钥为 $MK=(MK_0, MK_1, MK_2, MK_3)$，
则首先生成首轮轮密钥： $(K_0, K_1, K_2, K_3)=(MK_0 \bigoplus FK_0, MK_1 \bigoplus FK_1, MK_2 \bigoplus FK_2, MK_3 \bigoplus FK_3)$。  
则32个轮密钥都由前3个产生，即 
$$rk_i=K_{i+4}=K_i \bigoplus T'(K_{i+1} \bigoplus K_{i+2} \bigoplus K_{i+3} \bigoplus CK_{i})$$

$T'()$ 是将合成转置中的 $L$ 替换成 $L'$，即 $L'(x) = x \bigoplus (x<<<13) \bigoplus (x<<<23)$ 。

### 加密算法  

进行32次迭代计算：  $$X_{i+4}=F(X_i,X_{i+1},X_{i+2},X_{i+3},𝑟𝑘_{i})$$  

将最后结果反序： 

$$(Y_0,Y_1,Y_2,Y_3) = R(X_{32},X_{33},X_{34},X_{35}) = (X_{35},X_{34},X_{33},X_{32})$$


### 解密算法  

本算法的解密变换与加密变换结构相同，不同的仅是轮密钥的使用顺序。解密时，使用
轮密钥序 $(rk_{31},rk_{30},...,rk_{0})$。


kernel中sm3实现在 *crypto/sm4_generic.c*，用户态实现见[sm4实现](https://github.com/fengkx/sm4)、 [SM4实现](https://github.com/windard/sm4)。  

## 祖冲之序列密码算法




# 参考文献
1. [安全体系（一）—— DES算法详解](http://www.cnblogs.com/songwenlong/p/5944139.html)
2. [迪菲-赫尔曼密钥交换](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)
3. [分组密码工作模式](https://zh.wikipedia.org/wiki/%E5%88%86%E7%BB%84%E5%AF%86%E7%A0%81%E5%B7%A5%E4%BD%9C%E6%A8%A1%E5%BC%8F)
4. [密码算法详解——AES](http://www.cnblogs.com/luop/p/4334160.html)
5. [密码标准应用指南](http://www.gmbz.org.cn/upload/2018-03-24/1521879142922000396.pdf)
6. [学习密码学的一套教程 CRYPTO101](https://www.crypto101.io/)


