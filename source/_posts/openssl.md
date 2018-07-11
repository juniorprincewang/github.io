---
title: openssl
date: 2018-06-11 16:00:14
tags:
- openssl
---
介绍openssl 软件和库的一些知识点。
<!-- more -->


# hmac

openssl 的摘要和数字签名算法指令可以通过 `openssl dgst -` 命令查看。

```
max@MAX:~/GPU/code/Curve25519$ openssl dgst -
unknown option '-'
options are
-c              to output the digest with separating colons
-r              to output the digest in coreutils format
-d              to output debug info
-hex            output as hex dump 								//以16进制打印输出结果
-binary         output in binary form
-hmac arg       set the HMAC key to arg
-non-fips-allow allow use of non FIPS digest
-sign   file    sign digest using private key in file
-verify file    verify a signature using public key in file
-prverify file  verify a signature using private key in file
-keyform arg    key file format (PEM or ENGINE) 				//指定密钥文件格式，pem或者engine
-out filename   output to filename rather than stdout
-signature file signature to verify
-sigopt nm:v    signature parameter
-hmac key       create hashed MAC with key              		//指定hmac的密钥为key，可以加引号或不加引号
-mac algorithm  create MAC (not neccessarily HMAC)
-macopt nm:v    MAC algorithm parameters or key
-engine e       use engine e, possibly a hardware device.
-md4            to use the md4 message digest algorithm
-md5            to use the md5 message digest algorithm
-ripemd160      to use the ripemd160 message digest algorithm
-sha            to use the sha message digest algorithm
-sha1           to use the sha1 message digest algorithm
-sha224         to use the sha224 message digest algorithm
-sha256         to use the sha256 message digest algorithm
-sha384         to use the sha384 message digest algorithm
-sha512         to use the sha512 message digest algorithm
-whirlpool      to use the whirlpool message digest algorithm

```

## HMAC

HMAC是密钥相关的哈希运算消息认证码，HMAC运算利用哈希算法，以一个密钥和一个消息为输入，生成一个消息摘要作为输出。


```
max@MAX:~/GPU/code/Curve25519$ echo -n "hello world" | openssl dgst -sha256 -hmac 123456
(stdin)= 83b3eb2788457b46a2f17aaa048f795af0d9dabb8e5924dd2fc0ea682d929fe5
```

[这里 `echo -n` 的目的是将输入的字符串去掉自动换行。](https://stackoverflow.com/questions/7285059/hmac-sha1-in-bash)

### 参考代码：

```
#include <openssl/hmac.h>  
#include <string.h>  
  
  
int HmacEncode(const char * algo,  
                const char * key, unsigned int key_length,  
                const char * input, unsigned int input_length,  
                unsigned char * &output, unsigned int &output_length) {  
        const EVP_MD * engine =  EVP_sha256();  
        output = (unsigned char*)malloc(EVP_MAX_MD_SIZE);  
  
        HMAC_CTX ctx;  
        HMAC_CTX_init(&ctx);  
        HMAC_Init_ex(&ctx, key, strlen(key), engine, NULL);  
        HMAC_Update(&ctx, (unsigned char*)input, strlen(input));        // input is OK; &input is WRONG !!!  
  
        HMAC_Final(&ctx, output, &output_length);  
        HMAC_CTX_cleanup(&ctx);  
  
        return 0;  
}  
```
还有一种非常简单直接的接口：
```
HMAC(EVP_sha1(), key, strlen(key), (unsigned char*)data, strlen(data);
```

# 参考
[1] [OpenSSL SHA256 Hashing Example in C++](http://www.askyb.com/cpp/openssl-sha256-hashing-example-in-cpp/)
[2] [Using openssl to generate HMAC using a binary key](http://nwsmith.blogspot.com/2012/07/using-openssl-to-generate-hmac-using.html)
[3] [openssl 摘要和签名验证指令dgst使用详解](https://www.cnblogs.com/gordon0918/p/5382541.html)
[4] [关于openssl加解密文件的几个API](https://blog.csdn.net/jiangheng0535/article/details/41719259)
[5] [用OpenSSL 做HMAC（C++）](https://blog.csdn.net/yasi_xi/article/details/9066003)
[6] [OpenSSL HMAC Hasing Example in C++](http://www.askyb.com/cpp/openssl-hmac-hasing-example-in-cpp/)