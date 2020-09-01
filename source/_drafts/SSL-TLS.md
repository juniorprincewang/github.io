---
title: SSL/TLS
date: 2019-08-15 15:49:26
tags:
- TLS
- SSL
categories:
- [security,crypto]
---
汇总整理SSL/TLS的知识点。
<!-- more -->


暂时先列一些参考文献。  

+ [Difference between SSL & TLS](https://stackoverflow.com/q/3690734)
+ [How does SSL/TLS work?](https://security.stackexchange.com/q/20803)
+ [How does the symmetric key get exchanged in SSL/TLS handshake?](https://security.stackexchange.com/q/130938)  
+ 看大牛的博客 https://www.davidwong.fr/tls13/
+ 还有非常nice简单直观的说明：https://tls13.ulfheim.net/


# How Does the SSL Certificate Create a Secure Connection?

1. **Browser** connects to a web server (website) secured with SSL (https). Browser requests that the server identify itself.
2. **Server** sends a copy of its SSL Certificate, including the server’s public key.
3. **Browser** checks the certificate root against a list of trusted CAs and that the certificate is unexpired, unrevoked, and that its common name is valid for the website that it is connecting to. If the browser trusts the certificate, it creates, encrypts, and sends back a symmetric session key using the server’s public key.
4. **Server** decrypts the symmetric session key using its private key and sends back an acknowledgement encrypted with the session key to start the encrypted session.
5. **Server** and **Browser** now encrypt all transmitted data with the session key.

由于公钥密码算法占用计算资源大和时间长，因此在协议中只在创建对称密钥时使用一次。  
