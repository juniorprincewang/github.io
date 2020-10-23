---
title: 数字证书和PKI
tags:
  - certificate
  - PKI
categories:
  - - security
    - crypto
date: 2020-10-23 15:04:06
---

汇总整理数字签名和证书的知识点。
<!-- more -->


# 数字签名

基于公钥密码学的数字签名能够用于通信的消息鉴别、数据完整性和非否认服务。
基于公钥密码算法，Alice可以使用私钥对消息进行运算，Bob使用Alice公钥对消息进行验证。
因此，数字签名由签名和验证两个过程组成。  

签名的信息通过在其后面附加信息的摘要来签名。 摘要是通过单向哈希函数生成的，而加密是使用签名者的私钥计算的。  


验证签名过程分三步：  
1). 计算原始消息的哈希值；  
2). 使用签名者的公钥解密签名
3). 将两个结果比较看是否一致。  

<!-- ![验签过程](/img/digital-certificate/Digital_Signature_Verification.png) -->
![签名/验签过程](/img/digital-certificate/Digital_Signature_diagram.svg)
[Public Key Cryptography](https://kulkarniamit.github.io/whatwhyhow/security/public-key-cryptography.html)  

# ASN.1  

[ASN.1](https://www.itu.int/en/ITU-T/asn1/Pages/introduction.aspx)（Abstract Syntax Notation One） 一种数据定义语言，提供了一套数据类型表示和定义的方法，并且提供相应的编码和解码的规则，使得不同的系统之间可以采用统一的一套方式进行数据的通信。  

ASN.1是描述在网络上传输信息格式的标准方法。它有两部分：一部分描述信息内数据，数据类型及序列格式；另一部分描述如何将各部分组成消息。
即ASN.1 定义了一些简单类型，然后通过组合简单类型可以构造出复杂类型。  

简单类型包括：  integers (INTEGER), booleans (BOOLEAN), character strings (IA5String, UniversalString...), bit strings (BIT STRING) 等。  
构造类型包括： structures (SEQUENCE), lists (SEQUENCE OF), choice between types (CHOICE) 等。  

最基本的表达式如：　`Name ::= type` ，表示为定义某个名称为Name的元素，它的类型为type。  
再比如：  
```
PublicKey::= SEQUENCE {
           KeyType       BOOLEAN(0),
           Modulus        INTEGER,
           PubExponent INTEGER
        }
```

使用ASN.1描述的数据结构，需要将数据结构编码成二进制文件。
DER（Distinguished Encoding Rules，可辨别编码规则)是 ASN.1 语法中的一种编码方式。  
ASN.1 编码还包括 Basic Encoding Rules (BER)、 Canonical Encoding Rules (CER)、、 XML Encoding Rules (XER)、 Packed Encoding Rules (PER)、 Generic String Encoding Rules (GSER)。  

DER编码采用`TLV`三元组的形式，即Type-Length-Value组织形式。
例如：  02 01 05 的解码含义：  

>02 -- tag indicating INTEGER 
>01 -- length in octets 
>05 -- value 

<https://lapo.it/asn1js/>是证书解码器，可以解码  ASN.1 DER 结构的PEM文件。  

# 公钥证书（Public-key certificate，PKC） 

公钥证书由三部分组成：  
1. 证书内容 tbsCertificate，注：tbs=ToBeSigned
2. 签名算法 signatureAlgotithm
3. 签名结果 signatureValue。  

[公钥证书的ASN.1 描述](https://tools.ietf.org/html/rfc5280)为：  

```
Certificate  ::=  SEQUENCE  {
        tbsCertificate       TBSCertificate,
        signatureAlgorithm   AlgorithmIdentifier,
        signatureValue       BIT STRING  }
```

其中 证书内容：  
```
TBSCertificate ::= SEQUENCE {
    version [0] Version DEFAULT v1,
    serialNumber CertificateSerialNumber,
    signature AlgorithmIdentifier{{SupportedAlgorithms}},
    issuer Name,
    validity Validity,
    subject Name,
    subjectPublicKeyInfo SubjectPublicKeyInfo,
    issuerUniqueIdentifier [1] IMPLICIT UniqueIdentifier OPTIONAL,
    ...,
    [[2: -- if present, version shall be v2 or v3
    subjectUniqueIdentifier [2] IMPLICIT UniqueIdentifier OPTIONAL]],
    [[3: -- if present, version shall be v2 or v3
    extensions [3] Extensions OPTIONAL]]
    -- If present, version shall be v3]]
}
```


签名类型 ASN.1 描述为  
```
SIGNED{ToBeSigned} ::= SEQUENCE {
    toBeSigned ToBeSigned,
    COMPONENTS OF SIGNATURE{ToBeSigned},
... }
```

签名算法为 AlgorithmIdentifier，签名结果为 BIT STRING 类型。   

```
ENCRYPTED-HASH{ToBeSigned} ::= BIT STRING (CONSTRAINED BY {
    -- shall be the result of applying a hashing procedure to the DER-encoded (see 6.2)
    -- octets of a value of -- ToBeSigned -- and then applying an encipherment procedure
    -- to those octets -- } )

SIGNATURE{ToBeSigned} ::= SEQUENCE {
    algorithmIdentifier AlgorithmIdentifier{{SupportedAlgorithms}},
    encrypted ENCRYPTED-HASH{ToBeSigned},
    ... }
```

更具体的证书内容包括：  

1. 版本号  

通常为 v3，用2表示。  
```
Version ::= INTEGER {v1(0), v2(1), v3(2)}
```

2. 序列号

序列号用来在当前CA签发的唯一一个标识证书。  

```
CertificateSerialNumber ::= INTEGER
```


3. 签名算法  

签名算法给出了CA签发证书使用到的数字签名算法。  
```
AlgorithmIdentifier{ALGORITHM:SupportedAlgorithms} ::= SEQUENCE {
    algorithm ALGORITHM.&id({SupportedAlgorithms}),
    parameters ALGORITHM.&Type({SupportedAlgorithms}{@algorithm}) OPTIONAL,
... }
```


4. 签发者（issuer）

签发者标识了签发证书的CA实体，类型为Name。
Name用DN（Distinguished Name，DN）表示，DN是由（Relative Distinguished Name，RDN）构成的序列。  
RDN用 **属性类型=属性值**的形式表示。  
比如 CN=Google，CN为Common Name的缩写。  

5. 证书主体 (subject)

证书主体标识了证书持有者，类型为Name。同issuer。    

6. 有效期 (valid)

```
Validity ::= SEQUENCE {
    notBefore Time,
    notAfter Time,
... }

Time ::= CHOICE {
    utcTime UTCTime,
    generalizedTime GeneralizedTime }
```

7. 主体公钥信息  

主体公钥信息给出了证书所绑定的加密算法和公钥。  

```
SubjectPublicKeyInfo ::= SEQUENCE {
    algorithm AlgorithmIdentifier{{SupportedAlgorithms}},
    subjectPublicKey BIT STRING,
... }
```

8. 签发者唯一标识符和主体唯一标识符  

一般不推荐使用这两个字段。  

证书除了上述基本内容，还可以包括一些扩展项。这里不再展开阐述，可以参见[certificate extensions](https://tools.ietf.org/html/rfc5280#section-4.2)。   


# PKI  

public-key infrastructure (PKI): 能够支持公钥管理的基础结构，该公钥能够支持身份鉴别、加密、完整性或不可否认服务。  

ITU-T X.509 标准规定PKI包括三种不同功能的实体：  
1. 证书认证中心（CA）  
CA具有自己的公私钥，负责为他人签发证书。
2. 证书持有者（certificate holder）  
证书持有者的身份信息和对应的公钥会出现在证书中。  
3. 依赖方  
使用他人证书来实现安全功能的通信实体称为依赖方。  

此外，PKI系统还包括其他提供辅助服务的组件，如 注册机构（registration authority，RA），密钥管理系统（key management system，KMS），OCSP（Online Certificate Status Protocol），CRL Issuer等。  

PKI系统为证书提供了证书生成、使用、撤销、更新、归档等管理。  

PKI体系中，CA成层级出现，除了作为信任锚的根CA外，还有其他的下级CA，每级CA拥有的证书都由上级CA签发。
而根CA证书是一种自签名的证书，无法通过PKI技术手段对其进行验证，只能通过带外方式获取。  

而对证书的验证过程涉及以下步骤：  
1. 首先审查持有者的证书是否有效，包括查看证书的有效日期并查看证书的撤销情况。
2. 获取该证书签发者的数字证书来验证该证书上的数字签名。拿到签发者证书后，同样需要检查证书的失效日期，并查看证书的撤销情况。
3. 根据证书签发路径，一致查找到根CA的签名，由于根CA证书是自签名证书，因此不需要其他证书来验证根证书上的签名了。
4. 用根CA的公钥来验证签发者CA证书的签名，并逐个验证证书的签名信息。
5. 注意查看证书撤销列表（Certificate Revocation List，CRL）的时候，撤销列表也有根CA和下级CA的签名，验证撤销列表证书的过程也如前所述。  


![证书路径](/img/digital-certificate/certificate_path.png)

[Verify SSL/TLS Certificate Signature](https://kulkarniamit.github.io/whatwhyhow/howto/verify-ssl-tls-certificate-signature.html)给出了验证stackoverflow.com服务器的证书的过程。  


