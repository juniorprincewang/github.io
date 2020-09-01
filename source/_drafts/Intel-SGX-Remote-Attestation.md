---
title: Intel SGX Remote Attestation
tags:
- sgx
categories:
- [security,crypto]
---

整理SGX Remote Attestation 的流程。

<!--more-->




# 术语  


+ QE： Quoting Enclave 
+ ISV : Independent Software Vendor.独立软件厂商。  
+ SP ： ISV Service provider   
+ PSE: Platform Services Enclave  
+ AE: 平台提供的enclave，Architectural enclaves. Enclaves that are part of the Intel(R) Software Guard Extensions framework. They include the quoting enclave (QE), provisioning enclave (PvE), launch enclave (LE)  
+ ra:  remote attestation  
+ IAS: Intel® Attestation Service (IAS)  
+ SPID: service provider ID (SPID)  
+ EPID: Intel® Enhanced Privacy ID (Intel® EPID)  

    Intel EPID is a group signature scheme, which allows platforms to cryptographically sign
    objects while at the same time preserving the signer’s privacy.

# Remote Attestation  

远程认证的内容包含了：

    its identity
    That it has not been tampered with
    That it is running on a genuine platform with Intel SGX enabled
    That it is running at the latest security level, also referred to as the Trusted Computing Base (TCB) level


在这个过程中，
1. 客户端的软硬件平台信息，以及相关Enclave的指纹信息等将会首先发送到开发者的服务器(Service Provider)。  
2. 然后由开发者的服务器转发给SGX的远程认证服务器(Intel Attestation Service)。  
3. SGX远程认证服务器将会对收到的信息进行合法性验证，并将验证结果返回给开发者的服务器。  
4. 此时开发者的服务器便可得知发起验证的客户端是否可信，并采取对应的下一步行动。  



## 客户端 - 服务器协议，DHKE  
远程认证（ra）利用经过修改的 Sigma 协议来协助client与server之间的 Diffie-Hellman 密钥交换 (DHKE)。  
服务提供商使用从交换中获得的共享密钥（shared key）来加密提供给客户端的机密。
客户端安全区能够获得相同的密钥，并用它解密机密。

## 过程  
整个过程类似于HTTPS的握手协议。  
客户端发起请求服务，而服务端向客户端发出认证挑战（challenge）。

### Msg0 (client to server)  

客户端为了响应挑战，需要构建远程认证流程的初始消息。  

1. 创建enclave
2. 执行 ECALL 进入enclave安全区  
3. 在安全区内：  
    1. 调用 sgx_ra_init()  
    2. 将结果和 DHKE 上下文参数返回不可信应用  
4. 调用 sgx_get_extended_epid_group_id()  


`sgx_ra_init()` 函数将服务提供商SP的**公钥**视为参数，并使用公钥在远程认证过程中向 DHKE 返回不透明的上下文（上下文格式为 *sgx_ra_context_t*）。  

公钥的格式定义为*sgx_tcrypto.h*中的*sgx_ec256_public_t*，EC 密钥被表示为 x 坐标和 y 坐标，小端表示：   
```
typedef struct _sgx_ec256_public_t
{
    uint8_t gx[SGX_ECP256_KEY_SIZE];
    uint8_t gy[SGX_ECP256_KEY_SIZE];
} sgx_ec256_public_t;
```

SP的公钥应被硬编码至安全区。  

+ PSE  

如果安全区需要访问平台服务，平台服务安全区 (PSE) 必须包含在认证序列中。
PSE 是英特尔 SGX 软件包中的架构安全区，为可信时间和单向性计数器（a monotonic counter）提供服务。
它们可用于随机数生成过程中的回放保护，以及安全地计算机密有效的时长。  

为了使用 PSE，该流程变为：  

1. 创建enclave  
2. 执行 ECALL 进入安全区  
3. 在安全区内，执行以下步骤（按顺序）：
    1. sgx_create_pse_session()  
    2. sgx_ra_init()  
    3. sgx_close_pse_session()  
    4. 将结果和 DHKE 上下文参数返回不可信应用  
4. 调用 sgx_get_extended_epid_group_id()  

一旦 `sgx_ra_init()` 返回成功结果，客户端接下来调用 `sgx_get_extended_epid_group_id()`，以检索英特尔® Enhanced Privacy ID（Intel EPID）的扩展组 ID (GID)。
英特尔 EPID 是一个用于身份验证的匿名签名方案。

客户端需要将Intel EPID 发送至服务提供商，这将作为 msg0 的参数。
预发送的msg0 的格式与来自服务器的详细响应信息可以自己定义。
但是，服务提供商必须验证它接收的 EPID是否受到支持；如果不支持的话，SP中止认证过程。
英特尔认证服务仅支持EPID值为0。  

### Msg1 (client to server)  

客户端调用 `sgx_ra_get_msg1()`函数，以构建包含面向 DHKE 的客户端公钥的 *msg1* 对象并将其发送至服务提供商SP。  
该方法还需要额外的参数，包括从之前步骤中获取的 DHKE 上下文（*sgx_ra_context_t*）和计算客户端 DHKE 密钥的 `sgx_ra_get_ga()` 桩函数的指针。
此函数需要引入 *sgx_tkey_exchange* 库的 *sgx_tkey_exchange.edl*。  

msg1 的数据结构为：  
```c
typedef struct _ra_msg1_t
{
    sgx_ec256_public_t   g_a;  //g_a_x 32 bytes, g_a_y 32 bytes
    sgx_epid_group_id_t  gid;  // 4 bytes
} sgx_ra_msg1_t;
```

### Msg2 (server to client)  


SP在接收来自client的 msg1 后，检查请求中的值，生成自己的 DHKE 参数，并向 IAS 发送一个查询，来检索client发送的Intel EPID GID 的吊销列表 (SigRL)。  

为了处理 msg1 与生成 msg2，服务提供商执行以下步骤：  
1. 使用 P-256 曲线生成一个随机 EC 密钥。该密钥将成为 $Gb$。
2. 从 $Ga$ 和 $Gb$ 中导出 **密钥派生密钥KDK**：  
    1. 使用客户端的公钥 $Ga$ 和服务提供商SP的私钥（从第一步中获取）$Gb$ 计算共享机密。运算结果将成为 $Gab$ 的 $x$ 坐标，表示为 $Gab_x$。  
    2. 将 $Gab_x$ 的字节顺序转换成小端顺序。  
    3. 使用0x00字节的块作为密钥，以 $Gab_x$ 的小端顺序执行 AES-128 CMAC。。
    4. 2.3 的结果为 KDK。
3. 通过对字节序列：`0x01 || SMK || 0x00 || 0x80 || 0x00` 执行 AES-128 CMAC，从 KDK 中导出 SMK，将 KDK 用作密钥。请注意，`||`表示字符串连接联，`SMK`就是字符串。


msg2 的数据格式定义在 *sgx_key_exchange.h*，

```c
typedef struct _ra_msg2_t
{
    sgx_ec256_public_t       g_b;
    sgx_spid_t               spid;
    uint16_t                 quote_type;
    uint16_t                 kdf_id;
    sgx_ec256_signature_t    sign_gb_ga;
    sgx_mac_t                mac;
    uint32_t                 sig_rl_size;
    uint8_t                  sig_rl[];
} sgx_ra_msg2_t;
```

需要注意的还有几点：  

1. 确定应向客户端请求的quote type（0x0 表示不可关联 unlinkable，0x1 表示可关联linkable）。请注意，SPID 必须与正确的quote type相关联。  
2. spid是服务提供商的ID信息。16 bytes  
2. 设置 `KDF_ID`，通常是 0x1。  
3. 使用SP的私钥计算 `Gbx || Gby || Gax || Gay` 的ECDSA签名。  
4. 使用SMK为密钥计算 `Gb || SPID || Quote_Type || KDF_ID || SigSP` 的AES-128 CMAC。  
5. 查询 IAS，以获取面向客户端的Intel EPID GID 的 SigRL。


该SP的公钥和 SigRL 信息被打包至 msg2 并发送到客户端，以响应 msg1 请求。

### Msg3 (client to server)  

在客户端，接收 msg2 后，调用 `sgx_ra_proc_msg2()`函数，以生成 msg3。该函数执行以下任务：  

1. 验证服务提供商签名。  
2. 检查 SigRL。  
3. 返回 msg3，后者包含用于验证特定enclave的quote。  


`sgx_ra_proc_msg2()`还需要的两个参数为 *sgx_ra_proc_msg2_trusted* 和 *sgx_ra_get_msg3_trusted*。
这些是 edger8r 工具自动生成的函数指针，必须在enclave工程内做以下操作：

1. 连接可信服务的库（Linux 上的 libsgx_tservice.a 和 Windows 上的 sgx_tservice.lib）。  
2. 在 EDL 文件 trusted 部分引用include以下内容：  

```
include "sgx_tkey_exchange.h"
from "sgx_tkey_exchange.edl" import *;
```

`sgx_ra_get_msg2_trusted()`执行的部分操作包括获取 enclave quote。   quote包括使用平台的EPID密钥签名的当前运行enclave的哈希（或称为度量）。   只有英特尔认证服务（IAS）可以验证此签名。   quote还包含有关平台上平台服务区域（PSE）的信息，IAS也将对其进行验证。  


msg3定义在 *sgx_key_exchange.h* 中。  
```c
typedef struct _ra_msg3_t
{
    sgx_mac_t                mac
    sgx_ec256_public_t       g_a;
    sgx_ps_sec_prop_desc_t   ps_sec_prop;
    uint8_t                  quote[];
} sgx_ra_msg3_t;

```

quote的数据结构为：  
```c
typedef struct _quote_t
{
    uint16_t            version;        /* 0   */
    uint16_t            sign_type;      /* 2   */
    sgx_epid_group_id_t epid_group_id;  /* 4   */
    sgx_isv_svn_t       qe_svn;         /* 8   */
    sgx_isv_svn_t       pce_svn;        /* 10  */
    uint32_t            xeid;           /* 12  */
    sgx_basename_t      basename;       /* 16  */
    sgx_report_body_t   report_body;    /* 48  */
    uint32_t            signature_len;  /* 432 */
    uint8_t             signature[];    /* 436 */
} sgx_quote_t;
```

### Msg4 (server to client)  

从客户端接收 msg3 后，SP 必须执行以下操作：  

1. 验证 msg3 中的 `Ga` 是否是 msg1 中的 `Ga`。  
2. 验证使用SMK做密钥的CMAC(M)。  
3. 验证 report 数据的前32字节是否匹配 (Ga || Gb || VK)的SHA-256结果。VK由 `0x01 || "VK" || 0x00 || 0x80 || 0x00` 使用`KDK`为密钥做AES-128 CMAC计算而来。  
4. 验证客户端提供的认证数据。  
    1. 从 msg3 中提取quote。  
    2. 将quote提交至 IAS，调用 API 函数以验证认证数据。  
    3. 验证report响应接收的签名证书。  
    4. 使用签名证书验证report签名。  
5. 如果在第 3 步成功验证了quote，执行以下操作：  
    1. 提取enclave的认证状态（如果提供）和 PSE。  
    2. 检查enclave标识（MRSIGNER）、安全版本和产品 ID。  
    3. 决定是否信任enclave和 PSE（如果提供）。
6. 派生会话密钥 `SK` 和 `MK`。 client可以使用`sgx_ra_get_keys()`获取会话密钥，而server必须计算 `MK: 0x01 || "MK" || 0x00 || 0x80 || 0x00` 和 `SK: 0x01 || "SK" || 0x00 || 0x80 || 0x00` 的以 `KDK` 为密钥的AES-128 CMAC值。  
7. 生成 msg4 并将其发送至客户端。

验证认证数据（the attestation evidence）要求服务提供商向 IAS 提交quote并获得认证报告。  
该报告由 IAS Report Signing私钥签名，服务提供商必须使用 IAS Report Signing公钥验证该签名。

msg4 的格式由服务提供商决定。它必须至少包含：  
1. 安全区是否可信。
2. 如果提交了 PSE 清单，PSE 清单是否可信。  


至此，描述了一个较为完整的远程认证流程。  


## Remote Attestation Sample

SDK包括了密钥交换库KE（Key Exchange）。  

+ RemoteAttestation  

上面已经讲完了。  
+ Remote Key Exchange (KE) Libraries  

不可信应用使用 *libsgx_ukey_exchange.a* 、 *sgx_ukey_exchange.h* 。   
可信应用 需要在edl文件中引用 *sgx_tkey_exchange.edl* 中的ecall函数。 

+ Remote Attestation and Protected Session Establishment  

+ Remote Attestation with a Custom Key Derivation Function (KDF)  

当 `sgx_ra_init` 用于生成远程认证的context时，在`sgx_ra_get_keys`中使用了KDF。  

# 参考文献  

+ [Code Sample: Intel® Software Guard Extensions Remote Attestation End-to-End Example](https://software.intel.com/en-us/articles/code-sample-intel-software-guard-extensions-remote-attestation-end-to-end-example)  
这是远程认证端对端的代码以及解释。  
+ [Example of Linux SGX remote attestation](https://github.com/svartkanin/linux-sgx-remoteattestation)   
+ [SGX 101- Attestation](https://sgx101.gitbook.io/sgx101/sgx-bootstrap/attestation)