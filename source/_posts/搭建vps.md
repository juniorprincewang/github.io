---
title: 搭建vps
date: 2017-08-04 16:47:02
tags:
---

本篇讲述了自己动手翻墙访问谷歌的过程。

利用服务商Vultr的海外专用虚拟网络主机(VPS)搭建Shadowsocks的服务，利用VPN技术实现访问代理。

<!-- more -->

# 购买云主机

经别人推荐，共有几款不同的购买平台，
+ 1 [搬瓦工](https://bandwagonhost.com/)，这个网站我是打不开，据说被墙了。
+ 2 [Linode](https://www.linode.com/)，老牌VPS提供商，但是我还没尝试过。
+ 3 [VULTR](https://www.vultr.com/)，我是奔着5$/月的价格去的，去了才发现，售罄！

我最后选择了VULTR这家，买VPS的流程很简单，注册=>绑定信用卡或者PayPal甚至比特币=>勾选要买的Server地址=>选择服务器的类型=>

![location](../搭建vps/location.png)

## 
# 

搭建VPS的过程中遇到的问题。
安装的操作系统是CENTOS 7。


没有netstat工具。
```
	yum install net-tools
```

用ps查看进程的id号：
```
	ps -ef | grep Name 
```
查看到进程id之后，使用netstat命令查看其占用的端口：

```
netstat -nap | grep pid  
```


# 参考资料

[1] http://blog.csdn.net/zwc591822491/article/details/52802692
[2] https://www.vultrclub.com/174.html