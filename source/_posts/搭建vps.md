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

![服务器选择](../搭建vps/location.png)


我让国外的同学绑定了他的信用卡^-^，才得以购买成功。买好服务器后， 可以查看服务器的相关信息。需要注意的是，IP Address，Username，Password在之后SSH登陆服务器的时候需要用到。如果需要用到**IPV6**，那么在选择机型的时候，勾选`Enable IPv6`。
![服务器信息](../搭建vps/server information.png)

# 搭建 Shadowsocks 服务

搭建VPS的过程中遇到的问题。
安装的操作系统是CENTOS 7。

首先通过Xshell5客户端通过ssh连接到的服务器。

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

## 安装组件

```
yum install m2crypto python-setuptools 
easy_install pip 
pip install shadowsocks
```

安装完成后配置服务器参数
```
vi /etc/shadowsocks.json
```
并写入如下配置
```
{
    "server":"0.0.0.0", 
    "server_port":443, 
    "local_address": "127.0.0.1", 
    "local_port":1080, 
    "password":"123456", 
    "timeout":300, 
    "method":"aes-256-cfb", 
    "fast_open": false 
}
```
多端口的如下：
```
{ 
    "server":"0.0.0.0", 
    "local_address": "127.0.0.1", 
    "local_port":1080, 
    "port_password": 
    { 
        "443": "443", 
        "8888": "8888" 
    }, 
    "timeout":300, 
    "method":"aes-256-cfb", 
    "fast_open": false 
}
```

这里`server`是本机的IP地址，这里设置成`0.0.0.0`实现了监听IPv4的地址，可以还可以设置成`::`，这样可以监听IPv4和IPv6的地址。
`password`是自己用于连接这个`shadow socks`的密码，自定义就好。

## 安装防火墙
为了进一步提高安全性，安装防火墙并开启防火墙。

```
# 安装防火墙 
yum install firewalld 
# 启动防火墙 
systemctl start firewalld
# 端口号是自己设置的端口 
firewall-cmd --permanent --zone=public --add-port=443/tcp 
firewall-cmd --reload
```

## 启动服务

启动 Shadowsocks 服务
```
ssserver -c /etc/shadowsocks.json
```

如果想干点其他的实现后台运行，使用
```
nohup ssserver -c /etc/shadowsocks.json &
```

# 下载SS客户端

下载客户端，可以直接去[github](https://github.com/ziggear/shadowsocks)上找。这里面资料比较全。找到`Client`->[Windows](https://github.com/shadowsocks/shadowsocks-windows)。去[Download](https://github.com/shadowsocks/shadowsocks-windows/releases)里面找最新的客户端程序。

在SS客户端，填写服务器IP地址，端口号，密码，加密对应`/etc/shadowsocks.json`中的`server`、`server_port`、`password`、`method`这四项。服务器IP一定要填写真实IP地址。

填写完之后点击确定，然后到托盘中右键选择开启"启用系统代理"。

到此，就可以访问[油管](www.youtube.com)啦。

# 参考资料

[1] http://blog.csdn.net/zwc591822491/article/details/52802692
[2] https://www.vultrclub.com/174.html