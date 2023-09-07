---
title: 搭建vps
date: 2017-08-04 16:47:02
tags:
- vps
categories:
- [vps]
---

本篇讲述了自己动手翻墙访问谷歌的过程。

利用服务商Vultr的海外专用虚拟网络主机(VPS)搭建Shadowsocks的服务，利用VPN技术实现访问代理。

增加了利用IPv6访问谷歌学术的方法。

<!-- more -->

# 购买云主机

经别人推荐，共有几款不同的购买平台，
+ 1 [搬瓦工](https://bandwagonhost.com/)，这个网站我是打不开，据说被墙了。
+ 2 [Linode](https://www.linode.com/)，老牌VPS提供商，但是我还没尝试过。
+ 3 [VULTR](https://www.vultr.com/)，我是奔着5$/月的价格去的，去了才发现，售罄！

我最后选择了VULTR这家，买VPS的流程很简单，注册=>绑定信用卡或者PayPal甚至比特币=>勾选要买的Server地址=>选择服务器的类型=>

![服务器选择](/img/vps/location.png)


我让国外的同学绑定了他的信用卡^-^，才得以购买成功。买好服务器后， 可以查看服务器的相关信息。需要注意的是，IP Address，Username，Password在之后SSH登陆服务器的时候需要用到。如果需要用到**IPV6**，那么在选择机型的时候，勾选`Enable IPv6`。
![服务器信息](/img/vps/server_information.png)

# 搭建 Shadowsocks 服务

新项目地址迁移到了 [shadowsocks-rust](https://github.com/shadowsocks/shadowsocks-rust)，构建方式换成rust而已。

先安装rust工具集rustup：
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

根据提示导入环境变量：
```
source $HOME/.cargo/env
```
查看rust版本

```
rustc --version
```

那么可以根据 shadowsocks-rust 的 README 指引选择一种安装方式即可。
可以从 crates.io 上安装
```
cargo install shadowsocks-rust
```

或者源码编译安装
```
git clone https://github.com/shadowsocks/shadowsocks-rust.git
cd shadowsocks-rust
cargo build --release
sudo make install TARGET=release
```
安装成功后编写server配置文件

```
{
    "server": "::",
    "server_port": 8388,
    "password": "rwQc8qPXVsRpGx3uW+Y3Lj4Y42yF9Bs0xg1pmx8/+bo=",
    "method": "chacha20-ietf-poly1305",
    // ONLY FOR `sslocal`
    // Delete these lines if you are running `ssserver` or `ssmanager`
    //"local_address": "127.0.0.1",
    //"local_port": 1080
}
```

启动server服务：
```
ssserver -c config.json
```

注意将server服务端口在云服务器安全组规则中和防火墙中放行，启动服务后可以在本地测试下该服务是否成功启动。
```
telnet XX.XX.XX.XX 8388
```

安卓客户端在 [Shadowsocks for Android](https://github.com/shadowsocks/shadowsocks-android) 这个项目，windows客户端在 [Shadowsocks for Windows](https://github.com/shadowsocks/shadowsocks-windows)，下载release版本后安装配置再连接测试即可。

[如何部署Shadowsocks-rust和Cloak](https://mirror.xyz/0x78874f895B96BEc9f48e67BAE188309D285b45a0/Q6n5_2LXgPVDla_oJtcO3EZ3Z98z4LDlryIGId2yMLY)  
[ShadowSocks Rust的配置与优化](https://blog.substitute.tech/blog/20220506-shadowsocks-rust.html)

--------------------------------------
**以下为历史版本**

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

## ubuntu client

这里实验了ss GUI client。  

先去下载客户端 Shadowsocks-Qt5-3.0.1-x86_64.AppImage
： <https://github.com/shadowsocks/shadowsocks-qt5/releases>  

```
chmod +x Shadowsocks-Qt5-3.0.1-x86_64.AppImage
# run 
./Shadowsocks-Qt5-3.0.1-x86_64.AppImage
```
运行此软件后会弹出配置窗口，进行添加、配置就行，选择SOCKS5，最后点击connect。  
至此，TCP代理运行在 127.0.0.1:1080。  
> TCP server listening at 127.0.0.1:1080

为系统网络设置proxy 127.0.0.1:1080。

但此时所有的流量都走代理，包括国内网站。因此还需要设置pac让国内网站不经过代理。  

**配置PAC文件**  

安装 `genpac`  

```
# 如果没有pip工具则先执行安装：
sudo apt install python-pip
sudo pip install genpac
```

```
genpac --pac-proxy "SOCKS5 127.0.0.1:1080" --gfwlist-proxy="SOCKS5 127.0.0.1:1080" --gfwlist-url=https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt --output="autoproxy.pac"
```

或者 GitHub 找到 gfwlist的仓库，把内容复制到你放置 pac文件的文件夹中的gfwlist.txt中

```
genpac --pac-proxy "SOCKS5 127.0.0.1:1080" --gfwlist-proxy="SOCKS5 127.0.0.1:1080" --gfwlist-local="gfwlist.txt" --output="autoproxy.pac"
```

在设置->network中配置automatic，配置路径输入：  
```
file:///xxxx/xxx/xxx/xxx.pac
```



# 访问谷歌学术

你是否有这样的烦恼，访问谷歌学术就得到 "We're sorry..." 的页面。尤其最近2018-12月份IPv6科学上外网方法又被过滤掉后，这种情况一度让人头疼。
按照以上的方法在vultr服务器上配置的SS服务不能成功访问谷歌学术 <https://scholar.google.com> 。

服务器启用IPv6，利用IPv6访问谷歌学术。
具体方法是，这里<https://raw.githubusercontent.com/lennylxx/ipv6-hosts/master/hosts> 有一直维护的IPv6网址，找到谷歌学术这一栏。
```
## Scholar 学术搜索
2404:6800:4008:c06::be scholar.google.com
2404:6800:4008:c06::be scholar.google.com.hk
2404:6800:4008:c06::be scholar.google.com.tw
2404:6800:4005:805::200e scholar.google.cn #www.google.cn
```
并将其添加到 `/etc/hosts` 中，再重启ss，这样就能够在墙内科学上谷歌学术了。

# 后续问题：用了一段时间无法使用

切换服务器，由于vultr按时间收费，可以尝试下不同的位置节点。操作系统我选用 centos6，启用IPv6。  
这里布置服务我选用网上的脚本，安装shadowsocks-libev。

```
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-libev.sh
chmod +x shadowsocks-libev.sh
./shadowsocks-libev.sh 2>&1 | tee shadowsocks-libev.log
```

此加速教程为谷歌BBR加速，Vultr的服务器框架可以装BBR加速，加速后对速度的提升很明显，所以推荐部署加速脚本。该加速方法是开机自动启动，部署一次就可以了。

```
wget --no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh
chmod +x bbr.sh
./bbr.sh
```

安装按成后会提示重启，重启完成后：

查看内核：` uname -r `
结果为：
> 4.18.12-041812-generic

包含4.18就说明内核替换成功。

3.检查是否开启BBR

```
sysctl net.ipv4.tcp_available_congestion_control
# 返回值一般为：net.ipv4.tcp_available_congestion_control = bbr cubic reno

sysctl net.ipv4.tcp_congestion_control
# 返回值一般为：net.ipv4.tcp_congestion_control = bbr

sysctl net.core.default_qdisc
# 返回值一般为：net.core.default_qdisc = fq

lsmod | grep bbr
# 返回值有tcp_bbr则说明已经启动
```

这里启动的是 `ss-server` 进程。
重启的话可以采用

```
ps aux | grep ss-server
kill [$PID of ss-server]
/usr/local/bin/ss-server -v -c /etc/shadowsocks-libev/config.json -f /var/run/shadowsocks-libev.pid
```

其中，默认的配置文件在 */etc/shadowsocks-libev/config.json* 。

# 参考资料

[1] http://blog.csdn.net/zwc591822491/article/details/52802692
[2] https://www.vultrclub.com/174.html
[3] [通过VPS使用VPN或ShadowSocks访问Google或Google Schoolar出现验证码等的解决方法](https://www.polarxiong.com/archives/%E9%80%9A%E8%BF%87VPS%E4%BD%BF%E7%94%A8VPN%E6%88%96ShadowSocks%E8%AE%BF%E9%97%AEGoogle%E6%88%96Google-Schoolar%E5%87%BA%E7%8E%B0%E9%AA%8C%E8%AF%81%E7%A0%81%E7%AD%89%E7%9A%84%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95.html)
[4] [vultr搭建ss/ssr教程(个人学习专用)](https://segmentfault.com/a/1190000016601413?utm_source=tag-newest)
[5] [用Vultr自己搭建ss/ssr服务器教程](https://www.vpscn.net/40.html)