---
title: 解决Ubuntu18.04开机启动缓慢
date: 2019-02-27 14:12:14
tags:
- ubuntu
categories:
- solutions
---
本篇博客解决了Ubuntu18.04开机启动过程中，长时间停留在Ubuntu LOGO，再进入桌面系统的登录界面。
<!-- more -->

查看启动过程的进程占用时间
```
systemd-analyze blame
```
得到的结果令人咋舌

> 3min 9.523s plymouth-quit-wait.service
>      5.423s NetworkManager-wait-online.service
> 	   2.930s dev-sda1.device

原来这是个Ubuntu的bug，安装 `haveged` or `rng-tools ` 即可。

```
sudo apt install haveged
```

重启果然变快了。

# 参考  
[ubuntu 16.04 开机慢、ubuntu 18.04](https://blog.csdn.net/u010953692/article/details/85038213)
[Why does plymouth-quit-wait.service take me 3 minutes to enter my desktop?
](https://askubuntu.com/questions/1103750/why-does-plymouth-quit-wait-service-take-me-3-minutes-to-enter-my-desktop)