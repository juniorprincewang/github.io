---
title: 解决ubuntu升级失败无法登录系统
date: 2018-08-21 17:10:34
tags:
- ubuntu
categories:
- [solutions]
---
本篇博客解决 `ubuntu` 系统升级中断后无法正常登录的问题。
<!-- more -->
虚拟机 `ubuntu 16.04 LTS` 系统提示升级 `ubuntu 18.04 LTS` ， 手贱点了更新，但是途中关机，导致开机无法正常登录。

# 解决
UI界面无法正常操作，只好进入到终端，CTRL+ALT+F1…F6可以开启多个虚拟终端。切换到 `root` 用户下。
考虑到是升级问题，执行 `apt-get upgrade` ，报错。
将显示的问题输出到指定文件中。 `apt-get upgrade > /tmp/upgrade` ，然后查看问题。
系统给出的提示建议使用 `sudo apt-get -f install` 命令。OK，那就执行。
再执行
```
sudo apt-get dist-upgrade
```
更新完毕， `reboot` 后成功恢复。

## `update` 、 `upgrade` 、 `dist-upgrade` 三者的区别

> update是下载源里面的metadata的. 包括这个源有什么包, 每个包什么版本之类的.
> upgrade:系统将现有的Package升级,如果有相依性的问题,而此相依性需要安装其它新的Package或影响到其它Package的相依性时,此Package就不会被升级,会保留下来. 

> dist-upgrade:可以聪明的解决相依性的问题,如果有相依性问题,需要安装/移除新的Package,就会试着去安装/移除它. (以通常这个会被认为是有点风险的升级) 

# 参考
1. [Ubuntu apt-get upgrade报错问题](https://www.cnblogs.com/ocean1100/articles/7641875.html)

