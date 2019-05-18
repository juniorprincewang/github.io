---
title: 安全入门
date: 2017-08-01 22:13:00
tags:
- pwn
categories:
- [security,pwn]
---

这里有关于pwn、web、逆向的资料整理。
<!-- more -->

# ctf writeup整理

https://github.com/ctfs

# pwn的入门联系网站

http://pwnable.kr/play.php，

http://smashthestack.org/wargames.html

http://ctf.idf.cn/



# web的入门练习网站

## (网络信息安全攻防学习平台)[http://hackinglab.cn/index.php]

提供基础知识考查、漏洞实战演练、教程等资料。实战演练以 Web 题为主，包含基础关、脚本关、注入关、上传关、解密关、综合关等。

http://hackinglab.cn/index.php


## XCTF_OJ 练习平台

XCTF-OJ （X Capture The Flag Online Judge）是由XCTF组委会组织开发并面向XCTF联赛参赛者提供的网络安全技术对抗赛练习平台。XCTF-OJ平台将汇集国内外CTF网络安全竞赛的真题题库，并支持对部分可获取在线题目交互环境的重现恢复，XCTF联赛后续赛事在赛后也会把赛题离线文件和在线交互环境汇总至XCTF-OJ平台，形成目前全球CTF社区唯一一个提供赛题重现复盘练习环境的站点资源。 
地址：http://oj.xctf.org.cn/


## i春秋

国内比较好的安全知识在线学习平台，把复杂的操作系统、工具和网络环境完整的在网页进行重现，为学习者提供完全贴近实际环境的实验平台。 
地址：http://www.ichunqiu.com/main

## 实验吧

http://www.shiyanbar.com/


## 工具

### IDA Pro

反汇编工具

### gdb

Linux下的调试工具，但是需要安装插件。peda, gef, pwndbg。

- peda

项目：<https://github.com/longld/peda>

```
git clone https://github.com/longld/peda.git ~/peda
echo "source ~/peda/peda.py" >> ~/.gdbinit
```
- gef

据说对堆操作有优势。
项目：<https://github.com/hugsy/gef>
```
wget -O ~/.gdbinit-gef.py -q https://github.com/hugsy/gef/raw/master/gef.py
echo source ~/.gdbinit-gef.py >> ~/.gdbinit
```







# 参考网站

[1] [进攻即是最好的防御！19个练习黑客技术的在线网站](http://blog.csdn.net/AliMobileSecurity/article/details/53929049)