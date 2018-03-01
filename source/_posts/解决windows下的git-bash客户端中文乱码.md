---
title: 解决windows下的git bash客户端中文乱码
date: 2018-03-01 10:52:31
tags:
- git bash
- windows
categories:
- solutions
---
本篇博客解决了，在windows中使用Git Bash时，中文总是以`linux\346\223\215\346\226\207\344\273\266\346\223\215\344\275\234`乱码形式出现的问题。
<!-- more -->

# 原因

Git是Linux中开发的，Linux的编码方式是`UTF-8`，而windows由于历史问题无法全面支持`UTF-8`编码方式。因此Windows中的Git Bash会出现中文乱码问题。

# 解决办法

```
git config --global core.quotepath false  		# 显示 status 编码
git config --global gui.encoding utf-8			# 图形界面编码
git config --global i18n.commit.encoding utf-8	# 提交信息编码
git config --global i18n.logoutputencoding utf-8	# 输出 log 编码
export LESSCHARSET=utf-8
# 最后一条命令是因为 git log 默认使用 less 分页，所以需要 bash 对 less 命令进行 utf-8 编码
```
# 参考
[1] [解决 Git 在 windows 下中文乱码的问题](https://gist.github.com/nightire/5069597)
[2] [Git for windows 中文乱码解决方案](https://segmentfault.com/a/1190000000578037)