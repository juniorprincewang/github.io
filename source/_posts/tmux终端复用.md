---
title: Tmux终端复用
date: 2018-01-24 15:03:58
tags:
- linux
- tmux
categories:
- linux
---

之前在Ubuntu下使用多个终端习惯了，方便来回切换，但通过SSH远程连接到服务器，无法开启多个终端，只能再次ssh连接一次。那么问题来了，怎么通过ssh连接Linux，使用多个终端呢？答案是终端模拟器Tmux。
<!--more-->
# Tmux

Tmux（"Terminal Multiplexer"的简称）是一款BSD 协议发布的终端复用软件，用来在服务器端托管同时运行的 Shell，可以让我们在单个屏幕的灵活布局下开出很多终端。

使用Tmux的好处不仅可以在一个终端复用多个终端，Tmux还有一个session的概念，在session中可以保存当前的终端。在SSH连接终端再连接服务器之后，通过连接上次session可以恢复SSH断开前的状态。这简直是远程办公的神器。

# 安装

在Ubuntu下安装很简单。
```
sudo apt install tmux
```

# Tmux基本概念

Tmux的元素共分为三层。
1. Session：一组窗口的集合，通常用来概括同一个任务。session可以有自己的名字便于任务之间的切换。
2. Window：单个可见窗口。Windows有自己的编号，类似于Tab。
3. Pane ：在Window中被划分成小块的窗口。

# 基本操作

基本操作分为在Tmux之外通过tmux执行的命令和在Tmux内通过`prefix`执行的命令。

## session管理

使用Tmux的最好方式是使用会话的方式，这样你就可以以你想要的方式，将任务和应用组织到不同的会话中。如果你想改变一个会话，会话里面的任何工作都无须停止或者杀掉。
而在Tmux之外的常用命令就是开启session和连接session。

|操作     |       命令|
|---------|-----------|
|tmux new -s blog|创建一个叫做blog的session|
|tmux attach -t blog |重新开启叫做blog的session|
|tmux switch -t project| 转换到叫做project的session|
|tmux list-sessions / tmux ls | 列出现有的所有 session|
|tmux detach | 离开当前开启的 session|
|tmux kill-server |关闭所有 session|

更常用的是在 tmux 中直接通过**Prefix-Command**前置操作：所有下面介绍的快捷键，都必须以前置操作开始。tmux默认的前置操作是`CTRL+b`。例如，我们想要新建一个窗体，就需要先在键盘上摁下`CTRL+b`，松开后再摁下`c`键。

下面所有操作的`prefix`均代表**`CTRL+b`**，也就是书说需要先摁`CTRL+b`再摁以下操作。

seesion的常用操作可以简化为如下命令。

|操作     |       命令|
|---------|-----------|
|?        |# 快捷键帮助列表|
|:new<CR> |# 创建新的 Session，其中 : 是进入 Tmux 命令行的快捷键|
|s        |# 列出所有 Session，可通过 j, k, 回车切换|
|d        |# detach，退出 Tmux Session，回到父级 Shell|
|$        |# 为当前 Tmux Session 命名|

## window管理

|操作     |       命令|
|---------|-----------|
|c        |# 创建 Window|
|&        |# 关闭当前Window| 
|[0-9]    |# 切换到第 n 个 Window|
|,        |# 为当前 Window 重命名|
|p        |# 切换至上一窗口|
|n        |# 切换至下一窗口|
|l        |# 前后窗口间互相切换|
|w        |# 通过窗口列表切换窗口|
|.        |# 修改当前窗口编号，相当于重新排序|
|f        |# 在所有窗口中查找关键词，便于窗口多了切换|

## pane管理

|操作     |       命令|
|---------|-----------|
|%        |# 左右切分 Pane|
|"        |# 上下切分 Pane|
|space键  |# 切换 Pane 布局|
|z        |# 暂时把一个窗体放到最大|
|x        |# 关闭当前分屏|
|!        |# 将当前面板置于新窗口,即新建一个窗口,其中仅包含当前面板|
|ctrl+方向键 |# 以1个单元格为单位移动边缘以调整当前面板大小|
|alt+方向键 |# 以5个单元格为单位移动边缘以调整当前面板大小|
|q        |# 显示面板编号|
|o        |# 选择当前窗口中下一个面板|
|方向键   |# 移动光标选择对应面板|
|{        |# 向前置换当前面板|
|}        |# 向后置换当前面板|
|alt+o    |# 逆时针旋转当前窗口的面板|
|ctrl+o   |# 顺时针旋转当前窗口的面板|
|page up  |# 向上滚动屏幕，q 退出|
|page down |# 向下滚动屏幕，q 退出|





# 参考文献
1. [Tmux - Linux从业者必备利器](http://cenalulu.github.io/linux/tmux/)
2. [tmux 指南](http://wdxtub.com/2016/03/30/tmux-guide/)
3. [优雅地使用命令行：Tmux 终端复用](http://harttle.land/2015/11/06/tmux-startup.html)
4. [如何使用Tmux提高终端环境下的效率](https://linux.cn/article-3952-1.html)