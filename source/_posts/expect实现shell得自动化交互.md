---
title: expect实现shell得自动化交互
date: 2019-05-18 13:09:09
tags:
- shell
- expect
categories:
- [linux,shell]
---
expect命令可以帮助shell自动化交互，省去了手动输入选项得烦恼。本篇博客讲解了expect得shell脚本编写。
<!-- more -->

Expect是Unix系统中用来进行自动化控制和测试的软件工具，作为**Tcl**脚本语言的一个扩展，应用在交互式软件中如telnet，ftp，Passwd，fsck，rlogin，tip，ssh等等。该工具利用Unix伪终端包装其子进程，允许任意程序通过终端接入进行自动化控制。

# 安装

```
sudo apt install expect
```

# interpreter

脚本得解释器是 `which expect` 得路径，而不是通常使用的 `/bin/bash`。这两个的语法很多地方不同，要注意。  

```
#!/usr/bin/expect -f
```

不使用expect的解释器，会报错：  
> spawn - command not found!

当 脚本既用到 `#!/bin/bash` 又用到 `#!/usr/bin/expect` 时候，可以将 expect逻辑单独写一个脚本，然后让 bash 脚本调用。

> However, within your script, you have expect commands such as spawn and send. 
> Since the script is being read by bash and not by expect, this fails. 
> You could get around this by writing different expect scripts and calling them from your bash script or by translating the whole thing to expect.  

from [spawn - command not found!](https://unix.stackexchange.com/a/187366)

# 语法

expect使用的是 tcl语法。

+ 一条Tcl命令由空格分割的单词组成. 其中, 第一个单词是命令名称, 其余的是命令参数 
cmd arg arg arg
+ $符号代表变量的值. 在本例中, 变量名称是foo. 
$foo
+ 方括号执行了一个嵌套命令. 例如, 如果你想传递一个命令的结果作为另外一个命令的参数, 那么你使用这个符号
[cmd arg]
+ 双引号把词组标记为命令的一个参数. "$"符号和方括号在双引号内仍被解释
"some stuff"
+ 大括号也把词组标记为命令的一个参数. 但是, 其他符号在大括号内不被解释
{some stuff}
+ 反斜线符号是用来引用特殊符号. 例如：n 代表换行. 反斜线符号也被用来关闭"$"符号, 引号,方括号和大括号的特殊含义
+ 输出用 puts

对于传入参数的处理：
+ $argv，参数数组，
    使用[lindex $argv n]获取，$argv 0为脚本名字  

+ $argc，参数个数

在使用时候要`set` 赋值。
```
set username [lindex $argv 1]  # 获取第1个参数
set passwd [lindex $argv 2]    # 获取第2个参数
```


# 例子

实现自动的telnet会话的简单例子。

```
 # 假定 $remote_server, $my_user_id, $my_password, 和$my_command 已经读入。
  # 向远程服务器请求打开一个telnet会话，并等待服务器询问用户名
  spawn telnet $remote_server
  expect "username:"

  # 输入用户名，并等待服务器询问密码
  send "$my_user_id\r"
  expect "password:"

  # 输入密码，并等待键入需要运行的命令
  send "$my_password\r"
  expect "%"

  # 输入预先定好的密码，等待运行结果
  send "$my_command\r"
  expect "%"

  # 将运行结果存入到变量中，显示出来或者写到磁盘中
  set results $expect_out(buffer)

  # 退出telnet会话，等待服务器的退出提示EOF
  send "exit\r"
  expect eof
```

# 参考

[expect - 自动交互脚本](http://xstarcd.github.io/wiki/shell/expect.html)
[Expect wiki](https://zh.wikipedia.org/wiki/Expect)
[用expect命令实现Shell的自动化交互](https://segmentfault.com/a/1190000012194543)