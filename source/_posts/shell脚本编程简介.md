---
title: shell脚本编程简介
date: 2018-04-16 09:43:08
tags:
- shell
categories:
- [linux,shell]
---

本篇博客介绍shell脚本的语法知识。

<!-- more -->

# 解释器
第一行一般是这样：

	#!/bin/bash
	#!/usr/bin/php

`#!` 是一个约定的标记，它告诉系统这个脚本需要什么解释器来执行。

# 变量
## 定义变量
定义变量时，变量名不加美元符号 `$`，如：

	your_name="test"

注意，**变量名和等号之间不能有空格**。

除了显式地直接赋值，还可以用语句给变量赋值，如：

	for file in `ls /etc`

## 使用变量
使用一个定义过的变量，只要在变量名前面加美元符号 `$` 即可，如：

	your_name="test"
	echo $your_name
	echo ${your_name}

变量名外面的花括号是可选的，加不加都行，加花括号是为了帮助解释器识别变量的边界，比如下面这种情况：

	for skill in Python C Shell Java; do
		echo "I am good at ${skill}Script"
	done

如果不给 `skill` 变量加花括号，写成 `echo "I am good at $skillScript"` ，解释器就会把 `$skillScript` 当成一个变量（其值为空），代码执行结果就不是我们期望的样子了。

推荐给所有变量加上花括号，这是个好的编程习惯。

## 重定义变量
已定义的变量，可以被重新定义，如：

	your_name="test"
	echo $your_name
	
	your_name="foo"
	echo $your_name
	
这样写是合法的，但注意，第二次赋值的时候不能写 `$your_name="foo"` ，使用变量的时候才加美元符。

## 特殊变量

+ `$0` - Bash 脚本的名字.
+ `$1` - `$9` - 传入 Bash 脚本的第1个到第9个参数.
+ `$#` - 传入 Bash 脚本的参数个数.
+ `$*` - 传入 Bash 脚本所有参数.
+ `$@` - 传递给脚本或函数的所有参数。被双引号(" ")包含时，与 $* 稍有不同。
+ `$?` - 上个命令的退出状态，或函数的返回值.
+ `$$` - 当前Shell进程ID。对于 Shell 脚本，就是这些脚本所在的进程ID。.
+ $USER - The username of the user running the script.
+ $HOSTNAME - The hostname of the machine the script is running on.
+ $SECONDS - The number of seconds since the script was started.
+ $RANDOM - Returns a different random number each time is it referred to.
+ $LINENO - Returns the current line number in the Bash script.

# 注释

`BASH` 文件，单行注释用 `#` ，多行注释可以用 `<<COMMENT ……   COMMENT` 来包裹需要注释的命令行。
```
<<COMMENT

something else.

COMMENT
```

# 字符串
字符串是 `shell` 编程中最常用最有用的数据类型，字符串可以用单引号，也可以用双引号，也可以不用引号。单双引号的区别跟PHP类似。

## 单引号

	str='this is a string'

单引号字符串的限制：

- 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的
- 单引号字串中不能出现单引号（对单引号使用转义符后也不行）
 
## 双引号

	your_name='test'
	str="Hello, I know your are \"$your_name\"! \n"

- 双引号里可以有变量
- 双引号里可以出现转义字符

## 字符串操作
### 拼接字符串
	
	your_name="test"
	greeting="hello, "$your_name" !"
	greeting_1="hello, ${your_name} !"
	
	echo $greeting $greeting_1

### 获取字符串长度：

	string="abcd"
	echo ${#string} #输出：4

### 提取子字符串

	string="foos bar"
	echo ${string:1:3} #输出：foo

## 更多
参见本文档末尾的参考资料中[Advanced Bash-Scripting Guid Chapter 10.1](http://tldp.org/LDP/abs/html/string-manipulation.html)

# 测试

内置命令 test 根据表达式expr 求值的结果返回 `0（真）` 或 `1（假）` 。
也可以使用方括号： `test expr` 和 `[ expr ]` 是等价的。 可以用 `$?` 检查返回值；可以使用 `&&` 和 `||` 操作返回值；也可以用后面介绍的各种条件结构测试返回值。


常见的测试命令选项：

|操作符 |	特征|
|-------|-------|
|! EXPRESSION 		|	EXPRESSION 条件为假	|
|-n STRING			|  STRING 长度大于0 		|
|-z STRING			|  STRING 长度为0		|
|STRING1 = STRING2  |	STRING1 与 STRING2 字符串相同 |
|STRING1 != STRING2	| STRING1 与 STRING2 字符串不同 |
|INTEGER1 -eq INTEGER2 |	INTEGER1 数值上与 INTEGER2相等 |
|INTEGER1 -gt INTEGER2 |	INTEGER1 数值上比 INTEGER2 大|
|INTEGER1 -lt INTEGER2 |	INTEGER1 数值上比 INTEGER2 小|
|-d FILE |	FILE 目录存在|
|-e FILE |	FILE 存在.|
|-r FILE |	FILE 存在并且可读.|
|-s FILE |	FILE 存在并且非空.|
|-w FILE |	FILE 存在并且可写.|
|-x FILE |	FILE 存在并且可执行.|
|-f FILE|	FILE普通文件|
|-h FILE|	FILE符号连接（也可以用 -L）|
|-p FILE|	FILE命名管道|
|-S FILE|	FILE套接字|
|-N FILE|	FILE从上次读取之后已经做过修改|

需要注意的是 `=` 与 `-eq` 略有不同， `=` 作字符串比较， `-eq` 作数值比较。
```
root@PowerEdge-R610:~/tools# test 001 = 1
root@PowerEdge-R610:~/tools# echo $?
1
root@PowerEdge-R610:~/tools# test 001 -eq 1
root@PowerEdge-R610:~/tools# echo $?
0
```

# 条件判断

## 流程控制

和Java、PHP等语言不一样，`bash` 的流程控制不可为空，如：

	<?php
	if (isset($_GET["q"])) {
		search(q);
	}
	else {
		//do nothing
	}

在 `bash shell` 里可不能这么写，如果else分支没有语句执行，就不要写这个else。

还要注意，`BASH` 里的 `if [ $foo -eq 0 ]`，这个方括号跟 `Java/PHP` 里 `if` 后面的圆括号大不相同，它是一个可执行程序（和 `ls` , `grep` 一样），想不到吧？在CentOS上，它在 `/usr/bin` 目录下：

	ll /usr/bin/[
	-rwxr-xr-x. 1 root root 33408 4月  16 2018 /usr/bin/[

正因为方括号在这里是一个可执行程序，方括号后面必须加空格，不能写成 `if [$foo -eq 0]`。


## if else
### if

	if condition
	then
		command1 
		command2
		...
		commandN 
	fi

写成一行（适用于终端命令提示符）：

	if `ps -ef | grep ssh`;  then echo hello; fi
	
末尾的fi就是if倒过来拼写，后面还会遇到类似的

### if else
	if condition
	then
		command1 
		command2
		...
		commandN
	else
		command
	fi

### if else-if else

	if condition1
	then
		command1
	elif condition2
		command2
	else
		commandN
	fi

## for while
### for
在开篇的示例里演示过了：

	for var in item1 item2 ... itemN
	do
		command1
		command2
		...
		commandN
	done

写成一行：

	for var in item1 item2 ... itemN; do command1; command2… done;

### C风格的for

	for (( EXP1; EXP2; EXP3 ))
	do
		command1
		command2
		command3
	done

### while
	while condition
	do
		command
	done
	
### 无限循环

	while :
	do
		command
	done

或者

	while true
	do
		command
	done

或者

	for (( ; ; ))

### until

	until condition
	do
		command
	done

## case

	case "${opt}" in
		"Install-Puppet-Server" )
			install_master $1
			exit
		;;

		"Install-Puppet-Client" )
			install_client $1
			exit
		;;

		"Config-Puppet-Server" )
			config_puppet_master
			exit
		;;

		"Config-Puppet-Client" )
			config_puppet_client
			exit
		;;

		"Exit" )
			exit
		;;

		* ) echo "Bad option, please choose again"
	esac

case的语法和C family语言差别很大，它需要一个esac（就是case反过来）作为结束标记，每个case分支用右圆括号，用两个分号表示break

## `&&` 和 `||`

```
#!/bin/bash
if [ -r $1 ] && [ -s $1 ]
then
	echo "This file is useful."
fi
```

# 文件包含
可以使用source和.关键字，如：

	source ./function.sh
	. ./function.sh

在bash里，source和.是等效的，他们都是读入function.sh的内容并执行其内容（类似PHP里的include），为了更好的可移植性，推荐使用第二种写法。

包含一个文件和执行一个文件一样，也要写这个文件的路径，不能光写文件名，比如上述例子中:

	. ./function.sh

不可以写作：

	. function.sh

如果function.sh是用户传入的参数，如何获得它的绝对路径呢？方法是：

	real_path=`readlink -f $1`#$1是用户输入的参数，如function.sh
	. $real_path


# 用户输入
## 执行脚本时传入
## 脚本运行中输入
## select菜单

# stdin和stdout

# 常用的命令
sh脚本结合系统命令便有了强大的威力，在字符处理领域，有grep、awk、sed三剑客，grep负责找出特定的行，awk能将行拆分成多个字段，sed则可以实现更新插入删除等写操作。

## ps
查看进程列表

## grep
### 排除grep自身
### 查找与target相邻的结果

## awk

## sed
### 插入
### 替换
### 删除

## xargs
## curl

# 实战


+ set environment variables in existing shell

```
export PATH="/home/path/to/bin/:$PATH"
```

直接运行脚本只会在子进程 subshell 里面执行，不会对当前shell设置环境变量。  

在当前shell中设置环境变量有两种办法： `source` or `.` 。
```
source ./myscript.sh
```
或者
```
. ./myscript.sh
```

[Shell script to set environment variables](https://stackoverflow.com/a/18548047)

+ 单独以root执行某一命令

```
sudo -u <username> <command>
su <otheruser> -c <command >
# 例如： su root -c 'echo  "hello from $USER"'
```
[How can I execute a script as root, execute some commands in it as a specific user and just one command as root](https://unix.stackexchange.com/a/264239)  
[Run a shell script as another user that has no password](https://askubuntu.com/questions/294736/run-a-shell-script-as-another-user-that-has-no-password)  
[How to write a shell script that runs some commands as superuser and some commands not as superuser, without having to babysit it?](https://stackoverflow.com/a/10220200)  

+ 比较版本

[How to compare a program's version in a shell script?](https://unix.stackexchange.com/a/285928)
[How to compare two strings in dot separated version format in Bash?](https://stackoverflow.com/questions/4023830/how-to-compare-two-strings-in-dot-separated-version-format-in-bash)

比较gcc得版本

```
#!/bin/bash
currentver="$(gcc -dumpversion)"
requiredver="5.0.0"
 if [ "$(printf '%s\n' "$requiredver" "$currentver" | sort -V | head -n1)" = "$requiredver" ]; then 
        echo "Greater than or equal to 5.0.0"
 else
        echo "Less than 5.0.0"
 fi
```

使用[通配符匹配](https://unix.stackexchange.com/users/135943/wildcard)  


+ 判断文件是否存在

[Check if a directory exists in a shell script](https://stackoverflow.com/a/59839)  
```
if [ -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY exists.
fi
```




# 参考
- [Shell脚本编程30分钟入门](https://github.com/qinjx/30min_guides/blob/master/shell.md)
- [Advanced Bash-Scripting Guide](http://tldp.org/LDP/abs/html/)，非常详细，非常易读，大量example，既可以当入门教材，也可以当做工具书查阅
- [Unix Shell Programming](http://www.tutorialspoint.com/unix/unix-shell.htm)
- [Linux Shell Scripting Tutorial - A Beginner's handbook](http://bash.cyberciti.biz/guide/Main_Page)
- [Bash Scripting Tutorial - 2. Variables](https://ryanstutorials.net/bash-scripting-tutorial/bash-variables.php)
- [Bash Scripting Tutorial - 5. If Statements](https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php)
- [Shell特殊变量：Shell $0, $#, $*, $@, $?, $$和命令行参数](http://c.biancheng.net/cpp/view/2739.html)