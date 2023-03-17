---
title: shell脚本编程简介
date: 2018-04-16 09:43:08
tags:
- shell
categories:
- [linux,shell]
---

本篇博客介绍shell脚本的语法知识。更多Bash使用请参考[Bash Reference Manual](https://www.gnu.org/software/bash/manual/bash.html)。

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

## 数组  

1. 声明数组  
```
declare -a array
```

2. 直接定义

```
(1) array=(var1 var2 var3 ... varN)
(2) array=([0]=var1 [1]=var2 [2]=var3 ... [n]=varN)
(3) array[0]=var1
    arrya[1]=var2
    ...
    array[n]=varN
```

使用数组：  

```
${array[i]}                     # 引用数组
${array[*]} 或${array[@]}       # 数组所有成员
${!array[*]} 或${!array[@]}     # 数组所有下标
${#array[*]} 或${#array[@]}     # 数组元素个数
${#array[0]}                    # 数组第一个成员的长度
```

比如：  
```
COLOR=("red" "green" "yellow" "blue" [5]="orange")
echo ${#COLOR[*]}
> 5

echo ${!COLOR[*]}
> 0 1 2 3 5

for item in ${COLOR[*]}
do
    printf "   %s/n" $item
done
```

<https://blog.csdn.net/ilovemilk/java/article/details/4959747>

## 使用变量
使用一个定义过的变量，只要在变量名前面加美元符号 `$` 即可，如：
```
	your_name="test"
	echo $your_name
	echo ${your_name}
```
变量名外面的花括号是可选的，加不加都行，加花括号是为了帮助解释器识别变量的边界，比如下面这种情况：
```
	for skill in Python C Shell Java; do
		echo "I am good at ${skill}Script"
	done
```
如果不给 `skill` 变量加花括号，写成 `echo "I am good at $skillScript"` ，解释器就会把 `$skillScript` 当成一个变量（其值为空），代码执行结果就不是我们期望的样子了。

推荐给所有变量加上花括号，这是个好的编程习惯。

## 重定义变量
已定义的变量，可以被重新定义，如：
```
	your_name="test"
	echo $your_name
	
	your_name="foo"
	echo $your_name
```
这样写是合法的，但注意，第二次赋值的时候不能写 `$your_name="foo"` ，使用变量的时候才加美元符。

## 变量的测试和内容替换


|变量配置方式 	|	str 没有配置 |	str 为空字符串 |	str 已配置非为空字符串 |
|--|	--|	--| -- |
| `var=${str-expr}`	| `var=expr` |	`var=`	| `var=$str` |
| `var=${str:-expr}`| `var=expr` |	`var=expr` | 	`var=$str` |
| `var=${str+expr}` |	`var=` |	`var=expr` |	`var=expr` |
| `var=${str:+expr}` |	`var=` |	`var=` |	`var=expr` |
| `var=${str=expr}` |	`str=expr`  `var=expr` |	str 不变 var=	| str 不变 `var=$str` |
| `var=${str:=expr}` |	`str=expr` `var=expr` |	`str=expr` `var=expr`	| str 不变 `var=$str` |
| `var=${str?expr}` |	expr 输出至 stderr |	var= 				|	`var=$str` |
| `var=${str:?expr}` |	expr 输出至 stderr |	expr 输出至 stderr 	 |	`var=$str` |

说明：冒号的作用是，被测试的变量未被配置或者是已被配置为空字符串时，都能够用后面的内容来替换与配置。

[变量的测试与内容替换](http://cn.linux.vbird.org/linux_basic/0320bash.php)

## 特殊变量

+ `$0` - Bash 脚本的名字.
+ `$1` - `$9` - 传入 Bash 脚本的第1个到第9个参数.
+ `$#` - 传入 Bash 脚本的参数个数.
+ `$*` - 传入 Bash 脚本所有参数.
+ `$@` - 传递给脚本或函数的所有参数。被双引号(" ")包含时，与 $* 稍有不同。
+ `$?` - 上个命令的退出状态，或函数的返回值.
+ `$$` - 当前Shell进程ID。对于 Shell 脚本，就是这些脚本所在的进程ID。  
+ `$USER` - The username of the user running the script.
+ `$HOSTNAME` - The hostname of the machine the script is running on.
+ `$SECONDS` - The number of seconds since the script was started.
+ `$RANDOM` - Returns a different random number each time is it referred to.
+ `$LINENO` - Returns the current line number in the Bash script.

### 脚本传参

[How can I pass a command line argument into a shell script?](https://unix.stackexchange.com/a/31419)  
"`$0`" 为脚本名称。  
"`$1`", "`$2`", "`$3`"等分别为第1、第2、第3个参数。  
`$#` 为参数数量。  

```
echo "First arg: $1"
```

help 函数： 

```
helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB -c parameterC"
   echo -e "\t-a Description of what is parameterA"
   echo -e "\t-b Description of what is parameterB"
   echo -e "\t-c Description of what is parameterC"
   exit 1 # Exit script after printing help
}

if [ -z "$1" ] || [ -z "$2" ]
then
   helpFunction
fi

```

### `shift [n]`

`shift` 命令用于对参数左移n个位置，同时左边被覆盖的参数都被销毁，默认n为1。
比如：
```
#!/bin/bash
while [ $# != 0 ];do
	echo "第一个参数为：$1,参数个数为：$#"
	shift
done
```

> run.sh a b c d e f

	第一个参数为：a,参数个数为：6
	第一个参数为：b,参数个数为：5
	第一个参数为：c,参数个数为：4
	第一个参数为：d,参数个数为：3
	第一个参数为：e,参数个数为：2
	第一个参数为：f,参数个数为：1

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
```
	str='this is a string'
```
单引号字符串的限制：

- 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的
- 单引号字串中不能出现单引号（对单引号使用转义符后也不行）
 
## 双引号
```
	your_name='test'
	str="Hello, I know your are \"$your_name\"! \n"
```
- 双引号里可以有变量
- 双引号里可以出现转义字符

## 字符串操作
### 拼接字符串

```
	your_name="test"
	greeting="hello, "$your_name" !"
	greeting_1="hello, ${your_name} !"
	
	echo $greeting $greeting_1
```

### 获取字符串长度：

```
	string="abcd"
	echo ${#string} #输出：4
```

### 提取子字符串
```
	string="foos bar"
	echo ${string:1:3} #输出：foo
```

### 子字符串替换

|format|description|
|--|--|
| `${变量#关键词}` 	| 若变量内容从头开始的数据符合『关键词』，则将符合的**最短**数据删除 |
| `${变量##关键词}` 	| 若变量内容从头开始的数据符合『关键词』，则将符合的**最长**数据删除 |
| `${变量%关键词}` 	| 若变量内容从尾向前的数据符合『关键词』，则将符合的**最短**数据删除 |
| `${变量%%关键词}` 	| 若变量内容从尾向前的数据符合『关键词』，则将符合的**最长**数据删除 |
| `${变量/旧字符串/新字符串}` | 若变量内容符合『旧字符串』则『**第一个**旧字符串会被新字符串取代』|
| `${变量//旧字符串/新字符串}` | 若变量内容符合『旧字符串』则『**全部**的旧字符串会被新字符串取代』|

例子：
```
[centos ~]# x="a1 b1 c2 d2"
[centos ~]# echo ${x}
a1 b1 c2 d2
[centos ~]# echo ${x#*1}
b1 c2 d2
[centos ~]# echo ${x##*1}
c2 d2
[centos ~]# echo ${x%1*}
a1 b
[centos ~]# echo ${x%%1*}
a
[centos ~]# echo ${x/1/3}
a3 b1 c2 d2
[centos ~]# echo ${x//1/3}
a3 b3 c2 d2
[centos ~]# echo ${x//?1/z3}
z3 z3 c2 d2
[centos ~]# basename /usr/bin
bin
[centos ~]# dirname /usr/bin
/usr
```

[What is the meaning of the ${0##...} syntax with variable, braces and hash character in bash?](https://stackoverflow.com/questions/2059794/what-is-the-meaning-of-the-0-syntax-with-variable-braces-and-hash-chara)  
[变量内容的删除、取代与替换](http://cn.linux.vbird.org/linux_basic/0320bash.php)

## 更多
参见本文档末尾的参考资料中[Advanced Bash-Scripting Guid Chapter 10.1](http://tldp.org/LDP/abs/html/string-manipulation.html)

# 测试

内置命令 test 根据表达式expr 求值的结果返回 `0（真）` 或 `1（假）` 。
也可以使用方括号： `test expr` 和 `[ expr ]` 是等价的。 可以用 `$?` 检查返回值；可以使用 `&&` 和 `||` 操作返回值；也可以用后面介绍的各种条件结构测试返回值。


常见的测试命令选项：

|操作符 |	特征|
|-------|-------|
|! EXPRESSION 		|	EXPRESSION 条件为假	|
|EXPRESSION1 **`-a`** EXPRESSION2  |	EXPRESSION1 与 EXPRESSION2 都为 true，此处 `a` 为and |
|EXPRESSION1 **`-o`** EXPRESSION2  |	EXPRESSION1 或者 EXPRESSION2 为 true，此处 `o` 为or |
|-n STRING			|  STRING 长度大于0 	|
|-z STRING			|  STRING 长度为0		|
|STRING1 = STRING2  |	STRING1 与 STRING2 字符串相同 |
|STRING1 != STRING2	| STRING1 与 STRING2 字符串不同 |
|INTEGER1 **`-eq`** INTEGER2 |	INTEGER1 数值上与 INTEGER2相等，equal |
|INTEGER1 **`-ge`** INTEGER2 |	INTEGER1 数值上比 INTEGER2 大或相等， greater than or equal to|
|INTEGER1 **`-gt`** INTEGER2 |	INTEGER1 数值上比 INTEGER2 大， greater than|
|INTEGER1 **`-le`** INTEGER2 |	INTEGER1 数值上比 INTEGER2 小或相等， less than or equal to|
|INTEGER1 **`-lt`** INTEGER2 |	INTEGER1 数值上比 INTEGER2 小，less than|
|INTEGER1 **`-ne`** INTEGER2 |	INTEGER1 数值上比 INTEGER2 不相等，not equal to|
|FILE1 **`-ef`** FILE2 |	FILE1 与 FILE2 有相同的 device 和 inode numbers|
|FILE1 **`-nt`** FILE2 |	FILE1 与 FILE2 相比更新（modification date），newer than|
|FILE1 **`-ot`** FILE2 |	FILE1 与 FILE2 相比更旧，older than|
|-b FILE |	FILE 存在并且是block文件|
|-c FILE |	FILE 存在并且是character文件|
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
```
	<?php
	if (isset($_GET["q"])) {
		search(q);
	}
	else {
		//do nothing
	}
```
在 `bash shell` 里可不能这么写，如果else分支没有语句执行，就不要写这个else。

还要注意，`BASH` 里的 `if [ $foo -eq 0 ]`，这个方括号跟 `Java/PHP` 里 `if` 后面的圆括号大不相同，它是一个可执行程序（和 `ls` , `grep` 一样），想不到吧？在CentOS上，它在 `/usr/bin` 目录下：
```
	ll /usr/bin/[
	-rwxr-xr-x. 1 root root 33408 4月  16 2018 /usr/bin/[
```
正因为方括号在这里是一个可执行程序，方括号后面必须加空格，不能写成 `if [$foo -eq 0]`。


## if else
### if
```
	if condition
	then
		command1 
		command2
		...
		commandN 
	fi
```
写成一行（适用于终端命令提示符）：

	if `ps -ef | grep ssh`;  then echo hello; fi
	
末尾的fi就是if倒过来拼写，后面还会遇到类似的

### if else
```
	if condition
	then
		command1 
		command2
		...
		commandN
	else
		command
	fi
```
### if else-if else
```
	if condition1
	then
		command1
	elif condition2
		command2
	else
		commandN
	fi
```
## for while
### for
在开篇的示例里演示过了：
```
	for var in item1 item2 ... itemN
	do
		command1
		command2
		...
		commandN
	done
```
写成一行：
```
	for var in item1 item2 ... itemN; do command1; command2… done;
```
循环内部同样可以使用 `break` 、 `continue` 等命令。  

### C风格的for
```
	for (( EXP1; EXP2; EXP3 ))
	do
		command1
		command2
		command3
	done
```
### while
```
	while condition
	do
		command
	done
```
### 无限循环
```
	while :
	do
		command
	done
```
或者
```
	while true
	do
		command
	done
```
或者
```
	for (( ; ; ))
```
### until
```
	until condition
	do
		command
	done
```
## case
```
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
```
`case` 的语法和C family语言差别很大，它需要一个esac（就是case反过来）作为结束标记，每个case分支用右圆括号，用两个分号表示break

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
```
	source ./function.sh
	. ./function.sh
```
在bash里，`source` 和 `.` 是等效的，他们都是读入 `function.sh` 的内容并执行其内容（类似PHP里的 `include`），为了更好的可移植性，推荐使用第二种写法。

包含一个文件和执行一个文件一样，也要写这个文件的路径，不能光写文件名，比如上述例子中:
```
	. ./function.sh
```
不可以写作：
```
	. function.sh
```
如果function.sh是用户传入的参数，如何获得它的绝对路径呢？方法是：
```
	real_path=`readlink -f $1`#$1是用户输入的参数，如function.sh
	. $real_path
```

# 用户输入
## 执行脚本时传入
## 脚本运行中输入
## select菜单

# stdin、stdout和stderr

文件描述符0 代表 `stdin`，文件描述符1 代表 `stdout`，文件描述符2 代表 `stderr`。

`&` 指示后面是文件描述符，而不是文件名。因此，使用 `2>&1`，将`>&`视为重定向合并运算符。
而 `2>1` 会将标准错误输出重定向到名为"1"的文件里。

而 `&>` 会将 `stdout` 与 `stderr` 都重定向到输出文件中。

比如 `command &>file` 也可以写成 `command >file 2>&1`。

[What does " 2>&1 " mean?](https://stackoverflow.com/a/818284)

# 常用的命令
sh脚本结合系统命令便有了强大的威力，在字符处理领域，有grep、awk、sed三剑客，grep负责找出特定的行，awk能将行拆分成多个字段，sed则可以实现更新插入删除等写操作。

## ps
查看进程列表

## grep
### 排除grep自身
### 查找与target相邻的结果

## sed
## awk

awk 可对文本的每行进行查找、筛选和处理，是 grep、sed的集大成者。  

awk 内置变量：  

```
ARGC               命令行参数个数
ARGV               命令行参数排列
ENVIRON            支持队列中系统环境变量的使用
FILENAME           awk浏览的文件名
FNR                浏览文件的记录数
FS                 设置输入域分隔符，等价于命令行 -F选项
NF                 浏览记录的域的个数
NR                 已读的记录数
OFS                输出域分隔符
ORS                输出记录分隔符
RS                 控制记录分隔符
```

此外, `$0` 变量是指整条记录， `$1` 表示当前行的第一个域, `$2` 表示当前行的第二个域,......以此类推。  

awk中同时提供了`print` 和 `printf` 两种打印输出的函数。  
其中 `print` 的参数可以是变量、数值或者字符串。字符串必须用双引号引用，参数用逗号分隔。如果没有逗号，参数就串联在一起而无法区分。这里，逗号的作用与输出文件的分隔符的作用是一样的，只是后者是空格而已。  
`printf` 函数，可以格式化字符串，与C语言用法一致。  

awk 的语法和C语言很相似，有变量和赋值、条件语句、循环语句、数组（字典）等，非常方便我们对搜索字段进行复杂处理。  

比如，可以根据一个文件中的关键字去另一个文件筛选包含关键字的行。  

a.txt  
```
key1, value1
key2, value2
key3, value3
key4, value4
```

b.txt  
```
key2
key4
```

```sh
awk -F '[,]' 'NR==FNR{a[$1]=$2;next}NR!=FNR{if($1 in a)print $0":"a[$1]}’ a.txt b.txt
```
此命令的原理是： 
先对a.txt 进行读取处理，再对 b.txt 读取处理。  NR 表示读取的所有行数， FNR 表示读取的当前文件的行数，所以 `NR==FNR` 表示开始读取的第一个文件，而 NR!=FNR 表示读取的第二个文件。  
对每行以 `,` 为分隔符进行分割处理。当处理第一个文件时，每行分割后的结果是 $1="key1"，但是 $2=" value1"，这里**注意**，`$2` 是包含前置空格的。这个有时候需要处理的。
我们定义了一个数组（这里是哈希数组），以 `$1` 为关键字，以 `$2` 为值进行存储，即a["key1"]=" value1"。 `next` 表示不进行对后面命令。
当处理第二个文件时，判断第二个文件的每行的第一个字段 `$1` 是否出现在前面生成的数组 `a` 的关键字中，如果出现，则打印以 key:value 形式打印。  


可以对筛选到的文本进行变为大写字符操作：

```sh
awk '{ print toupper($0) }' <<< "your string"
```

去除前后空格：  

```sh
awk -F, '/,/{gsub(/ /, "", $2); print$1","$2} ' input.txt
# 更具体地 对前空格删除
gsub(/^[ \t]+/,"",$2) 
# 对后空格删除
gsub(/[ \t]+$/,"",$2)}
```

可参考的资料：  
[Using AWK to Process Input from Multiple Files](https://stackoverflow.com/questions/14984340/using-awk-to-process-input-from-multiple-files)  
[What are NR and FNR and what does “NR==FNR” imply?](https://stackoverflow.com/questions/32481877/what-are-nr-and-fnr-and-what-does-nr-fnr-imply)   
[Using multiple delimiters in awk](https://stackoverflow.com/questions/12204192/using-multiple-delimiters-in-awk)  
[Check if an awk array contains a value](https://stackoverflow.com/questions/26746361/check-if-an-awk-array-contains-a-value)  
[Trim leading and trailing spaces from a string in awk](https://stackoverflow.com/questions/20600982/trim-leading-and-trailing-spaces-from-a-string-in-awk)  
[Can I use awk to convert all the lower-case letters into upper-case?](https://stackoverflow.com/questions/14021899/can-i-use-awk-to-convert-all-the-lower-case-letters-into-upper-case)  
[Idiomatic awk](https://backreference.org/2010/02/10/idiomatic-awk/)  
[[awk shell] 判断一个文件中内容在另一个文件中](https://blog.csdn.net/luanjinlu/article/details/78122429)  

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

+ 获得版本  

[Extract version number from file in shell script](https://stackoverflow.com/a/6245903)  

获得CUDA 版本号  

```sh
#!/bin/bash

version="$(cat /usr/local/cuda/version.txt | head -n1|cut -d " " -f3)"
echo $version

majorminor=${version%.*}
echo $majorminor
```

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


# Makefile 中判断一个文件是否存在

Makefile中调用shell函数判断：

```
exist = $(shell if [ -f $(FILE) ]; then echo "exist"; else echo "notexist"; fi;)
ifeq (exist, "exist")
#do something here
endif
```


# 参考
- [Shell脚本编程30分钟入门](https://github.com/qinjx/30min_guides/blob/master/shell.md)
- [Advanced Bash-Scripting Guide](http://tldp.org/LDP/abs/html/)，非常详细，非常易读，大量example，既可以当入门教材，也可以当做工具书查阅
- [Unix Shell Programming](http://www.tutorialspoint.com/unix/unix-shell.htm)
- [Linux Shell Scripting Tutorial - A Beginner's handbook](http://bash.cyberciti.biz/guide/Main_Page)
- [Bash Scripting Tutorial - 2. Variables](https://ryanstutorials.net/bash-scripting-tutorial/bash-variables.php)
- [Bash Scripting Tutorial - 5. If Statements](https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php)
- [Shell特殊变量：Shell $0, $#, $*, $@, $?, $$和命令行参数](http://c.biancheng.net/cpp/view/2739.html)