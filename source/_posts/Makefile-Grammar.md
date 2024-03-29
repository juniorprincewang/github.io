---
title: make 命令
date: 2018-04-24 17:29:39
tags:
- Makefile
categories:
- [linux,Makefile]
---
Makefile 文件由 make 命令构建，make 只是一个根据指定的 Shell 命令进行构建的工具。本篇讲解 make 的命令。
<!-- more -->

Make 命令一般写在 `makefile` 或者 `Makefile` 中，但是也可以用命令行参数 `-f` 指定为其他文件名。

# Makefile文件格式
## 概述

Makefile文件由一系列规则（rules）构成。每条规则的形式如下。
```
<target> : <prerequisites> 
[tab]  <commands>
```
上面第一行冒号前面的部分，叫做"目标"（target），冒号后面的部分叫做"前置条件"（prerequisites）；第二行必须由一个tab键起首，后面跟着"命令"（commands）。

"目标"是必需的，不可省略；"前置条件"和"命令"都是可选的，但是两者之中必须至少存在一个。

每条规则就明确两件事：构建目标的前置条件是什么，以及如何构建。下面就详细讲解，每条规则的这三个组成部分。

## 目标

一个目标（target）就构成一条规则。目标通常是文件名，指明Make命令所要构建的对象，比如上文的 a.txt 。目标可以是一个文件名，也可以是多个文件名，之间用空格分隔。

除了文件名，目标还可以是某个操作的名字，这称为"伪目标"（phony target）。
```
clean:
      rm *.o
```
上面代码的目标是clean，它不是文件名，而是一个操作的名字，属于"伪目标 "，作用是删除对象文件。
```
make  clean
```
但是，如果当前目录中，正好有一个文件叫做 `clean` ，那么这个命令不会执行。因为Make发现 `clean` 文件已经存在，就认为没有必要重新构建了，就不会执行指定的 `rm` 命令。

为了避免这种情况，可以明确声明 `clean` 是"伪目标"，写法如下。

```
.PHONY: clean
clean:
        rm *.o temp
```

声明clean是"伪目标"之后，make就不会去检查是否存在一个叫做clean的文件，而是每次运行都执行对应的命令。像.PHONY这样的内置目标名还有不少，可以查看手册。

如果Make命令运行时没有指定目标，默认会执行Makefile文件的第一个目标。

```
make
```

上面代码执行Makefile文件的第一个目标。

## 前置条件

前置条件通常是一组文件名，之间用 `空格` 分隔。它指定了"目标"是否重新构建的判断标准：只要有一个前置文件不存在，或者有过更新（前置文件的 `last-modification` 时间戳比目标的时间戳新），"目标"就需要重新构建。
```
result.txt: source.txt
    cp source.txt result.txt
```
上面代码中，构建 `result.txt` 的前置条件是 `source.txt` 。如果当前目录中，`source.txt` 已经存在，那么 `make result.txt` 可以正常运行，否则必须再写一条规则，来生成 `source.txt` :

```
source.txt:
    echo "this is the source" > source.txt
```
上面代码中， `source.txt` 后面没有前置条件，就意味着它跟其他文件都无关，只要这个文件还不存在，每次调用 `make source.txt` ，它都会生成。

```
$ make result.txt
$ make result.txt
```
上面命令连续执行两次 `make result.txt` 。第一次执行会先新建 `source.txt`，然后再新建 `result.txt`。第二次执行，Make发现 source.txt 没有变动（时间戳晚于 result.txt），就不会执行任何操作，result.txt 也不会重新生成。

如果需要生成多个文件，往往采用下面的写法。
```
source: file1 file2 file3
```
上面代码中，`source` 是一个伪目标，只有三个前置文件，没有任何对应的命令。
```
make source
```

执行 `make source` 命令后，就会一次性生成 `file1，file2，file3` 三个文件。这比下面的写法要方便很多。
```
make file1
make file2
make file3
```
## 命令

命令（commands）表示如何更新目标文件，由一行或多行的Shell命令组成。它是构建"目标"的具体指令，它的运行结果通常就是生成目标文件。

每行命令之前必须有一个 `tab键` 。如果想用其他键，可以用内置变量 `.RECIPEPREFIX` 声明。
```
.RECIPEPREFIX = >
all:
> echo Hello, world
```

上面代码用 `.RECIPEPREFIX` 指定，大于号（`>`）替代 `tab键` 。所以，每一行命令的起首变成了 `>` ，而不是 `tab键`。

需要注意的是，每行命令在一个单独的shell中执行。这些Shell之间没有继承关系。
```
var-lost:
    export foo=bar
    echo "foo=[$$foo]"
```
上面代码执行后（` make var-lost` ），取不到 foo 的值。因为两行命令在两个不同的进程执行。一个解决办法是将两行命令写在一行，中间用分号分隔。
```
var-kept:
    export foo=bar; echo "foo=[$$foo]"
```
另一个解决办法是在换行符前加反斜杠转义。
```
var-kept:
    export foo=bar; \
    echo "foo=[$$foo]"
```
最后一个方法是加上.ONESHELL:命令。

```
.ONESHELL:
var-kept:
    export foo=bar; 
    echo "foo=[$$foo]"
```

# Makefile文件语法

## 注释
`#` 在Makefile中表示注释。

## 回声（echoing）

正常情况下，make会打印每条命令，然后再执行，这就叫做回声（ `echoing` ）。

```
test:
    # 这是测试
```

执行上面的规则，会得到下面的结果。
```
$ make test
# 这是测试
```

在命令的前面加上@，就可以关闭回声。
```
test:
    @# 这是测试
```
现在再执行 `make test` ，就不会有任何输出。

由于在构建过程中，需要了解当前在执行哪条命令，所以通常只在注释和纯显示的echo命令前面加上@。
```
test:
    @# 这是测试
    @echo TODO
```

## 通配符
通配符（ `wildcard` ）用来指定一组符合条件的文件名。Makefile 的通配符与 Bash 一致，主要有星号（`*`）、问号（`?`）和 `[...]` 。比如， `*.o` 表示所有后缀名为 `o` 的文件。
```
clean:
        rm -f *.o
```

## 模式匹配

Make命令允许对文件名，进行类似正则运算的匹配，主要用到的匹配符是 `%` 。比如，假定当前目录下有 `f1.c` 和 `f2.c` 两个源码文件，需要将它们编译为对应的对象文件。

```
%.o: %.c
```

等同于下面的写法。
```
f1.o: f1.c
f2.o: f2.c
```
使用匹配符 `%` ，可以将大量同类型的文件，只用一条规则就完成构建。

## 变量和赋值符

Makefile 允许使用等号自定义变量。
```
txt = Hello World
test:
    @echo $(txt)
```
上面代码中，变量 `txt` 等于 `Hello World`。调用时，变量需要放在 `$( )` 之中。

调用Shell变量，需要在美元符号前，再加一个美元符号，这是因为Make命令会对美元符号转义。

```
test:
    @echo $$HOME
```
有时，变量的值可能指向另一个变量。
```
v1 = $(v2)
```
上面代码中，变量 `v1` 的值是另一个变量 `v2` 。这时会产生一个问题，`v1` 的值到底在定义时扩展（静态扩展），还是在运行时扩展（动态扩展）？如果 `v2` 的值是动态的，这两种扩展方式的结果可能会差异很大。

为了解决类似问题，Makefile一共提供了四个赋值运算符 `（=、:=、？=、+=）` ，它们的区别请看StackOverflow。
```
VARIABLE = value
# 在执行时扩展，允许递归扩展。

VARIABLE := value
# 在定义时扩展。

VARIABLE ?= value
# 只有在该变量为空时才设置值。

VARIABLE += value
# 将值追加到变量的尾端。
```

## 内置变量（Implicit Variables）
Make命令提供一系列内置变量，比如，`$(CC)` 指向当前使用的编译器，`$(MAKE)` 指向当前使用的Make工具。这主要是为了跨平台的兼容性，详细的内置变量清单见手册。
```
output:
    $(CC) -o output input.c
```

## 自动变量（Automatic Variables）
Make命令还提供一些自动变量，它们的值与当前规则有关。主要有以下几个。

- `$@`

`$@` 指代当前目标，就是Make命令当前构建的那个目标。比如，`make foo` 的 `$@` 就指代 `foo`。
```
a.txt b.txt: 
    touch $@
```
等同于下面的写法。
```
a.txt:
    touch a.txt
b.txt:
    touch b.txt
```
- `$<`

`$<` 指代第一个前置条件。比如，规则为 `t: p1 p2` ，那么 `$<` 就指代 `p1`。
```
a.txt: b.txt c.txt
    cp $< $@ 
```
等同于下面的写法。
```
a.txt: b.txt c.txt
    cp b.txt a.txt 
```
- `$?`

`$?` 指代比目标更新的所有前置条件，之间以空格分隔。比如，规则为 `t: p1 p2` ，其中 `p2` 的时间戳比 `t` 新，`$?` 就指代 `p2` 。

- `$^`

`$^` 指代所有前置条件，之间以空格分隔。比如，规则为 `t: p1 p2` ，那么 `$^` 就指代 `p1 p2` 。

- `$*`

`$*` 指代匹配符 `%` 匹配的部分， 比如 `%` 匹配 `f1.txt` 中的 `f1` ，`$*` 就表示 `f1`。

- `$(@D)` 和 `$(@F)`

`$(@D)` 和 `$(@F)` 分别指向 `$@` 的目录名和文件名。比如，`$@` 是 `src/input.c`，那么 `$(@D)` 的值为 `src` ，`$(@F)` 的值为 `input.c` 。

- `$(<D)` 和 `$(<F)`

`$(<D)` 和 `$(<F)` 分别指向 `$<` 的目录名和文件名。  


所有的自动变量清单，请看手册。下面是自动变量的一个例子。
```
dest/%.txt: src/%.txt
    @[ -d dest ] || mkdir dest
    cp $< $@
```
上面代码将 `src` 目录下的 `txt` 文件，拷贝到 `dest` 目录下。首先判断 `dest` 目录是否存在，如果不存在就新建，然后，`$<` 指代前置文件（`src/%.txt`）， `$@` 指代目标文件（`dest/%.txt`）。

## 判断和循环
Makefile使用 Bash 语法，完成判断和循环。
```
ifeq ($(CC),gcc)
  libs=$(libs_for_gcc)
else
  libs=$(normal_libs)
endif
```
上面代码判断当前编译器是否是 `gcc` ，然后指定不同的库文件。
```
LIST = one two three
all:
    for i in $(LIST); do \
        echo $$i; \
     done
```
等同于
```
all:
    for i in one two three; do \
        echo $i; \
    done
```
上面代码的运行结果。

	one
	two
	three
+ `foreach`

`foreach` 的语法：`$(foreach  var, list, text)`，参数`list`中的单词逐一取出放到参数`var`中，然后再执行`text`所包含的表达式。每一次会返回一个字符串，循环过程中，`text`所返回的每个字符串会以空格分隔，最后当整个循环结束时，`text`返回的每个字符串所组成的整个字符串（以空格分隔）将会是 `foreach` 函数的返回值。
例如:
```
names := a b c d
files := $(foreach n,$(names),$(n).o)
```
`$(files)`的值是 `a.o b.o c.o d.o`。

## 函数
Makefile 还可以使用函数，格式如下。
```
$(function arguments)
```
或者
```
${function arguments}
```
Makefile提供了许多内置函数可供调用。下面是几个常用的内置函数。更多函数请参考 [8 Functions for Transforming Text](https://www.gnu.org/software/make/manual/html_node/Functions.html)。

- shell 函数

shell 函数用来执行 shell 命令
```
srcfiles := $(shell echo src/{00..99}.txt)
```
- wildcard 函数

wildcard 函数用来在 Makefile 中，替换 Bash 的通配符。
```
srcfiles := $(wildcard src/*.txt)
```
- subst 函数

subst 函数用来文本替换，格式如下。
```
$(subst from,to,text)
```
下面的例子将字符串"feet on the street"替换成"fEEt on the strEEt"。
```
$(subst ee,EE,feet on the street)
```
下面是一个稍微复杂的例子。
```
comma:= ,
empty:=
# space变量用两个空变量作为标识符，当中是一个空格
space:= $(empty) $(empty)
foo:= a b c
bar:= $(subst $(space),$(comma),$(foo))
# bar is now `a,b,c'.
```

- patsubst函数

patsubst 函数用于模式匹配的替换，格式如下。

```
$(patsubst pattern,replacement,text)
```
下面的例子将文件名"x.c.c bar.c"，替换成"x.c.o bar.o"。
```
$(patsubst %.c,%.o,x.c.c bar.c)
```

- 替换后缀名

替换后缀名函数的写法是：`变量名 + 冒号 + 后缀名` 替换规则。它实际上patsubst函数的一种简写形式。
```
min: $(OUTPUT:.js=.min.js)
```
上面代码的意思是，将变量 `OUTPUT` 中的后缀名 `.js` 全部替换成 `.min.js` 。

- call 函数

`call` 可以用于常见新的参数化函数，类似于函数调用，格式如下。
```
$(call variable,param,param,…)
```
`make` 命令会将其展开，每个 `param` 会依次赋值成临时变量 `$(1)`、`$(2)` 等，变量 `$(0)` 被赋值成 `variable`。
例如：
```
reverse = $(2) $(1)

foo = $(call reverse,a,b)
```
得到的结果是 *b a*。

# 参考
- [跟我一起写 Makefile（一）](https://blog.csdn.net/haoel/article/details/2886)
- [Make命令教程](https://www.kancloud.cn/kancloud/make-command/45592)


