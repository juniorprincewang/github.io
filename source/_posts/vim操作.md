---
title: vim操作
date: 2017-10-04 12:27:42
tags:
- vim
- linux
categories:
- [linux,vim]
---

平时操作vim有些命令经常查，索性都记录在这篇博客里。包括搜索替换，行操作。
<!-- more -->

# 移动光标

|命令	|作用|
|--|--|
|h,j,k,l	|h表示往左，j表示往下，k表示往上，l表示往右|
|Ctrl+f	|下一页|
|Ctrl+b	|上一页|
|w, e, W, E	|跳到单词的后面，小写包括标点|
|b, B	|以单词为单位往前跳动光标，小写包含标点|
|O	|开启新的一行|
|^	|一行的开始|
|$	|一行的结尾|
|gg	|文档的第一行|
|[N]G	|文档的第N行或者最后一行|

# 搜索

|命令|作用|
|--|--|
|/pattern|向后搜索|
|?pattern|向前搜索|
|/\Cpattern |区分大小写的查找|
|/\cpattern | 不区分大小写的查找|
|:set ic(ignorecase 的缩写) | 通过指令指定设置忽略大小写|
|:set noic(noignorecase 的缩写)| 通过指令指定设置不忽略大小写|
|`/\<pattern\>`|全字匹配pattern|  
|N	|光标到达搜索结果的前一个目标|
|n	| 光标到达搜索结果的后一个目标|
|:set hlsearch| 搜索的所有匹配项将高亮显示 |




# 插入模式

|命令	|作用|
|---|---|
|i	|插入到光标前面|
|I	|插入到行的开始位置|
|a	|插入到光标的后面|
|A	|插入到行的最后位置|
|o	|插入到光标所在行的下一行|
|O	|插入到光标所在行的上一行|
|Esc	|关闭插入模式|

# 编辑

|命令 |	作用 |
|--------|---------------|
|r  |在插入模式替换光标所在的一个字符|
|J	|合并下一行到上一行|
|s	|删除光标所在的一个字符, 光标还在当行|
|S	|删除光标所在的一行，光标还在当行，不同于dd|
|`u`	|撤销上一步操作|
|`ctrl+r`	|恢复上一步操作|
|.	|重复最后一个命令|
|~	|变换为大写|
|[N]>>	|一行或N行往右移动一个tab|
|[N]<<	|一行或N行往左移动一个tab|

# 剪切和复制

|命令|	作用|
|------|------|
|dd|	删除一行|
|dw|	删除一个单词|
|x|	删除后一个字符|
|X|	删除前一个字符|
|D|	删除一行最后一个字符|
|[N]yy|	复制一行或者N行|
|yw|	复制一个单词|
|p|	粘贴|

# 替换

## substitute命令

格式如下：
```
:[range]s/{pattern}/{string}/[c|e|g|i]

参数定义：
	1. range：范围，1,7表示第1-7行，
            1,$表示1-最后一行；
            常用%表示所有行；
            #表示前一次编辑的文章。
	2. /{pattern}/ ：与查找一样，正则表达式
	3. /{string}/ ：需要替换的内容

	c : 确认替换，每次替换询问
	e : 不显示error
	g : `整行替换`，不加这个开关只会替换一行最开始的匹配项。
	i : 不区分大小写

```

替换域中的特殊字符，可以通过查询 `:h sub-replace-special` 获悉。  

+ 全局替换命令  
```
:%s/{pattern}/{string}/[flag]
```

# 视觉模式

|命令|	作用|
|----|------------|
|v|	选中一个或多个字符|
|V|	选中一行|

# 纵向编辑模式

|命令|	作用|
|----|------------|
|ctrl-v | 进入纵向编辑模式|
|r | 进入修改模式|
|I |进入行首插入模式|
|A |进入行尾插入模式|

# 行号操作

|命令|	作用|
|----|------------|
|:set nu | 显示行号|
|:set nonu | 不显示行号|
|12gg | 光标跳到12行|

# 刷新文件（reload）

```
`:e`
```

# 管道操作

`%!Shell`命令。

`%!` 符号可以将 `VIM` 当前缓冲区中的内容输出到管道中，并启动后面的 `Shell` 命令

比如，`%!xxd`，将文本转换成二进制形式，以十六进制hex格式展示。

# `ctrl + s`

使用vim时，如果你不小心按了 `ctrl + s`后，发现不能输入任何东西了，像死掉了一般。  
其实vim并没有死掉，这时vim只是停止向终端输出而已，要想退出这种状态，只需按 `ctrl + q` 即可恢复正常。  

# 查看文件编码  
```
:set fileencoding
```
即可显示文件编码格式。  
在Vim中直接进行转换文件编码,比如将一个文件转换成utf-8格式。 
```
:set fileencoding=utf-8
```

# 查看文件格式  

> /bin/sh^M: bad interpreter: No such file or directory

这是不同系统编码格式引起的：在windows系统中编辑的.sh文件可能有不可见字符，所以在Linux系统下执行会报以上异常信息。  

```
:set fileformat?
```
设置文件格式为 unix  
```
:set fileformat=unix
```

# 会话和viminfo

使用会话(session)和viminfo，可以把你编辑环境保存下来，然后你在下次启动vim后，可以再恢复回这个环境。

+ 会话信息中保存了所有窗口的视图，外加全局设置。
+ viminfo信息中保存了命令行历史(history)、搜索字符串历史(search)、输入行历史、非空的寄存器内容(register)、文件的位置标记(mark)、最近搜索/替换的模式、缓冲区列表、全局变量等信息。

保存：
```
:mksession project.vim               "创建一个会话文件
:wviminfo project.viminfo            "创建一个viminfo文件
:qa    
```

恢复：
```
:source path/to/project.vim  '载入会话文件
:rviminfo project.viminfo            '读入viminfo文件
```

更详细的要参考[vi/vim使用进阶: 使用会话和viminfo](https://blog.easwy.com/archives/advanced-vim-skills-session-file-and-viminfo/)。

# 参考
[1] [Vim 常用命令总结](http://pizn.github.io/2012/03/03/vim-commonly-used-command.html)
[2] [VIM进阶笔记(1) —— 查找与替换](http://brieflyx.me/2015/linux-tools/vim-advanced-1/)
[3] [vim 技巧 – 查找的时候忽略大小写](https://xwsoul.com/posts/472)
[4] [Training and Tutorials Vim 101: A Beginner’s Guide to Vim](https://www.linux.com/training-tutorials/vim-101-beginners-guide-vim/)  
