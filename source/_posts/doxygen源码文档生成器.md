---
title: doxygen源码文档生成器
date: 2019-01-09 19:02:25
tags:
- vim
- linux
categories:
- vim
---

苦于看内核驱动源码的烦恼，Doxygen工具很好的解决了此问题。  Doxygen可以很好的构建出类层次结构和全局变量，不同的用户定义类型，以及函数调用图分析等等。

<!-- more -->
Doxygen是一个适用于C++、C、Java、Objective-C、Python、IDL（CORBA和Microsoft flavors）、Fortran、VHDL、PHP、C#和D语言的文档生成器。   

可产生出来的文档格式有：HTML、 XML、 LaTeX、 RTF、 CHM 。

# 安装 doxygen

这里需要注意，Doxygen版本低于 **1.8.14** 的话 函数调用图会出现问题，详见
[Wrong call/caller graph with Doxygen and GraphViz in C++
](https://stackoverflow.com/questions/47778485/wrong-call-caller-graph-with-doxygen-and-graphviz-in-c)。
**不能**用 `apt-get install doxygen` 安装Doxygen，安装的版本是 **1.8.11**。  
去官网 <http://www.doxygen.nl/download.html>  下载新版本 *1.8.15* 的源码包或者从 github上下载最新版本的。

## Ubuntu/Debian
先安装视图工具 `graphviz`。
```
sudo apt-get install graphviz
```
下载并安装最新版本的Doxygen
```
git clone https://github.com/doxygen/doxygen.git
// or 下载 doxygen-1.8.15.src.tar.gz (4.9MB)
// 		tar -zxvf doxygen-1.8.15.src.tar.gz
// 		cd doxygen-1.8.15
cd doxygen
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make
sudo make install
```

## Windows

先在 <https://graphviz.gitlab.io/download/> 下载 Graphviz的Windows版本，并安装。  
在 <http://www.doxygen.nl/download.html> 下载`doxygen-1.8.15-setup.exe (45.2MB) ` ，并安装。

把doxygen的安装路径写入环境变量PATH中。

# 使用 doxygen 生成文档

## 生成配置文件  

此方法适用于无图形界面操作的Ubuntu和Windows。

```
doxygen -g
```

这个命令在当前目录中生成一个可编辑的配置文件 *oxyfile* 。

## 编辑配置文件

配置文件采用 `<TAGNAME> = <VALUE>` 这样的结构，与 Make 文件格式相似。  
下面是比较重要的标记：

|Tagname|解释| 
|-----------------------|----------------------------|
| DOXYFILE_ENCODING | Doxygen文件的编码方式，默认为UTF-8，若希望支持中文，最好设置为 GB2312|
|PROJECT_NAME | Project 的名字，以一个单词为主，多个单词请使用双引号括住。|
| OUTPUT_DIRECTORY | 输出路径。产生的文件会放在这个路径之下。如果没有填这个路径，将会以目前所在路径作为输出路径。 |
| OUTPUT_LANGUAGE | 输出语言, 默认为English 。 |
| `EXTRACT_ALL` | 默认为NO，只解释有doxygen格式注释的代码；为YES，解析所有代码，即使没有注释 |  
| EXTRACT_PRIVATE | 是否解析类的私有成员 | 
| EXTRACT_STATIC | 是否解析静态项 |
| EXTRACT\_LOCAL_CLASSES | 是否解析源文件（cpp文件）中定义的类 |
| INPUT | 这个标记创建一个以空格分隔的所有目录的列表，这个列表包含需要生成文档的 C/C++ 源代码文件和头文件。<br> 例如，请考虑以下代码片段： *INPUT = /home/user1/project/kernel /home/user1/project/memory* ，  <br> 在这里，doxygen 会从这两个目录读取 C/C++ 源代码。 <br> 如果项目只有一个源代码根目录，其中有多个子目录，那么只需指定根目录并把 <RECURSIVE> 标记设置为 Yes。 | 
| FILE\_PATTERNS | 如果您的INPUT Tag 中指定了目录。您可以透过这个Tag来要求Doxygen在处理时，只针对特定的档案进行动作。 <br>例如：您希望对目录下的扩展名为.c, .cpp及.h的档案作处理。您可设定FILE_PATTERNS = *.c, *.cpp, *.h。    |
| RECURSIVE | 这是一个布尔值的Tag，只接受YES或NO。当设定为YES时，INPUT所指定目录的所有子目录都会被处理。 |
| EXCLUDE | 如果您有某几个特定档案或是目录，不希望经过Doxygen处理。您可在这个Tag中指定。   |
| `EXCLUDE_PATTERNS` | 类似于FILE_PATTERNS的用法，只是这个Tag是供EXCLUDE所使用。 |
| GENERATE_HTML | 若设定为YES ，就会产生HTML版本的说明文件。HTML文件是Doxygen预设产生的格式之一。 |
| HTML\_OUTPUT | HTML文件的输出目录。这是一个相对路径，所以实际的路径为 OUTPUT\_DIRECTORY加上HTML_OUTPUT。这个设定预设为html。      |
| GENERATE_HTMLHELP | 是否生成压缩HTML格式文档（.chm） |
| HTML_HEADER | 要使用在每一页HTML文件中的Header。如果没有指定，Doxygen会使用自己预设的Header。 |
| HTML_FOOTER | 要使用在每一页HTML文件中的Footer。如果没有指定，Doxygen会使用自己预设的Footer。 |
| GENERATE_HTMLHELP | 如设定为YES，Doxygen会产生一个索引文件。这个索引文件在您需要制作windows 上的HTML格式的HELP档案时会用的上。 |
| GENERATE_TREEVIEW | 若设定为YES，Doxygen会帮您产生一个树状结构，在画面左侧。这个树状结构是以JavaScript所写成。所以需要新版的Browser才能正确显示。 |
| GENERATE_LATEX | 设定为YES 时，会产生LaTeX 的文件。不过您的系统必需要有安装LaTeX 的相关工具。   |
| LATEX\_OUTPUT | LaTeX文件的输出目录，与HTML\_OUTPUT用法相同，一样是指在OUTPUT_DIRECTORY之下的路径。预设为latex。 |
| CLASS_DIAGRAMS | 这个标记用来生成类继承层次结构图。要想生成更好的视图，可以从 Graphviz 下载站点 下载 dot 工具。Doxyfile 中的以下标记用来生成图表 |
| `HAVE_DOT` | 如果这个标记设置为 Yes，doxygen 就使用 dot 工具生成更强大的图形，比如帮助理解类成员及其数据结构的协作图。注意，如果这个标记设置为 Yes，<CLASS_DIAGRAMS> 标记就无效了 |
| CLASS_GRAPH | 如果 <HAVE_DOT> 标记和这个标记同时设置为 Yes，就使用 dot 生成继承层次结构图 |
| GRAPHICAL_HIERARCHY | 设置为YES时，将会绘制一个图形表示的类图结构 |
| `<CLASS_GRAPH>`| 如果 <HAVE_DOT> 标记和这个标记同时设置为 Yes，就使用 dot 生成继承层次结构图，而且其外观比只使用 <CLASS_DIAGRAMS> 时更丰富。  |
| `<COLLABORATION_GRAPH>`| 如果 <HAVE_DOT> 标记和这个标记同时设置为 Yes，doxygen 会生成协作图（还有继承图），显示各个类成员（即包含）及其继承层次结构。   |
| `<CALL_GRAPH>`| 如果 <HAVE_DOT> 标记和这个标记同时设置为 Yes，就使用 dot 生成调用全局函数或者类函数的依赖图 |
| `<CALLER_GRAPH>`| 如果 <HAVE_DOT> 标记和这个标记同时设置为 Yes，就使用 dot 生成被调用全局函数或者类函数的依赖图 |





修改 DoxyFile 文件，主要修改以下几项：

```shell
CALL_GRAPH = YES
CALLER_GRAPH = YES
HAVE_DOT = YES
RECURSIVE = YES  （递归检索文件）
EXTRACT_ALL = YES (把源文件，注释都解析出来)
GENERATE_LATEX = NO (不生成Latex)
```
过滤不必要的目录

```shell
EXCLUDE_PATTERNS = */.git/*
EXCLUDE_PATTERNS += */docs/*
EXCLUDE_PATTERNS += */test/*
```

## Windows

对于从图形界面操作，运行 `path_to/doxygen/bin/doxywizard.exe` 可执行文件。
逐一按照要求去完成配置，可以参考 <https://blog.csdn.net/u013354805/article/details/51866991> ，注意 `DOT_PATH` 填写 Graphviz的可执行文件所在文件夹路径，比如 `D:/Graphviz/bin` 。

如果已经保存了配置文件Doxygen，可以从 `File->Open` 来打开。

## 运行 doxygen

```
doxygen Doxyfile
```
在生成文档期间，在 `<OUTPUT_DIRECTORY>` 标记指定的文件夹中，会创建两个子文件夹 `html` 和 `latex` ，直接打开 *html/index.html* 即可看到结果。  


# 参考
1. [Doxygen 的使用](https://www.jianshu.com/p/4e4ce6d6c666)
2. [学习用 doxygen 生成源码文档](https://www.ibm.com/developerworks/cn/aix/library/au-learningdoxygen/index.html)
3. <http://www.doxygen.nl/download.html>
4. <https://graphviz.gitlab.io/download/>