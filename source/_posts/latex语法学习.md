---
title: latex语法学习
date: 2019-08-15 15:48:04
tags:
- latex
categories:
- [latex]
---
整理经常使用到的latex语法和WinEdt 10工具。
<!-- more -->

写论文用到的在线Latex编辑和生成网站是 [overleaf](https://www.overleaf.com)。  
但是有时由于网络不稳定，会遇到掉线的情况。这时候选用的本地Latex编辑和生成软件是WinEdt。 

# latex语法

网上的latex语法教程简直数不胜数，我选择了Overleaf官网给出的教程<https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes>。  

latex的文档后缀名为 *.tex*。  

## hello latex

```
\documentclass{article}
 
\begin{document}
Hello Latex.
\end{document}
```


`\begin{document}` 之前的部分为 *preamble* ，这里的语句可以定义文档的类型，语言，包（package）等等。  
`\begin{document}` 和 `\end{document}` 之间的部分为 *body*。

## 文档类型

`\documentclass{}`  规定文档的类型，可以选择文章article，也可以选择其它类型，如book、letter等等
在写论文时，一般会有模板，这里套上相关的文档类型即可。  

`\documentclass[12pt, a4paper]{article}` 设置纸张大小为 A4。 如果不设置字体大小，默认的字体大小为 10pt。  

对于页大小和页边距的设置可参见：[Page size and margins](https://www.overleaf.com/learn/latex/Page_size_and_margins)。  


## 添加包

`\usepackage{}` ，比如添加图片的包： `\usepackage{graphicx}` 。  

## 添加标题、作者、日期等信息

```
\title{First document}
\author{Hubert Farnsworth \thanks{funded by the Overleaf team}}
\date{February 2017}

\begin{document} 
\maketitle 

Hello world! 

\end{document}
```

**注意**： 需要在*body* 中使用命令 `\maketitle`将这些信息按照预定的格式打印出来。  


## 添加注释

`%` 为注释符。如果文章内容中需要使用%的话，需要在%前面加上反斜杠\。

## 添加章节

```tex
-1  \part{part}
0   \chapter{chapter}
1   \section{section}
2   \subsection{subsection}
3   \subsubsection{subsubsection}
4   \paragraph{paragraph}
5   \subparagraph{subparagraph}
```

+ `\section{}`的花括号内的内容是标题，标题序号默认自动排序。如果不想要标题序号，使用 `\section*{}`。
+ `\part` 和 `\chapter` 用于 *report* 和 *book* 类型。

更多章节内容，参见 [Sections and chapters](https://www.overleaf.com/learn/latex/Sections_and_chapters)。  

## Abstracts

论文中，摘要（*Abstract*）是正文最开始的一段，它高度概括了本文的主旨。  

```
\begin{document}
 
\begin{abstract}
This is a simple paragraph at the beginning of the 
document. A brief introduction about the main subject.
\end{abstract}
\end{document}
```

## 段落

### 换行  

+ 如果要另起一段，需要摁 "Enter" 键两次。即两段之间有一空白行。  
+ `\\` 放在行位换行。  
+ `\par` 放在段尾，起到了换行的作用。  

### Paragraph Alignment(Text Justification)  

默认两端对齐，如果要调整，可以换成 *center*, *flushleft* 和 *flushright* 。  

```
\begin{flushleft}
paragraph
\end{flushleft}
```

### Paragraph Indentation

section默认首段不缩进，紧接着的下一段缩进可由 `\parindent` 控制。但是缩进的长度是由 class 决定得，这可通过命令 `\setlength` 更改。  
`\noindent`放置于段首，段落无缩进。  

```tex
\setlength{\parindent}{10ex}
This is the text in first paragraph. This is the text in first 
paragraph. This is the text in first paragraph. \par
\noindent %The next paragraph is not indented
This is the text in second paragraph. This is the text in second 
paragraph. This is the text in second paragraph.
```


## 粗体、斜体、下划线

+ `\textbf{...}`  设置粗体（Bold）
+ `\textit{...}`  设置斜体（Italics）
+ `\underline{...}` 设置下划线（underline）  
+ `\emph{...}` 命令很有用：在正常文本中被强调的部分呈 *斜体*，但是在斜体文本中，呈正常体。  

更多字体的设置可以参见 [Font sizes, families, and styles](https://www.overleaf.com/learn/latex/Font_sizes,_families,_and_styles)。  

+ Font sizes： `\large` 等
+ Font styles ： `\textbf` 等
+ font families： `\textrm` 等  


## 添加图片 

latex本身不管理图片，需要引入包 *graphicx*，此包包含两个命令 `\includegraphics{...}` 和 `\graphicspath{...}` 。  

+ `\graphicspath{ {images/} }` 表示图片存放路径在当前路径的 *images*文件夹下。  
+ `\includegraphics{universe} ` 表示在文档中引入名称为 *universe* 的图片， *universe* 没有带文件扩展名，文件名不包括空格和点。  

```tex
\usepackage{graphicx}
\graphicspath{ {images/} }
\begin{document}
\includegraphics{universe} 
\end{document}
```

### 标题、标签和引用

```tex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.25\textwidth]{mesh}
    \caption{a nice plot}
    \label{fig:mesh1}
\end{figure}  
As you can see in the figure \ref{fig:mesh1}, the 
function grows near 0. Also, in the page \pageref{fig:mesh1} 
is the same example.
```

+ `\caption{a nice plot}` 标题，可以放在图片上方或下方。
+ `\label{fig:mesh1}` 方便引用。 
+ `\ref{fig:mesh1}` 通过 `\ref{fig:mesh1}` 引用。

## 添加表格

```tex
Table \ref{table:data} is an example of referenced \LaTeX{} elements.
 
\begin{table}[h!]
\centering
\begin{tabular}{||c c c c||} 
 \hline
 Col1 & Col2 & Col2 & Col3 \\ [0.5ex] 
 \hline\hline
 1 & 6 & 87837 & 787 \\ 
 2 & 7 & 78 & 5415 \\
 3 & 545 & 778 & 7507 \\
 4 & 545 & 18744 & 7560 \\
 5 & 88 & 788 & 6344 \\ [1ex] 
 \hline
\end{tabular}
\caption{Table to test captions and labels}
\label{table:data}
\end{table}
```

+ `tabular` 是latex创建表格的tag。
+ ` {c c c c}` 表示有4列。 
+ `\begin{center}` 表示表格内容居中对齐，还可以选择 `l` 和 `r` 。
+ `&` 符号用于分隔表格的条目。  
+ `\\` 用于表格的换行。  
+ `{ |c|c|c| }` 中的 `|` 用于插入一条垂直线。  
+ `\hline` 用于插入一条水平线。

表格的 *标题、标签和引用* 和 图片的基本一致。  

### 表格脚注  

需要用到 `threeparttable` 这个包。  
注意在 `tablenotes` 标签之间使用 `\item`。  

```tex
\usepackage{threeparttable}
%A table with footnotes appearing at the bottom of the table:
\begin{table}
   \centering
   \begin{threeparttable}[b]
   \caption{Table with footnotes after the table}
   \label{tab:test2}
   \begin{tabular}{llll}
   \hline
   column 1 & column 2 & column 3\tnote{1} & column 4\tnote{2} \\
   \hline
   row 1 & data 1 & data 2 & data 3 \\
   row 2 & data 1 & data 2 & data 3 \\
   row 3 & data 1 & data 2 & data 3 \\
   \hline
   \end{tabular}
   \begin{tablenotes}
     \item[1] tablefootnote 1
     \item[2] tablefootnote 2
   \end{tablenotes}
  \end{threeparttable}
\end{table}
 
\end{document}
```

[Add notes under the table](https://tex.stackexchange.com/questions/12676/add-notes-under-the-table)  
[Latex给表格加脚注](https://blog.csdn.net/ShuqiaoS/article/details/86230367)  

## 添加列表

### 无序列表

```tex
\begin{itemize}
  \item The individual entries are indicated with a black dot, a so-called bullet.
  \item The text in the entries may be of any length.
\end{itemize}
```

### 有序列表

```tex
\begin{enumerate}
  \item This is the first entry in our list
  \item The list numbers increase with each entry we add
\end{enumerate}
```

## 添加引用

主要参见 [Bibliography management in LaTeX](https://www.overleaf.com/learn/latex/Bibliography_management_in_LaTeX)。  

### The bibliography file

```tex
@article{einstein,
    author = "Albert Einstein",
    title = "{Zur Elektrodynamik bewegter K{\"o}rper}. ({German})
    [{On} the electrodynamics of moving bodies]",
    journal = "Annalen der Physik",
    volume = "322",
    number = "10",
    pages = "891--921",
    year = "1905",
    DOI = "http://dx.doi.org/10.1002/andp.19053221004",
    keywords = "physics"
}
 
@book{dirac,
    title = {The Principles of Quantum Mechanics},
    author = {Paul Adrien Maurice Dirac},
    isbn = {9780198520115},
    series = {International series of monographs on physics},
    year = {1981},
    publisher = {Clarendon Press},
    keywords = {physics}
}
 
@online{knuthwebsite,
    author = "Donald Knuth",
    title = "Knuth: Computers and Typesetting",
    url  = "http://www-cs-faculty.stanford.edu/~uno/abcde.html",
    addendum = "(accessed: 01.09.2016)",
    keywords = "latex,knuth"
}
 
@inbook{knuth-fa,
    author = "Donald E. Knuth",
    title = "Fundamental Algorithms",
    publisher = "Addison-Wesley",
    year = "1973",
    chapter = "1.2",
    keywords  = "knuth,programming"
}

```

引用 `\cite{einstein}` , `\cite{knuth-fa}` 等。

### cite a website

尝试了 `misc` 和 `online` 两个标签，都不能将url显示出来。  
问题出在没有添加相关的包啊。  
**And remember to load a package such as hyperref or url.**

```
@misc{WinNT,
  title = {{MS Windows NT} Kernel Description},
  howpublished = {\url{http://web.archive.org/web/20080207010024/}},
  note = {Accessed: 2010-09-30}
}
```


+ [How can I use BibTeX to cite a web page?](https://tex.stackexchange.com/a/3608)
+ [How to cite a website in LaTeX](https://coderwall.com/p/wntyia/how-to-cite-a-website-in-latex)  

### quotes in bibliographic field

> use {} as delimiters for all fields in .bib files
> title = {"This is inside quote" and outside quote content}

都使用 `{}` 当作分隔符即可。  

+ [Handling quotes inside quotes in a bibliographic field](https://tex.stackexchange.com/q/65331)


## include sections of files

```
\input{sections/introduction.tex}
```

+ ['\include' sections of files?](https://tex.stackexchange.com/a/31907)


# WinEdt 10的安装

1. 在winedt官网下载winedt 10.2：<http://www.winedt.com/download.html>

[吾爱破解](https://www.52pojie.cn/thread-595351-1-1.html) 中给出了WinEdt 10.2的破解方法：

    注册码：
    Cracker TeCHiScy
    1130140925535334280 

2. 在MikTex官网下载MikTex2.9： <https://miktex.org/download>  

并将路径*path...\MiKTeX 2.9\miktex\bin\x64* 添加到环境变量path中。  

3. 在WinEdt菜单栏 *Options->Execution Mode->TeX System* 中更改 *TeX root* 的路径到MikTeX 2.9的安装目录。  

4. 使用 `pdflatex` 编译文件，使用 `bibtex` 编译参考文献。  

为避免出现 *LaTex Warning: citation undefined*错误, 要如下操作：
使用 pdflatex 编译   
之后使用 bibtex 编译   
之后在使用 pdflatex 编译两次  


# 简历 CV Resume

简历模板可以从[overleaf Gallery — Résumé / CV](https://www.overleaf.com/gallery/tagged/cv)获取，但是很少有支持中文的。  

## 中文的支持

在overleaf上找到的latex cv_resume不支持中文，又找了几个支持中文的resume：
+ [适合中文的简历模板收集](https://github.com/dyweb/awesome-resume-for-chinese)
有基于 ModernCV 模板的，进行了中文字体支持和优化，使用 xelatex 编译。即这个 [cv_resume](https://github.com/geekplux/cv_resume)和[demo of moderncv](https://gist.github.com/juniorprincewang/7a7e0c9bbc4884fc2eeafe041dd02946)
+ [moderncv 的笔记（支持中文）](https://www.xiangsun.org/tex/notes-on-moderncv)

主要的做发是：  

添加xeCJK中文包，使用 xelatex 命令编译源文件。  

```tex
% 该文件使用 xelatex 命令可以编译通过
\documentclass[12pt, a4paper]{article}
\usepackage{fontspec}
\usepackage[slantfont, boldfont]{xeCJK}

% 设置英文字体
\setmainfont{Microsoft YaHei}
\setsansfont{Comic Sans MS}
\setmonofont{Courier New}

% 设置中文字体
\setCJKmainfont{Microsoft YaHei}
\setCJKmonofont{Source Code Pro}
\setCJKsansfont{YouYuan}

% 中文断行设置
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
```

- \setCJKmainfont{} 命令用来设置正文的字体，同时也是 \textrm{} 命令使用的字体。
- \setCJKmonofont{} 用来设置 \texttt{} 命令中的中文使用的字体 。
- \setCJKsansfont{} 用来设置 \textsf{} 命令中的中文使用的字体。

可是仍然有几个偏僻字无法成功显示，这是由于 xeCJK 宏包提供了有限的中文字符。  
使用ctex 宏包或者ctexart 文档类。  
```
\documentclass[UTF8]{ctexart}
```
使用 ctexart documentclass 时候，最好加上 \usepackage[T1]{fontenc}。  

```
\documentclass{article}
\usepackage[UTF8]{ctex}
```

可以使用 latex，pdflatex，xelatex 或者 lualatex 命令来编译 生成 PDF 文件，CTeX 开发者推荐使用 xelatex 命令编译源文件。

+ [如何在 LaTeX 中使用中文](https://jdhao.github.io/2018/03/29/latex-chinese.zh/)
