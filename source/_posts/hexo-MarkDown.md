---
title: hexo中MarkDown语法
date: 2017-08-01 22:51:53
tags:
- markdown
- hexo
categories:
- [hexo]
mathjax: true
---

本篇博客介绍了Markdown最常见命令的语法，有代码有效果。持续更新

<!-- more -->

## 文章头格式

`front-matter`

格式如下：
```
---
title: hexo中MarkDown语法
date: 2017-08-01 22:51:53
tags: [Markdown,hexo]
categories: hexo
toc: true
mathjax: true
---
```
其中，tags可以这样
```
tags:
- markdown
- hexo
```

+ categories
将其分类到 Sports/Baseball 和 Play 两个不同的目录下
```
categories:
  - [Sports,Baseball]
  - [Play]
```


## 分级标题

这个很简单，就是在文字前添加个`#`和空格。`#`表示标签h1，`##` 表示标签h2。

例如：
```
# H1
## H2
### H3
#### H4
##### H5
###### H6
```

## 斜体与粗体

单个`*`或者`_`表示斜体，`**`或者`__`表示粗体。例如
```
	这是*斜体*， 这是**粗体**
	这也是_斜体_， 这也是__粗体__
```
效果为：

这是*斜体*， 这是**粗体**
这也是_斜体_， 这也是__粗体__


## 分割线和删除线

在单独的一行使用 *** 或者 --- 表示分割线。
使用 ~~ 表示删除线。
```
1~~23456~~78
```

1~~23456~~78

## 超链接

### 文字链接

```
[链接文字](链接地址)
```
或者也可以直接用< >，将网址或者邮箱地址放在中间，也能将地址直接转成链接：

```
	<http://juniorprincewang.github.io/>
```
这样可以省劲儿。效果
<http://juniorprincewang.github.io/>

### 图片链接
```
	![图片alt属性](图片链接或路径 "图片标题")，也可以使用html语句<img src="图片地址" width="200" height="200">来自定义图片的大小。
```

如
```
![Logo](/images/logo.png)
```

#### hexo的文章资源文件夹[3]

添加图片有**绝对路径**和**相对路径**两种方法，推荐使用绝对路径的方法，相较而言比较稳定。

+ 绝对路径添加图片的方法  
 
创建文件夹`source/images` 并将图片存放在此文件夹中，然后通过 `![](/images/image.jpg)` 来访问。  
这种方法可适用于 hexo drafts 和 posts 的文档。  

+ 相对路径添加图片的方法

通过将 config.yml 文件中的 post_asset_folder 选项设为 true 来打开。
```
_config.yml
post_asset_folder: true
```
当资源文件管理功能打开后，Hexo将会在你每一次通过 `hexo new [layout] <title> `命令创建新文章时自动创建一个文件夹。这个资源文件夹将会有与这个 markdown 文件一样的名字。  
**引入图片相对路径**

将所有与你的文章有关的资源放在这个关联文件夹中之后，你可以通过相对路径来引用它们。
目录名和文章名一致，只要使用 ~~`![logo](../本地图片测试/logo.jpg)`~~ `![logo](logo.jpg)`就可以插入图片。其中[]里面不写文字则没有图片标题。

+ [Hexo博客搭建之在文章中插入图片](https://yanyinhong.github.io/2017/05/02/How-to-insert-image-in-hexo-post/)  

### 视频链接

引用视频，需要加入一段iframe代码。
```
<script src="/js/youtube-autoresizer.js"></script>
<iframe width="640" height="360" src="https://www.youtube.com/embed/HfElOZSEqn4" frameborder="0" allowfullscreen></iframe>
```


## 引用

### 外引用
使用`>`表示文字引用。 例如：

```
> Never help a child with a task at which he feels he can succeed. ---- Maria Montessori
> > 当一个孩子觉得自己能成功完成一项任务时，千万别去帮他。 ---- Maria Montessori
```
> Never help a child with a task at which he feels he can succeed. ---- Maria Montessori
> > 当一个孩子觉得自己能成功完成一项任务时，千万别去帮他。 ---- Maria Montessori

### 内引用

当需要内引用时，就用空格缩进办法。例如
```
中国
	1. 北京
	2. 天津
```

中国
	1. 北京
	2. 天津

### 交叉引用

在需要引用的分级标题后设置一个 id。  
```
#入门-基本语法 {#start}
```

然后在其它任何地方使用这个唯一的标识符来进行引用以方便**文章内**的跳转。  
```
[入门-基本语法](#start)
```

## 表格

表格使用起来比较麻烦，用`|` 控制分列，`-` 控制分行，`:` 控制对齐方式。例如
```
| Item     | Value     | Qty   |
| :------- | --------: | :---: |
| Computer | 1600 USD  | 5     |
| Phone    | 12 USD    | 12    |
| Pipe     | 1 USD     | 234   |
```
效果：

| Item     | Value     | Qty   |
| :------- | --------: | :---: |
| Computer | 1600 USD  | 5     |
| Phone    | 12 USD    | 12    |
| Pipe     | 1 USD     | 234   |



## 序列

如果要有子级序列，只需要再符号前面添加两个空格。

### 无序列表

无序列表使用 `*`，`+`，`-` 表示。例如
```
+ 无序列表项 一
  + 子无序列表 一
    + 子无序列表 三
+ 无序列表项 二
+ 无序列表项 三
```
+ 无序列表项 一
	- 子无序列表 一
		* 子无序列表 三
+ 无序列表项 二
+ 无序列表项 三

### 有序列表
有序列表使用数字和点`.`表示，例如：
```
1. 有序列表项 一
	1. 子有序列表项 一
	2. 子有序列表项 二
2. 有序列表项 二
3. 有序列表项 三
```
1. 有序列表项 一
	1. 子有序列表项 一
	2. 子有序列表项 二
2. 有序列表项 二
3. 有序列表项 三


## 代码块

### 行内代码块

使用 \` \` 表示行内代码块。比如：

```
人生苦短，我用`python`。
```
效果：

人生苦短，我用`python`。

### 多行代码块

多行代码块使用\`\`\`和\`\`\` 包裹代码。例如
```
```
python -c "print '\x01'*10"
(此处应该有三个` ` `,但是现在无法显示lol)
```

反正效果就是

```
python -c "print '\x01'*10"
```

### 加强的代码块

支持多种编程语言的语法高亮，例如c语言
```
``` c
int main(){
  return main();
}
(此处应该有三个` ` `,但是现在无法显示lol)

```

高亮效果为：
``` c
int main(){
  return main();
}
```


## 注释

用 \ 表示注释，\ 后面的文字解析为纯文本格式。例如
```
\# 这样输出一级标题
```
效果：

\# 这样输出一级标题

## 段落与换行

当一个段落需要包含多个文本行时，需要先在行末敲入 `两个或以上空格` 再 `回车` 。

## LaTeX 公式

hexo 提供了两种公式渲染引擎 `mathjax` 和 `katex`，这里选择 `mathjax`。

### 支持markdown公式编辑配置

在根目录配置文件 *_config.yml*，`math` 的 `engine` 一栏默认选择的是 `mathjax`，对 `src` 注释掉。

```
#math

math:
  engine: 'mathjax' # or 'katex'
  mathjax:
    # src: custom_mathjax_source
    config:
      # MathJax config
  katex:
    css: custom_css_source
    js: custom_js_source # not used
    config:
      # KaTeX config
```

在主题Next配置文件 *blog\themes\next\_config.yml* 中，`math` 的 `per_page` 为 `true` 表示每个文件默认启用 mathjax，而为 `false` 则需要在 `front-matter` 中添加 `mathjax: true`，不加 `mathjax` 和 `mathjax: false` 效果一样，都不启用 mathjax。
例如
```
---
title: Will Render Math
mathjax: true
---
```

启用 `mathjax`，`enable: true`。
```
# Math Formulas Render Support
math:
  # Default (true) will load mathjax / katex script on demand.
  # That is it only render those page which has `mathjax: true` in Front-matter.
  # If you set it to false, it will load mathjax / katex srcipt EVERY PAGE.
  per_page: false

  # hexo-renderer-pandoc (or hexo-renderer-kramed) required for full MathJax support.
  mathjax:
    enable: true
    # See: https://mhchem.github.io/MathJax-mhchem/
    mhchem: false

  # hexo-renderer-markdown-it-plus (or hexo-renderer-markdown-it with markdown-it-katex plugin) required for full Katex support.
  katex:
    enable: false
    # See: https://github.com/KaTeX/KaTeX/tree/master/contrib/copy-tex
    copy_tex: false
```

并在博客文件夹下执行：
```
npm install hexo-math --save
npm un hexo-renderer-marked
npm i hexo-renderer-pandoc
```
`hexo-renderer-pandoc` 需要安装 `pandoc`，安装pandoc参见[这里](https://github.com/jgm/pandoc/blob/master/INSTALL.md)。Windows上安装完成后要重启电脑，不然启动hexo服务会一直提示pandoc exited with code null的错误。

重启后，执行 `hexo clean` 再重新生成或启动服务。
```
$ hexo clean && hexo g -d
# or hexo clean && hexo s
```

### 插入公式
行中公式可以用如下方法表示
```
 $ 数学公式 $
```

独立公式可以用如下方法表示：
```
 $$ 数学公式 $$
```


```
$$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$
```

$$x^{y^z}=(1+{\rm e}^x)^{-2xy^w}$$

### 输入上下标

`^` 表示上标,  `_` 表示下标。如果上下标的内容多于一个字符，需要用 `{}` 将这些内容括成一个整体。上下标可以嵌套，也可以同时使用。

```
$$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$
```
$$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$

### 输入括号和分隔符

`()` 、 `[]` 和 `|` 表示符号本身，使用 `\{\}` 来表示 `{}` 。当要显示大号的括号或分隔符时，要用 `\left` 和 `\right` 命令。

一些特殊的括号：

|输入|		显示 |		输入| 	显示|
|----|-----------|---------|-------|
|	`\langle` | $\langle$	|	`\rangle`	|	$\rangle$ |
|	`\lceil` |	$\lceil$	|	`\rceil`	|	$\rceil$	|
|	`\lfloor`	|	$\lfloor$	|	`\rfloor`	|	$\rfloor$	|
|	`\lbrace`	|	$\lbrace$	|	`\rbrace`	|	$\rbrace$	|


其他符号：  

`\times` : $\times$


更多的去参考 [Cmd Markdown 公式指导手册](https://www.zybuluo.com/codeep/note/163962)

## 参考网站

[1] [Hexo Markdown 简明语法手册](https://hyxxsfwy.github.io/2016/01/15/Hexo-Markdown-%E7%AE%80%E6%98%8E%E8%AF%AD%E6%B3%95%E6%89%8B%E5%86%8C/)
[2] [Hexo基础操作和Markdown语法](https://yuxishe.github.io/2016/11/15/Hexo/)
[3] [Hexo doc 资源文件夹](https://hexo.io/zh-cn/docs/asset-folders.html)
[4] [Markdown语法介绍](https://coding.net/help/doc/project/markdown.html#section-2)
[5] [少为人知的Markdown基础知识](https://sspai.com/post/37273)  
[6] [Mastering Markdown](https://guides.github.com/features/mastering-markdown/)
[7] [Hexo 一篇文章多个 categories](http://aiellochan.com/2018/02/13/hexo/Hexo-%E4%B8%80%E7%AF%87%E6%96%87%E7%AB%A0%E5%A4%9A%E4%B8%AA-categories/)