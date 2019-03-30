---
title: vim插件
date: 2018-01-29 19:49:33
tags:
- vim
categories:
- vim
---

为了提高编程效率，需要多利用现有的工具，比如插件！本篇博客介绍好使的vim插件。
<!-- more -->


# Vundle

[Vundle](https://github.com/VundleVim/Vundle.vim#quick-start)是Vim bundle的缩写，它是一款Vim的插件管理工具。
Vundle可以允许用户

    1. 在`.vimrc`中跟踪和配置插件；
    2. 安装插件，类似于script/bundle；
    3. 升级配置的插件；
    4. 按照名字搜索Vim脚本；
    5. 清除不使用的插件；
    ...

## 安装Vundle
安装很简单，只需要安装到`~/.vim/bundle/Vundle.vim`。
```
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

## 配置Vundle

将下面官网给出的配置信息写入`~/.vimrc`文件中，可以把`Plugin 'file:///home/gmarik/path/to/plugin'`用`"`注释掉，因为这是配置本机路径，很明显此文件在我的机器本地路径不存在。否则会报错。


    set nocompatible              " be iMproved, required
    filetype off                  " required

    " set the runtime path to include Vundle and initialize
    set rtp+=~/.vim/bundle/Vundle.vim
    call vundle#begin()
    " alternatively, pass a path where Vundle should install plugins
    "call vundle#begin('~/some/path/here')

    " let Vundle manage Vundle, required
    Plugin 'VundleVim/Vundle.vim'

    " The following are examples of different formats supported.
    " Keep Plugin commands between vundle#begin/end.
    " plugin on GitHub repo
    Plugin 'tpope/vim-fugitive'
    " plugin from http://vim-scripts.org/vim/scripts.html
    " Plugin 'L9'
    " Git plugin not hosted on GitHub
    Plugin 'git://git.wincent.com/command-t.git'
    " git repos on your local machine (i.e. when working on your own plugin)
    Plugin 'file:///home/gmarik/path/to/plugin'
    " The sparkup vim script is in a subdirectory of this repo called vim.
    " Pass the path to set the runtimepath properly.
    Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
    " Install L9 and avoid a Naming conflict if you've already installed a
    " different version somewhere else.
    " Plugin 'ascenator/L9', {'name': 'newL9'}

    " All of your Plugins must be added before the following line
    call vundle#end()            " required
    filetype plugin indent on    " required
    " To ignore plugin indent changes, instead use:
    "filetype plugin on
    "
    " Brief help
    " :PluginList       - lists configured plugins
    " :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
    " :PluginSearch foo - searches for foo; append `!` to refresh local cache
    " :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
    "
    " see :h vundle for more details or wiki for FAQ
    " Put your non-Plugin stuff after this line



## 安装插件

将想要安装的插件，按照地址填写方法，将地址填写在`~/.vimrc` 中的 **vundle#begin** 和 **vundle#end** 之间就可以。  
添加完 启动Vim，执行`:PluginInstall`即可安装。
可以通过`:PluginUpdate`一键更新所有插件。

参考：
1. [VundleVim/Vundle.vim](https://github.com/VundleVim/Vundle.vim#quick-start)

## 移除插件

1. 编辑.vimrc文件移除的你要移除的插件所对应的plugin那一行。
2. 保存退出当前的vim
3. 重新打开vim，输入命令BundleClean。

## 其他常用命令

+ 更新插件 `BundleUpdate` 
+ 列出所有插件 `BundleList`
+ 查找插件 `BundleSearch`

# 一些常用的插件

通过github repos来定义的插件 


    Bundle 'christoomey/vim-run-interactive'
    Bundle 'Valloric/YouCompleteMe'
    Bundle 'croaky/vim-colors-github'
    Bundle 'danro/rename.vim'
    Bundle 'majutsushi/tagbar'
    Bundle 'kchmck/vim-coffee-script'
    Bundle 'kien/ctrlp.vim'
    Bundle 'pbrisbin/vim-mkdir'
    Bundle 'scrooloose/syntastic'
    Bundle 'slim-template/vim-slim'
    Bundle 'thoughtbot/vim-rspec'
    Bundle 'tpope/vim-bundler'
    Bundle 'tpope/vim-endwise'
    Bundle 'tpope/vim-fugitive'
    Bundle 'tpope/vim-rails'
    Bundle 'tpope/vim-surround'
    Bundle 'vim-ruby/vim-ruby'
    Bundle 'vim-scripts/ctags.vim'
    Bundle 'vim-scripts/matchit.zip'
    Bundle 'vim-scripts/tComment'
    Bundle "mattn/emmet-vim"
    Bundle "scrooloose/nerdtree"
    Bundle "Lokaltog/vim-powerline"
    Bundle "godlygeek/tabular"
    Bundle "msanders/snipmate.vim"
    Bundle "jelera/vim-javascript-syntax"
    Bundle "altercation/vim-colors-solarized"
    Bundle "othree/html5.vim"
    Bundle "xsbeats/vim-blade"
    Bundle "Raimondi/delimitMate"
    Bundle "groenewege/vim-less"
    Bundle "Lokaltog/vim-easymotion"
    Bundle "tomasr/molokai"
    Bundle "klen/python-mode"

其中主要用到的是 *YouCompleteMe* 和 *nerdtree*，需要注意的是安装好后需要添加配置信息。

# YouCompleteMe

代码编辑器怎么能少了自动补齐功能？Vim下值得安装的工具[YouCompleteMe](https://github.com/Valloric/YouCompleteMe)简称[YCM](https://github.com/Valloric/YouCompleteMe)。

YCM是很强大的自动补全引擎，支持的语言包括 C-family，C#，Go，JavaScript， Python，Rust，TypeScript，当然也可以支持其他语言。 功能强大，比较重量级，安装起来也比较麻烦。
官网上给出的安装教程不是很完整，下面介绍我自己的安装过程。

## 安装环境
我的操作系统是`Ubuntu 16.04`，确保 vim 版本是 7.4.143 或以上，并且支持python 2/3脚本。官网给出的结果`Ubuntu 16.04`及以上系统足够配置。不过可以通过`vim --version`确认。

安装必要的开发工具，CMake和Python工具。
```
sudo apt-get install build-essential cmake
sudo apt-get install python-dev python3-dev
```

## 通过Vundle安装YCM

在 vim 的配置文件 `~/.vimrc` 中添加一行（在call vundle#begin() 和 call vundle#end() 之间）
```
Plugin 'Valloric/YouCompleteMe’
```
启动Vim，运行`:PluginInstall` 安装。
虽然安装成功，但是启动Vim后发现报错。
    
    YouCompleteme unavailable : no module named future

再继续下面操作。
## 编译YCM

这里举例支持C-family语法自动补齐功能的安装。
```
cd ~/.vim/bundle/YouCompleteMe
./install.py --clang-completer
```

这个过程会自动安装`clang`，从而在`~/.vim/bundle/YouCompleteMe/python`目录下产生libclang.so文件。

## 配置YCM

复制 .ycm_extra_conf.py 文件。
```
cp ~/.vim/bundle/YouCompleteMe/third_party/ycmd/examples/.ycm_extra_conf.py ~/.vim/
```

在`~/.vimrc`中添加 vim 配置注意下面的 python 解释器的路径要和编译 ycm_core 的时候使用的python 解释器是相同的版本（2 或 3）。
```
let g:ycm_server_python_interpreter='/usr/bin/python'
let g:ycm_global_ycm_extra_conf='~/.vim/.ycm_extra_conf.py'
```

其中`ycm_global_ycm_extra_conf`非常重要，里面设定ycm的搜索头文件路径。.ycm_extra_conf.py文件可以针对具体的代码工程建一个，如果不这么做，那么vim会按照.vimrc中设定的路径去找.ycm_extra_conf.py。
其他一些必要的配置:
    
    " 设置跳转到方法/函数定义的快捷键 
    nnoremap <leader>j :YcmCompleter GoToDefinitionElseDeclaration<CR> 
    " 触发补全快捷键 
    let g:ycm_key_list_select_completion = ['<TAB>', '<c-n>', '<Down>'] 
    let g:ycm_key_list_previous_completion = ['<S-TAB>', '<c-p>', '<Up>'] 
    let g:ycm_auto_trigger = 1 
    " 最小自动触发补全的字符大小设置为 3 
    let g:ycm_min_num_of_chars_for_completion = 3 
    " YCM的previw窗口比较恼人，还是关闭比较好 
    set completeopt-=preview 

其他语言的安装可参见官网。

参考:
1. [Valloric/YouCompleteMe](https://github.com/Valloric/YouCompleteMe#ubuntu-linux-x64)
2. [一步一步带你安装史上最难安装的 vim 插件 —— YouCompleteMe](https://www.jianshu.com/p/d908ce81017a)
3. [Vim+Vundle+YouCompleteMe](http://blog.csdn.net/vintage_1/article/details/21557277)

# NERDTree

[NERDTree](https://github.com/scrooloose/nerdtree)查看当前目录下的目录文件树，可以很方便的找到对应的文件并进行切换编辑。

## 安装
通过Vundle安装。在.vimrc，加入
```
Plugin 'scrooloose/nerdtree' " 加入NERDTree
```

保存退出，并重新进入vim，执行:PluginInstall，等待安装，安装完成会有Done状态提示，这样NERDTree插件就安装上了，执行`:NERDTree命令`，**这里一定要大小写分明**，侧栏文件树形目录就出来了。

## 配置NERDTree

配置NERDTree可以在`~/.vimrc`中添加如下命令。
```
" 想要打开Vim时自动打开NERDTree。
autocmd vimenter * NERDTree

" 使用F2键快速调出和隐藏它；
map <F2> :NERDTreeToggle<CR>

" 将 NERDTree 的窗口设置在 vim 窗口的右侧（默认为左侧）
let NERDTreeWinPos="right"

" 当打开 NERDTree 窗口时，自动显示 Bookmarks
let NERDTreeShowBookmarks=1
```

当然更重要的是如何使用这款插件。

在工作台和目录间切换。

    ctrl + w + h    光标 focus 左侧树形目录
    ctrl + w + l    光标 focus 右侧文件显示窗口
    ctrl + w + w    光标自动在左右侧窗口切换
    ctrl + w + r    移动当前窗口的布局位置

常用操作:

    o       在已有窗口中打开文件、目录或书签，并跳到该窗口
    go      在已有窗口 中打开文件、目录或书签，但不跳到该窗口
    t       在新 Tab 中打开选中文件/书签，并跳到新 Tab
    T       在新 Tab 中打开选中文件/书签，但不跳到新 Tab
    i       split 一个新窗口打开选中文件，并跳到该窗口
    gi      split 一个新窗口打开选中文件，但不跳到该窗口
    s       vsplit 一个新窗口打开选中文件，并跳到该窗口
    gs      vsplit 一个新 窗口打开选中文件，但不跳到该窗口
    !       执行当前文件
    O       递归打开选中 结点下的所有目录
    x       合拢选中结点的父目录
    X       递归 合拢选中结点下的所有目录
    e       Edit the current dif

    双击    相当于 NERDTree-o
    中键    对文件相当于 NERDTree-i，对目录相当于 NERDTree-e

    D       删除当前书签

    P       跳到根结点
    p       跳到父结点
    K       跳到当前目录下同级的第一个结点
    J       跳到当前目录下同级的最后一个结点
    k       跳到当前目录下同级的前一个结点
    j       跳到当前目录下同级的后一个结点

    C       将选中目录或选中文件的父目录设为根结点
    u       将当前根结点的父目录设为根目录，并变成合拢原根结点
    U       将当前根结点的父目录设为根目录，但保持展开原根结点
    r       递归刷新选中目录
    R       递归刷新根结点
    m       显示文件系统菜单
    cd      将 CWD 设为选中目录

    I       切换是否显示隐藏文件
    f       切换是否使用文件过滤器
    F       切换是否显示文件
    B       切换是否显示书签

    q       关闭 NerdTree 窗口
    ?       切换是否显示 Quick Help

切换标签页:
    
    :tabnew [++opt选项] ［＋cmd］ 文件      建立对指定文件新的tab
    :tabc   关闭当前的 tab
    :tabo   关闭所有其他的 tab
    :tabs   查看所有打开的 tab
    :tabp   前一个 tab
    :tabn   后一个 tab

    标准模式下：
    gT      前一个 tab
    gt      后一个 tab

    MacVim 还可以借助快捷键来完成 tab 的关闭、切换
    cmd+w   关闭当前的 tab
    cmd+{   前一个 tab
    cmd+}   后一个 tab


# 安装 vim-airline 插件

这个插件没有很大的实用性，但能增加逼格，增加vim的有趣性。

第一步，我们先把下面的需要配置的文件添加到 ~/.vimrc 中

    " ------------------------安装 vim-airline------------------

    set laststatus=2   " 永远显示状态栏
    set t_Co=256       " 在windows中用xshell连接打开vim可以显示色彩

    "Vim 在与屏幕/键盘交互时使用的编码(取决于实际的终端的设定)        
    :set encoding=utf-8
    :set langmenu=zh_CN.UTF-8
    :set fileencodings=utf-8
    :set fileencoding=utf-8
    :set termencoding=utf-8

    Plugin 'vim-airline'    
    let g:airline_theme="molokai"

    "这个是安装字体后 必须设置此项" 
    let g:airline_powerline_fonts = 1  
    "打开tabline功能,方便查看Buffer和切换,省去了minibufexpl插件
    let g:airline#extensions#tabline#enabled = 1
    let g:airline#extensions#tabline#buffer_nr_show = 1

    "设置切换Buffer快捷键"
    nnoremap <F4> :bn<CR>
    " 关闭状态显示空白符号计数
    let g:airline#extensions#whitespace#enabled = 0
    let g:airline#extensions#whitespace#symbol = '!'
    " 设置consolas字体"前面已经设置过
    "set guifont=Consolas\ for\ Powerline\ FixedD:h11
    if !exists('g:airline_symbols')
    let g:airline_symbols = {}
    endif
    " old vim-powerline symbols
    let g:airline_left_sep = '⮀'
    let g:airline_left_alt_sep = '⮁'
    let g:airline_right_sep = '⮂'
    let g:airline_right_alt_sep = '⮃'
    let g:airline_symbols.branch = '⭠'
    let g:airline_symbols.readonly = '⭤'

第二步：要安装字体，如果没有安装字体的话，vim-airline的效果就没法正确的显示

字体安装GitHub地址：<https://github.com/powerline/fonts>

在终端上一步步输入下面的内容即可：

```
# clone
git clone https://github.com/powerline/fonts --depth=1
# install
cd fonts
./install.sh
# clean-up a bit
cd ..
rm -rf fonts
```


# 参考：
1. [NERDTree 快捷键辑录](http://yang3wei.github.io/blog/2013/01/29/nerdtree-kuai-jie-jian-ji-lu/)
2. [NERDTree插件（vim笔记三）](https://www.jianshu.com/p/eXMxGx)