---
title: Git命令
date: 2019-05-13 14:58:10
tags:
- git
categories:
- [git]
---

总结经常使用到的git命令。
<!-- more -->

# init repository

在将写好的工程上传到远端git服务器时，需要以下几个步骤：

初始化本地repository。  
```
git init
```

对本地repository进行git管理。  
```
git add *
git commit -m "some info"
```

关联远端repository。
```
git remote add origin https://github.com/yourgithubID/gitRepo.git 
```

这个 *https://github.com/yourgithubID/gitRepo.git * 必须存在，需要提前到github上建好。当然这里不限于github，其他git服务器也可以。  

将本地仓库push到远程仓库  
```
git push -u origin master 
```

由于在建立github repository时，在master分支创建了README.md，在push时候出现错误。  
可以通过如下命令进行代码合并【注：`pull=fetch+merge`]  
```
git pull --rebase origin master
```

执行上面代码后可以看到本地代码库中多了 *README.md* 文件。
再执行语句 `git push -u origin master` 即可完成代码上传到github。

[github新建本地仓库并将代码提交到远程仓库](https://blog.csdn.net/u010412719/article/details/72860193)
[如何解决failed to push some refs to git](https://www.jianshu.com/p/835e0a48c825)  


# clone 

`--depth <depth>` 可以加速repository的下载速度，指定commit history中的depth条记录。

比如，下条只取最近一次commit的history。

```
git clone --depth 1 git://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
```

+ [Download a specific tag with Git](https://stackoverflow.com/a/31666461)  
下载指定分支或者tag：  
```
git clone -b 'v2.0' --single-branch --depth 1 https://github.com/git/git.git
```

# branch

+ list branches

```
git branch -a
```

+ checkout a specific branch

```
git checkout <branch_name>
```

+ Create the branch on your local machine and switch in this branch  
```
git checkout -b [name_of_your_new_branch]
```

+ Push the branch on github  
```
git push origin [name_of_your_new_branch]
```

或者首次使用 `-u` 选项，之后仅仅使用`git push`即可。
```
git push -u origin <branch>
```
+ [How do I push a new local branch to a remote Git repository and track it too?]()

+ Delete a branch on your local filesystem  
```
git branch -d [name_of_your_new_branch]
```

+ To force the deletion of local branch on your filesystem   
```
git branch -D [name_of_your_new_branch]
```

+ Delete the branch on github  
```
git push origin :[name_of_your_new_branch]
```

[Create a new branch with git and manage branches](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches)  


# tag

git可以对某一时间点的版本[打标签tag](https://git-scm.com/book/zh/v1/Git-%E5%9F%BA%E7%A1%80-%E6%89%93%E6%A0%87%E7%AD%BE) 。  

+ 列出所有tag

```sh
git tag  
git tag -l
git tag -l 'v3.14.*'
```

+ checkout a specific tag

```sh
git checkout tags/<tag_name>
```

[Download a specific tag with Git](https://stackoverflow.com/a/792027)

+ add tag  

```sh
git tag -a v2.1.0 -m "xyz feature is released in this tag."
git tag v2.0.0
```

push tag  

just push particular tag  
```sh
git push origin v1.0.3
```

push all tags:

```sh
git push --tags
```
[Create a tag in a GitHub repository](https://stackoverflow.com/a/48534947)  

+ delete tag  

```sh
// delete a remote tag
git push --delete origin v1.0.0
// also delete the local tag
git tag --delete tagname
```

[How to delete a remote tag?](https://stackoverflow.com/a/5480292)  

# 版本管理

+ [5.2 代码回滚：Reset、Checkout、Revert 的选择](https://github.com/geeeeeeeeek/git-recipes/wiki/5.2-%E4%BB%A3%E7%A0%81%E5%9B%9E%E6%BB%9A%EF%BC%9AReset%E3%80%81Checkout%E3%80%81Revert-%E7%9A%84%E9%80%89%E6%8B%A9)  

## git reset

版本回滚，向前回滚两个版本，可以用 *HEAD~2*，也可以使用7位commit id号。  
```
git reset HEAD~2
git reset 1234567
```

末端的两个提交现在变成了悬挂提交。也就是说，下次 Git 执行垃圾回收的时候，这两个提交会被删除。  
当需要跳转回时，查看消失的两个commit可以使用`git reflog`命令。  

```
git reflog
```


## Revert

顾名思义，版本还原，即撤销一个提交的同时会创建一个新的提交。  
还原倒数第二个commit。  
```
git revert HEAD~2
```

`revert` 和 `reset` 区别是`revert`会改变提交历史，而`reset`不会。  

# undo all changes

+ unstage all files you might have staged with `git add`   

```
git reset
```

+ revert uncommitted changes

```
git checkout .
# or
git reset --hard HEAD
```

+ remove all local untracked files

preview files to be deleted!  

```
git clean -n
```

then remove all untracked files.

```
git clean -fdx
```

[git undo all uncommitted or unsaved changes](https://stackoverflow.com/a/14075772)

## ignoring changes to a single already versioned file

本次提交时忽略某一文件。

```
git add -u
git reset -- main/dontcheckmein.txt
```

[Add all files to a commit except a single file](https://stackoverflow.com/a/4475506)

## completely ignore a specific single file preventing it from being created at repository

完全忽略某一文件：
在仓库根目录创建 *.gitignore* 文件，并将要忽略的文件相对路径写入。

## *.gitkeep*  

git无法追踪一个空的文件夹，当用户需要追踪(track)一个空的文件夹的时候，按照惯例，大家会把一个称为.gitkeep的文件放在这些文件夹里。  

主要用在：使git忽略一个文件夹下的所有文件，并保留该文件夹。  

```
# .gitignore 

# ignore all files in lib/
lib/*
# except for .gitkeep
!.gitkeep
# ignore TODO file in root directory,not subdir/TODO
/TODO
```

当.gitignore采用上面的写法时，git会忽略lib文件夹下除了.gitkeep外的所有文件。  

+[.gitkeep说明](https://www.jianshu.com/p/2507dfdb35d8)  

## gitignore without binary files

+ [ignore binary file in git](https://stackoverflow.com/a/7834886)

```
find . -executable -type f >>.gitignore
```

## gitignore does not work

+ [Why doesn't Git ignore my specified file?](https://stackoverflow.com/a/3833675)  
+ [Gitignore not working](https://stackoverflow.com/a/25436481)  

*.gitignore* 忽略的是没有加入到仓库中的文件，如果已经添加到仓库，需要先删除。  
这条命令会将文件从仓库中删除，但不会物理删除。然后提交更改即可。  
```
git rm --cached files
```

或者简单粗暴全部删除再提交。  
```
git rm -rf --cached .
git add .
```

## git checkout error

git切换分支时出错：

> error: The following untracked working tree files would be overwritten by checkout

这是由于一些untracked working tree files引起的问题。解决方法：  
```
git clean -d -fx
```

`git clean` 参数:  
+ `-n` 显示将要删除的文件和目录；
+ `-x` 删除忽略文件已经对git来说不识别的文件
+ `-d` 删除未被添加到git的路径中的文件
+ `-f` 强制运行


# github操作

为了方便管理repository，免去每次push时候输入账户名和密码，接下来我们需要生成SSH公私钥对，并将公钥上传到 project->Settings->Deploy keys->Add deploy key。
首先[检查是否已经存在 SSH key](https://help.github.com/en/articles/checking-for-existing-ssh-keys) 。  

```
ls -al ~/.ssh
```
出现类似

    id_dsa.pub
    id_ecdsa.pub
    id_ed25519.pub
    id_rsa.pub

如果想继续使用上述密钥，可直接将公钥拷贝出来；否则，要继续自行生成。


[Generating a new SSH key and adding it to the ssh-agent](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)

```
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

如果想要保存不一样的位置或者名称，在下面的提示信息中输入即可。  
> Enter a file in which to save the key (/home/you/.ssh/id_rsa): [Press enter]

将 SSH key 添加到 ssh-agent 中。 

```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

执行ssh-add时出现 *Could not open a connection to your authentication agent* 。
可能当前SHELL不支持，退出当前SHELL即可。
```
ssh-agent bash
```
再次执行
```
ssh-add ~/.ssh/id_rsa
ssh-add -l
```


之后将 *~/.ssh/id_rsa.pub* 内容拷贝到github的project->Settings->Deploy keys->Add deploy key。

这时执行 `push` 命令可能还会出现输入密码的问题，或者说没有权限。  
查看*.git/config*文件查看传输协议, 如果是https模式改为ssh模式即可。  

[Github Deploy Key 操作如何避免输密码](https://www.jianshu.com/p/8d0fae451745)  

## github如何向开源项目提交pr

1. `fork` 到自己的仓库  
2. git clone 到本地  
3. 上游建立连接  
`git remote add upstream 开源项目地址`
4. 创建开发分支 (非必须)  
`git checkout -b dev`
5. 修改提交代码  
```
git status 
git add . 
git commit -m "message"
git push origin branch
```
6. 同步代码三部曲  
```
git fetch upstream 
git rebase upstream/master 
git push origin master
```
7. 提交pr  
去自己github仓库对应fork的项目下new pull request

+ [github如何向开源项目提交pr ](https://github.com/gnipbao/iblog/issues/19)  

## git fork后同步 upstream repository  

分两个步骤，
1. 给fork配置远程仓库

查看远程仓库。  
```sh
git remote -v
```

指定一个可以同步的远程upstream 仓库。  
```sh
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git
```

可以再次验证新的远程的upstream 仓库。  
```sh
git remote -v
```

2. 同步fork  

从上游仓库 fetch 分支和提交点，所有上游分支并会被存储在本地分支，比如 master分支的提交会存储在 upstream/master。  
```sh
git fetch upstream
```

切换到本地分支(比如master)
```sh
git checkout master
```

将 upstream/master分支的更新merge到本地master分支。  
```sh
git merge upstream/master
```

提交更新
```sh
git push origin master
```


[gitlab或github下fork后如何同步源的新更新内容？](https://www.zhihu.com/question/28676261)  
[Configuring a remote for a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork)  
[Sync a fork of a repository to keep it up-to-date with the upstream repository.](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)   