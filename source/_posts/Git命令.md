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

```
git tag -l
git tag -l 'v3.14.*'
```

+ checkout a specific tag

```
git checkout tags/<tag_name>
```

[Download a specific tag with Git](https://stackoverflow.com/a/792027)

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
