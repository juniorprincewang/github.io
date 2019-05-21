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


