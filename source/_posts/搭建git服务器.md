---
title: 搭建git服务器
date: 2018-08-14 14:21:38
tags:
- git
catogeries:
- linux
---
本篇博客介绍在ubuntu 16.04中搭建git服务器的过程，操作平台是windows 10。
<!-- more -->

# 安装Gitosis管理用户与项目
Gitosis是一套用来管理 `authorized_keys` 文件和实现简单连接限制的脚本，对项目、用户以及项目的读写权限进行管理，安装命令如下：

```
git clone https://github.com/res0nat0r/gitosis.git && cd gitosis && python setup.py install
```
Gitosis默认使用的Git根目录是 `/home/git/repositories`，其中 `git` 是即将新建的用户。如果你想把仓库放在别的地方，就用软连接将它与 `/home/git/repositories` 连接起来。


# 创建Git管理员账户
新建一个用户作为Git服务器的管理员：
```
useradd -m git
passwd git
```
用管理员公钥初始化Gitosis
这个管理员公钥的意思是你本机的公钥，是用来管理这个Gitosis的（默认会有gitosis-admin的读写权限），你可以将你本机（常用机器）的ssh key拷贝到服务器上来，从而实现管理的目的。

1.在本机生成公钥

windows中的ssh配置路径为 `c:\Users\username\.ssh` ，这里 `username` 为自己操作系统的用户名。
```
ssh-keygen -t rsa -C "youremail@yourcompany.com” -f ~/.ssh/dacas-rsa
```

2.上传公钥至服务器并激活Gitosis
将公钥拷贝到git用户下，因此先切换至git用户：

su git
之后用rz命令直接拷贝值服务即可。（Windows利用lrzsz拷贝文件至Linux）

3.初始化Gitosis
依然在git用户下，利用刚才上传的公钥初始化Gitosis：

gitosis-init < /home/git/id_rsa.pub

> Initialized empty Git repository in /home/git/repositories/gitosis-admin.git/
> Reinitialized existing Git repository in /home/git/repositories/gitosis-admin.git/

这里也可以采用SSH 免密码登录的操作：
创建 `authorized_keys` 文件，如果已经存在这个文件, 跳过这条。
```
touch ~/.ssh/authorized_keys 
```
必须将 `~/.ssh/authorized_keys` 的权限改为600, 该文件用于保存ssh客户端生成的公钥，可以修改服务器的ssh服务端配置文件 `/etc/ssh/sshd_config` 来指定其他文件名。
```
chmod 600 ~/.ssh/authorized_keys
```
将 `id_rsa.pub` 的内容追加到 `authorized_keys` 中, 注意使用追加 `>>` ，不要用 `>` ，否则会清空原有的内容，使其他人无法使用原有的密钥登录。
```
cat /home/git/id_rsa.pub/id_rsa.pub  >> ~/.ssh/authorized_keys 
```


# 在Git服务器新建一个项目
完成上一步之后，你Git服务器已经装好了。相关信息总结如下：

- 默认的仓库地址是在 `/home/git/repositories` 。
- Git管理用户是刚才创建的git。
- Gitosis管理用户权限是通过一个git项目实现的，那个项目地址在 `/home/git/repositories/gitosis-admin.git`，默认是你刚才上传公钥的电脑可以clone此仓库。

新建一个项目就是在默认的仓库地址下面新建一个空的git项目：
```
cd /home/git/repositories
git init --bare test.git
```
如此，便新建了一个test的项目。

# 在本机clone项目
为了方便操作，提前做一点简单配置。在 `c:\users\username\.ssh\config` 中输入以下内容。
```
Host 	dacasserver
Hostname	159.226.94.159
User			git
IdentityFile	~/.ssh/dacas-rsa
```

现在可以试一下用初始化 Gitosis 的公钥的拥有者身份 SSH 登录服务器，应该会看到类似下面这样：
```
$ ssh git@dacasserver
PTY allocation request failed on channel 0
ERROR:gitosis.serve.main:Need SSH_ORIGINAL_COMMAND in environment.
  Connection to gitserver closed.
```
说明 Gitosis 认出了该用户的身份，但由于没有运行任何 Git 命令，所以它切断了连接。那么，现在运行一个实际的 Git 命令 — 克隆 Gitosis 的控制仓库：

在你本地计算机上
```
git clone git@dacasserver:gitosis-admin.git
```
其中，git是你刚才新建的Git管理员，后面跟着的是你的服务器地址。

gitosis-admin的中包括一个keydir文件夹和一个gitosis.conf文件，前者是用来存放用户的ssh key的，后者是用来管理用户权限的，举个例子，我现在要给张三和李四的电脑读写test的权限：

1.将张三电脑和李四电脑的ssh key拷贝至keydir文件夹下

例如将张三的公钥文件保存为zhangsan.pub放在keydir下（这个文件名字与下面配置文件要一致）、李四的公钥文件保存为lisi.pub放在keydir下。

2.在gitosis.conf中添加相关配置

[group test]
members = zhangsan lisi
writable = test
如此，我便新建了一个test的group，其中用户有zhangsan和lisi，他们拥有读写权限。

此时，如果你想让王五只有读的权限，那么就将配置文件改成：

[group test]
members = zhangsan lisi
writable = test
[group test_read]
members = wangwu
readonly = test
如此，wangwu只能clone或者pull，却不能push。

3.将修改推送至服务器

此时只是完成了本地的修改，要将修改推送到服务器才能生效（add、commit、push），之后test那个仓库的权限就会像刚才在配置文件设置的那样。


# 参考
1. [git 配置多个SSH-Key](https://my.oschina.net/stefanzhlg/blog/529403)
2. [配置多个SSH-Key](https://www.jianshu.com/p/7a7b98d05fd8)
3. [Linux Ubuntu搭建Git服务器](https://segmentfault.com/a/1190000015020195)
4. [7 服务器上的 Git - Gitosis](https://git-scm.com/book/zh/v1/%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84-Git-Gitosis)
5. [利用SSH的用户配置文件Config管理SSH会话](https://www.hi-linux.com/posts/14346.html)