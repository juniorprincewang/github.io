---
title: Gitlab的搭建
date: 2019-01-16 08:40:38
tags:
- gitlab
categories:
- [git]
---


Gitlab 是 Git 服务端的集成管理平台，它拥有与Github类似的功能。 本篇博客记录如何在 Ubuntu 16.04 上搭建 Gitlab 服务。  

<!-- more -->

GitLab是一个利用Ruby on Rails开发的开源应用程序，实现一个自托管的Git项目仓库，可通过Web界面进行访问公开的或者私人项目。  

它提供的功能包括：

+ 代码托管服务
+ 访问权限控制
+ 问题跟踪，bug的记录、跟踪和讨论
+ Wiki，项目中一些相关的说明和文档
+ 代码审查，可以查看、评论代码


# 安装

[Gitlab官网](https://about.gitlab.com/install/#ubuntu) 给出了Ubuntu的安装过程。

+ 安装依赖项  

```
sudo apt-get update
sudo apt-get install -y curl openssh-server ca-certificates
```

安装邮件系统的软件，这个后面邮件通知需要使用到  

```
sudo apt-get install -y postfix
```
+ 安装Gitlab安装包

```
curl https://packages.gitlab.com/install/repositories/gitlab/gitlab-ee/script.deb.sh | sudo bash
sudo EXTERNAL_URL="http://gitlab.example.com" apt-get install gitlab-ee
```

这里的 `http://gitlab.example.com` 可以替换成自己本机的IP地址，比如 `http://192.168.1.123`

**网上大部分的教程，安装的都是gitlab-ce，官网给出的是gitlab-ee，要分清楚！**  

`gitlab-ee` 的安装包很大，最好提前将镜像设置为国内清华等的。

# 启动服务


需要提前开启 sshd 服务和 postfix 服务。默认开启了。

```
service sshd start
service postfix start
```

+ 配置并启动Gitlab

```
sudo gitlab-ctl reconfigure
```

检查Gitlab启动状态

```
sudo gitlab-ctl status
```

+ 网页访问  

启动成功后，在浏览器输入 `http://192.168.1.123` 即可访问。  

Gitlab网页首次访问会引导用户输入 `root` 账户的密码，输入完成后跳转到登录页面，再次输入 用户 `root` 和刚刚配置的密码即可。


|常用命令 	|	说明		|
|-----------|-----------|
|sudo `gitlab-ctl reconfigure`	|	重新加载配置，每次修改 */etc/gitlab/gitlab.rb* 文件之后执行|
|sudo `gitlab-ctl status`	|	查看 GitLab 状态|
|sudo `gitlab-ctl start`	|	启动 GitLab|
|sudo `gitlab-ctl stop`	|	停止 GitLab|
|sudo `gitlab-ctl restart`	|	重启 GitLab|
|sudo `gitlab-ctl tail`	|	查看所有日志|
|sudo `gitlab-ctl tail nginx/gitlab_acces.log`	|	查看 nginx 访问日志|
|sudo `gitlab-ctl tail postgresql`	|	查看 postgresql 日志|

## 遇到的问题

# 80端口被占用

+ 更改配置文件 */etc/gitlab/gitlab.rb*

更改的地方
```
externa_url 'http://192.168.1.123:82'
...
nginx['listen_port'] = 82
nginx['listen_address']=['*', '[::]']
```
这里的端口选个数字小的，常用的，我当时设置了个6666，结果浏览器不认这个端口，瞎忙活半天。

重新 `sudo gitlab-ctl reconfigure` ，然后在浏览器输入 *http://192.168.1.123:82* 。 

# 8080端口被占用

> 网页返回 502 Error  

原因是 Gitlab 使用的 unicorn 服务使用的是8080 端口，被占用了。

修改 `/etc/gitlab/gitlab.rb` 中

```
unicorn['port'] = 9090
```


# 参考
[Ubuntu 16.04 x64搭建GitLab服务器操作笔记](https://www.zybuluo.com/lovemiffy/note/418758)
[GitLab Installation](https://about.gitlab.com/install/#ubuntu)
[我所遇到的GitLab 502问题的解决](https://blog.csdn.net/wangxicoding/article/details/43738137)