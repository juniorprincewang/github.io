---
title: Gitlab的邮箱配置和使用
date: 2019-01-16 08:56:55
tags:
- gitlab
categories:
- [git]
---


本篇博客记录如何使用 Gitlab 服务和邮件配置。  

<!-- more -->

搭建好Gitlab后，创建仓库操作自不必说。只是在添加协作人员时候要邮件邀请，这里需要着重叙述以下。

# 配置邮箱

# 配置163 邮箱  

我这里使用的163邮箱，163邮箱SMTP地址：

|服务器名称 |	服务器地址 |	SSL协议端口号 | 非SSL协议端口号 |
|----------|--------------|---------------|---------------|
|SMTP|		smtp.163.com|	465/994|		25|

需要修改Gitlab配置文件：
配置gitlab的邮箱的时候，可以使用使用ssl，也可以不使用，这里我开始使用了ssl，后来报错，就没再使用。

```
#启用SMTP，邮件发送服务器必开
gitlab_rails['smtp_enable'] = true			
#发件人地址
gitlab_rails['smtp_address'] = "smtp.163.com"
#启用的端口
gitlab_rails['smtp_port'] = 25 
#发件人账号
gitlab_rails['smtp_user_name'] = "xxuser@163.com"
#用户登录密码
gitlab_rails['smtp_password'] = "xxpassword"
#SMTP 服务器主域
gitlab_rails['smtp_domain'] = "163.com"
#验证方式，登录
gitlab_rails['smtp_authentication'] = "login"
gitlab_rails['smtp_enable_starttls_auto'] = true

##修改gitlab配置的发信人

#gitlab 默认的email 用户，必须和SMTP定义的邮件地址一致
user["git_user_email"] = "xxuser@163.com"
#启用邮件
gitlab_rails['gitlab_email_enabled'] = true
# 必须和SMTP定义的邮件地址一致
gitlab_rails['gitlab_email_from'] = 'xxuser@163.com'
gitlab_rails['gitlab_email_display_name'] = 'Gitlab something'
```

## 重新配置服务  

```
gitlab-ctl reconfigure
```
若出现错误，再执行一遍此命令。

## 测试发送邮件

开启控制台  

```
gitlab-rails console
```
进入控制台，然后发送邮件

```
Notify.test_email('717350389@qq.com', '邮件标题', '邮件正文').deliver_now
```

# 遇到问题

+ Can not set relay address for email

> Gitlab: Can not set relay address for email (state=SSLv2/v3 read server hello A: unknown protocol")

需要开启 `smtp_tls` 选项。即：  

```
gitlab_rails['smtp_tls'] = false
```

+ Net::SMTPAuthenticationError: 535 Error: authentication failed

> Net::SMTPAuthenticationError: 535 Error: authentication failed

由于 163邮箱账号客户端登录时候启用的是授权码，不是账户密码，因此需要修改

```
gitlab_rails['smtp_password'] = "authorization code"
```

**这里不能是163邮箱登录密码必须是163客户端的授权密码**


# 参考  
[GitLab Docs:Project's members](https://docs.gitlab.com/ee/user/project/members/)
[GitLab 配置通过 smtp.163.com 发送邮件](https://ruby-china.org/topics/20450)
[gitlab 学习之004：调用第三方邮件接口发送通知邮件](https://www.linuser.com/forum.php?mod=viewthread&tid=369)
[如何开启客户端授权码？](https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b374173cfe9171305fa1ce630d7f67ac2cda80145a1742516)
[163免费邮客户端设置的POP3、SMTP、IMAP地址](http://help.163.com/09/1223/14/5R7P3QI100753VB8.html)
[Git忽略文件.gitignore的使用](https://www.jianshu.com/p/a09a9b40ad20)
[Gitlab之邮箱配置-yellowocng
](https://blog.csdn.net/yelllowcong/article/details/79939589 )