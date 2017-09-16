---
title: mysql知识点拾遗
date: 2017-08-09 09:32:45
tags:
- mysql
- sql
---

本篇介绍mysql以及sql知识点的补充和拾遗。

<!-- more -->

## 时间戳和字符串转换

MySQL将unix时间戳转化成时间字符串
```
select FROM_UNIXTIME(1349664611,'%Y-%m-%d %H:%i:%s')
```
mysql将时间字符串转化为unix时间戳
```
select UNIX_TIMESTAMP('2012-10-25 12:00:12')
```