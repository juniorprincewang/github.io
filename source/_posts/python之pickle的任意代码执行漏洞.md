---
title: python之pickle的任意代码执行漏洞
date: 2018-02-08 15:30:48
tags:
- python
- pickle
categories:
- python
---

    Warning The pickle module is not secure against erroneous or maliciously constructed data. 
    Never unpickle data received from an untrusted or unauthenticated source.

<!-- more -->

# pickle

python中的pickle模块可以将对象按照一定的格式序列化后保存在磁盘或进行网络传输。
python中pickle的对象序列化和反序列化方法包括：
```
pickle.dump(obj, file[, protocol])
pickle.load(file)
pickle.dumps(obj[, protocol])
pickle.loads(string)
```
其中带`s`的函数操作对象是字符串，而不带`s`的操作对象是文件。


# 参考
[1] [pickle — Python object serialization](https://docs.python.org/2.7/library/pickle.html?highlight=pickle#module-pickle)
[2] [Python Pickle的任意代码执行漏洞实践和Payload构造](http://www.polaris-lab.com/index.php/archives/178/)
[3] [Arbitrary code execution with Python pickles](https://www.cs.uic.edu/~s/musings/pickle/)
[4]