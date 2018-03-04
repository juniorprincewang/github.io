---
title: '解决su: Authentication failure'
date: 2018-03-01 16:51:46
tags:
- linux
categories:
- solutions
---

Why does su fail with “authentication error”?

<!-- more -->

# 问题


	user@host:~$ su
	Password: 
	su: Authentication failure

# 解决

The root account is disabled by default in Ubuntu, so there is no root password, that's why su fails with an authentication error.

```
sudo -i
```

