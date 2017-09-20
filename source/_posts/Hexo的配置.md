---
title: Hexo的搭建
date: 2017-08-07 20:44:39
tags:
---

本篇博客记录了如何对搭建好的Hexo主题博客进行设置，包括主题，布局，风格等等。

<!-- more -->


## 发布到github

今天电脑重装系统，我决定将这个hexo文件夹整个拷贝到新电脑，配置好本地环境后，再部署的时候遇到了问题。

    hexo Error: Host key verification failed. 
    fatal: Could not read from remote repository.

很明显没有在本地配置github的环境和账户。接下来步骤可参考<https://help.github.com/articles/connecting-to-github-with-ssh/>
所以打开`Git Bash`，检查是否已经存在了SSH keys。
```
# 查看.ssh目录下是否已经有私钥和公钥
ls -al ~/.ssh
# 如果有，删除。并重新生成
ssh-keygen -t rsa -C "your_github_email"
ssh-agent -s
ssh-add ~/.ssh/id_rsa
# 这里会出错，如果出错了输入下面的指令。
# eval `ssh-agent -s`
# ssh-add
# 拷贝公钥
clip < ~/.ssh/id_rsa.pub

```

然后到github.com个人中心的settings的SSH keys页面，点击`New SSH Key`添加公钥。
添加完毕后，测试是否连接成功。
```
ssh -T git@github.com
```
一般在添加完公钥之后, 很多人经常会遇到下面这种问题：
    
    **Host key verification failed.**
    fatal: Could not read from remote repository.

这是由于.ssh文件夹下缺少了`known_hosts`文件。输入以下命令即可。
```
ssh git@github.com
```

最后输入指令部署即可。
```
hexo d -g
```

## 添加访问统计

我喜欢在自己的布局文件的侧栏添加访问量。
路径为：`/themes/jacman/layout/_partial/sidebar.ejs`.
添加本站访问人数和访问人次总数。
```
<div class="linkslist">
<span id="busuanzi_container_site_uv"> 
  本站访客数<span id="busuanzi_value_site_uv"></span>人次
</span>
</div>
<div class="linkslist">
<span id="busuanzi_container_site_pv">
    本站总访问量<span id="busuanzi_value_site_pv"></span>次
</span>
</div>
```

## `hexo s`命令失效

我将hexo整个文件夹上传到github服务器，在另一台电脑下载下来，运行`hexo s`失效，从显示结果来看，不存在此命令。
这是由于Hexo3将`hexo-server`独立出来了，如果需要本地调试，需要先安装server。
```
npm install hexo-server --save
npm install 
```
重新生成静态文档，启动本地服务。
问题又出现了，网页空白，并且命令行窗口显示：
    
    WARN  No layout: index.html

这种情况的出现是由于主题配置出错。查看`themes`文件夹下是否有相关文件夹并且文件夹下是否有文件；`_config.yml`内`theme`是否和主题名称对应。

# 参考
[1] [Hexo｜快速搭建自己（Github）博客](http://www.jianshu.com/p/808554f12929)
[2] [给Hexo博客添加访问统计](http://www.jianshu.com/p/8a8f880f40c0)
[3] [搭建hexo部署到github图文教程 亲测可用超详细](https://m.paopaoche.net/new/85988)
[4] [用Hexo 3 搭建github blog](http://forsweet.github.io/hexo/%E7%94%A8Hexo%E6%90%AD%E5%BB%BAGithub%E5%8D%9A%E5%AE%A2/)