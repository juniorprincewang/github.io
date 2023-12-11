---
title: Hexo的搭建
date: 2017-08-07 20:44:39
tags:
- hexo
categories:
- [hexo]
---

本篇博客记录了如何简单搭建Hexo，并对搭建好的Hexo主题博客进行设置，包括主题，布局，风格等等。

<!-- more -->

# 搭建

下载NodeJS客户端，我选用的版本是 *8.9.4* 。 当前*9.0*以上的版本和npm存在兼容性的问题，所以我还是用原来的版本吧。

```
node -v
```

> v8.9.4

然后安装 hexo
```
npm install -g hexo-cli
```
完工。

## 发布到github

### Permission denied (publickey).

在使用 `git push`  时候报以下错误。

	Permission denied (publickey).
	fatal: Could not read from remote repository.

	Please make sure you have the correct access rights
	and the repository exists.

原因：电脑公钥（publickey）未添加至github,所以无法识别。 因而需要获取本地电脑公钥，然后登录github账号，添加公钥至github就OK了。  
设置Git的user name和email  

> git config --global user.name "yourname"
> git config --global user.email "youremail" 

生成密钥 
```
生存密钥：
ssh-keygen -t rsa -C "youremail" -f your_file
```

然后再登录github，进入个人设置settings--->ssh and gpg keys-->new ssh key 添加新生成的公钥即可。

### hexo Error: Host key verification failed.


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

## 主题

本博客更换到了 [NexT 主题UI](https://github.com/next-theme/hexo-theme-next)，直接clone下来最新代码到 `themes/next` 目录即可。

```
cd hexo-site
git clone https://github.com/next-theme/hexo-theme-next themes/next
```

修改 Hexo 的配置文件，改成next主题:
```
theme: next
```

查看next的安装包依赖 hexo 本地是否满足，一般都要升级依赖包，见下文。

如果本地预览和部署的文件有差异，猜测是由依赖包不符引起的，需要升级，并且部署前将 `.deploy_git` 文件夹删除。


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


[更多炫酷的主题个性化](http://shenzekun.cn/hexo%E7%9A%84next%E4%B8%BB%E9%A2%98%E4%B8%AA%E6%80%A7%E5%8C%96%E9%85%8D%E7%BD%AE%E6%95%99%E7%A8%8B.html)  

## 文章按照更新时间排序

找到主配置文件 *\_config.yml*，然后修改或者添加 `index_generator` 的 `order_by` 为 `-updated` 即可:

```
index_generator:
  path: ''
  per_page: 10
  order_by: -updated
```

重新生成页面：  
```
hexo g
```

[Hexo 文章按照更新时间排序](http://aiellochan.com/2018/02/13/hexo/Hexo-%E6%96%87%E7%AB%A0%E6%8C%89%E7%85%A7%E6%9B%B4%E6%96%B0%E6%97%B6%E9%97%B4%E6%8E%92%E5%BA%8F/)  

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

## 升级依赖包


```sh
npm i hexo-cli -g
npm update
```

查看更新到的版本。  
```
hexo vesion
```

检查更新。   
```
npm install -g npm-check
npm-check
```
升级。   
```
npm install -g npm-upgrade
npm-upgrade
```

一路回车确认安装即可。  
接下来在本地生产环境安装  
```
npm install --save
```

[Hexo博客及环境依赖包的正确升级方法](https://hexo.imydl.tech/archives/51612.html)  


# 操作

## 启动本地server

有时候本地server启动后会遇到端口被占用情况，可以在 `node_modules\hexo-server\index.js` 文件中设置 或者在启动命令行中加入 `-p <port>` 启动选项。

```js
hexo.config.server = Object.assign({
  port: 4321,
  log: false,
  // `undefined` uses Node's default (try `::` with fallback to `0.0.0.0`)
  ip: undefined,
  compress: false,
  header: true
}, hexo.config.server);

hexo.extend.console.register('server', 'Start the server.', {
  desc: 'Start the server and watch for file changes.',
  options: [
    {name: '-i, --ip', desc: 'Override the default server IP. Bind to all IP address by default.'},
    {name: '-p, --port', desc: 'Override the default port.'},
    {name: '-s, --static', desc: 'Only serve static files.'},
    {name: '-l, --log [format]', desc: 'Enable logger. Override log format.'},
    {name: '-o, --open', desc: 'Immediately open the server url in your default web browser.'}
  ]
}, require('./lib/server'));
```

## 发布新文章

执行 `new` 命令， 生成指定名称的文章至 `hexo\source_posts\postName.md` 。  

```
hexo new [layout] "postName" 
```

其中layout是可选参数，默认值为post。有哪些layout呢，请到scaffolds目录下查看，这些文件名称就是layout名称。  
当然你可以添加自己的layout，方法就是添加一个文件即可，同时你也可以编辑现有的layout，比如post的layout默认是 `hexo\scaffolds\post.md`。  



新文件的开头是属性，采用统一的 `yaml` 格式，用三条短横线分隔。
可以直接修改 *title*（标题名字）、 *date*（时间）等。 *description* 是文章概要，该项为空时hexo默认在首页会显示全部文章内容，如果文章比较长就会显的内容很乱；写让该项后hexo会显示摘要和“阅读全文”连接。下面是文章正文。  


    title: postName #文章页面上的显示名称，可以任意修改，不会出现在URL中
    date: 2013-12-02 15:30:16 #文章生成时间，一般不改，当然也可以任意修改
    categories: #文章分类目录，可以为空，注意:后面有个空格， 用格式
    - [tag1,tag2]
    tags: #文章标签，可空，多标签请用格式[tag1,tag2,tag3]，注意:后面有个空格
    - tag1
    - tag2
    description: #概要信息
    ---
    正文。

## 草稿

草稿默认不会显示在页面上，链接也搜索不到。会在 `source/_drafts` 目录下生成一个 `new-draft.md` 文件。  
```
hexo new draft "new draft"
```

如果预览草稿，可
1. 或者如下方式启动server：`hexo server --drafts`  
2. 更改配置文件（*_config.yml*）`render_drafts: true`，但是deploy前需要求改掉。    

下面这条命令可以把草稿变成文章，或者页面：
```sh
hexo publish [layout] <filename>
```



# 参考
+ [Hexo｜快速搭建自己（Github）博客](http://www.jianshu.com/p/808554f12929)  
+ [给Hexo博客添加访问统计](http://www.jianshu.com/p/8a8f880f40c0)  
+ [搭建hexo部署到github图文教程 亲测可用超详细](https://m.paopaoche.net/new/85988)
+ [用Hexo 3 搭建github blog](http://forsweet.github.io/hexo/%E7%94%A8Hexo%E6%90%AD%E5%BB%BAGithub%E5%8D%9A%E5%AE%A2/)  
+ [hexo使用心得（二）](https://lanjingling.github.io/2015/09/24/use-of-hexo-2/)