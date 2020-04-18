---
title: Hexo Next主题 解决公式块中公式换行问题
date: {{ date }}
categories:
 - Hexo
tags:
 - Hexo
 - Next
toc: true
mathjax: true
---



### 问题描述

公式块内公式不能换行。

<!--more-->

### 问题环境

+ 主题：[NexT 7.8.0 Released](https://theme-next.org/next-7-8-0-released/)

+ hexo：

    ```
    hexo-cli: 3.1.0
    os: Windows_NT 6.1.7601 win32 x64
    node: 12.14.0
    v8: 7.7.299.13-node.16
    uv: 1.33.1
    zlib: 1.2.11
    brotli: 1.0.7
    ares: 1.15.0
    modules: 72
    nghttp2: 1.39.2
    napi: 5
    llhttp: 1.1.4
    http_parser: 2.8.0
    openssl: 1.1.1d
    cldr: 35.1
    icu: 64.2
    tz: 2019c
    unicode: 12.1
    ```


### 原因定位

Next 主题默认使用 `//cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js`  进行公式渲染，但是 mathjax_v3 并不支持使用 ` \\` 进行换行。

> dpvc：
>
> Linebreaks are not yet implemented in version 3.  This is one of the  significant features from version 2 that been ported to version 3.
>
> https://github.com/mathjax/MathJax/issues/2312

### 解决方案

#### 更换 mathjax 版本

mathjax 支持公式块内公式换行，所以直接修改镜像

在`\themes\next\_config.yml` 中的 `vendors` 结点下修改

```
mathjax: https://cdn.jsdelivr.net/npm/mathjax@2.7.8/unpacked/MathJax.js?config=TeX-MML-AM_CHTML
```

更改 mathjax 版本后，发现原来行内公式不渲染了，原因是 v3 版本和 v2 版本对行内公式渲染的设置方式发生了变化。

#### 解决行间公式不渲染问题

进入`\themes\next\layout\_third-party\math\mathjax.swig`  进行修改

```html
{%- set mathjax_uri = theme.vendors.mathjax or '//cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' %}

<script>
  if (typeof MathJax === 'undefined') {
    // window.MathJax = {
    //   loader: {
    //     {%- if theme.math.mathjax.mhchem %}
    //       load: ['[tex]/mhchem'],
    //     {%- endif %}
    //     source: {
    //       '[tex]/amsCd': '[tex]/amscd',
    //       '[tex]/AMScd': '[tex]/amscd'
    //     }
    //   },
    //   tex: {
    //     inlineMath: {'[+]': [['$', '$']]},
    //     {%- if theme.math.mathjax.mhchem %}
    //       packages: {'[+]': ['mhchem']},
    //     {%- endif %}
    //     tags: 'ams'
    //   },
    //   options: {
    //     renderActions: {
    //       findScript: [10, doc => {
    //         document.querySelectorAll('script[type^="math/tex"]').forEach(node => {
    //           const display = !!node.type.match(/; *mode=display/);
    //           const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
    //           const text = document.createTextNode('');
    //           node.parentNode.replaceChild(text, node);
    //           math.start = {node: text, delim: '', n: 0};
    //           math.end = {node: text, delim: '', n: 0};
    //           doc.math.push(math);
    //         });
    //       }, '', false],
    //       insertedScript: [200, () => {
    //         document.querySelectorAll('mjx-container').forEach(node => {
    //           let target = node.parentNode;
    //           if (target.nodeName.toLowerCase() === 'li') {
    //             target.parentNode.classList.add('has-jax');
    //           }
    //         });
    //       }, '', false]
    //     }
    //   }
    // };
    window.MathJax = {
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        processEscapes: true
      }
    };
    (function () {
      var script = document.createElement('script');
      script.src = '{{ mathjax_uri }}';
      script.defer = true;
      document.head.appendChild(script);
    })();
  } else {
    MathJax.startup.document.state(0);
    MathJax.texReset();
    MathJax.typeset();
  }
</script>
```

v2 版本配置行内公式渲染的办法变为了

```javascript
	window.MathJax = {
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        processEscapes: true
      }
    };
```

