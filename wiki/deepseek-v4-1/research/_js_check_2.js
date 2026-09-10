
/* 站内文档链接与外部链接在新标签页打开 */
(function () {
  document.querySelectorAll('a[href]').forEach(function (a) {
    var href = a.getAttribute('href') || '';
    if (!href || href.charAt(0) === '#') return;
    if (/^[a-z][a-z0-9+.\-]*:/i.test(href) || /\.html?(?:[?#]|$)/i.test(href)) {
      a.target = '_blank';
      a.rel = 'noopener';
    }
  });
})();
