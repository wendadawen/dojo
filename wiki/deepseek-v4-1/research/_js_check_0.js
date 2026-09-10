
// ============ 进度条 + 返回顶部 ============
const progressBar = document.getElementById('progress');
const backToTop = document.getElementById('backToTop');
window.addEventListener('scroll', () => {
  const h = document.documentElement;
  const scrolled = (h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100;
  progressBar.style.width = scrolled + '%';
  backToTop.classList.toggle('visible', h.scrollTop > 400);
});
backToTop.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ============ 暗/亮模式切换（带 localStorage 持久化 + highlight.js 主题切换）============
const themeToggle = document.getElementById('themeToggle');
const savedTheme = localStorage.getItem('theme') ||
  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.setAttribute('data-theme', savedTheme);
function syncCodeTheme(theme) {
  const light = document.getElementById('prism-light');
  const dark = document.getElementById('prism-dark');
  if (light && dark) {
    light.disabled = (theme === 'dark');
    dark.disabled = (theme !== 'dark');
  }
}
syncCodeTheme(savedTheme);
themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  syncCodeTheme(next);
});

// ============ 自动生成侧边目录 ============
const tocList = document.getElementById('tocList');
const headings = document.querySelectorAll('body > h2, body > h3');
let currentH1Li = null;
let currentSubUl = null;
headings.forEach(h => {
  if (!h.id) {
    h.id = h.textContent.trim().replace(/[\s#?？：]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
  }
  const li = document.createElement('li');
  const a = document.createElement('a');
  a.href = '#' + h.id;
  a.textContent = h.textContent.replace(/^Q\d+[：:]\s*/, '').slice(0, 30);
  a.dataset.target = h.id;
  if (h.tagName === 'H3' && currentH1Li) {
    a.classList.add('sub');
    if (!currentSubUl) {
      currentSubUl = document.createElement('ul');
      currentH1Li.appendChild(currentSubUl);
    }
    li.appendChild(a);
    currentSubUl.appendChild(li);
  } else {
    currentSubUl = null;
    li.appendChild(a);
    tocList.appendChild(li);
    currentH1Li = li;
  }
});

// ============ 滚动高亮当前章节 ============
const sections = Array.from(document.querySelectorAll('body > h2, body > h3'));
function updateActiveToc() {
  let active = null;
  for (const s of sections) {
    if (s.getBoundingClientRect().top < 120) active = s.id;
  }
  document.querySelectorAll('.toc a').forEach(a => {
    a.classList.toggle('active', a.dataset.target === active);
  });
}
window.addEventListener('scroll', updateActiveToc, { passive: true });
updateActiveToc();

// ============ 阅读时间估计 ============
const bodyText = document.body.innerText || '';
const chineseChars = (bodyText.match(/[\u4e00-\u9fff]/g) || []).length;
const englishWords = (bodyText.match(/[a-zA-Z]+/g) || []).length;
// 中文 350 字/分钟，英文 200 词/分钟
const minutes = Math.max(1, Math.ceil(chineseChars / 350 + englishWords / 200));
const totalChars = chineseChars + englishWords;
document.getElementById('readingTime').textContent =
  `全文 ${totalChars.toLocaleString('zh-CN')} 字 · 约 ${minutes} 分钟阅读`;

// ============ 图片点击放大（lightbox）============
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxCaption = document.getElementById('lightboxCaption');

// 给所有文章里的 img 加可点击 class
document.querySelectorAll('body > img, p ~ img, .grid img').forEach(img => {
  if (img.id === 'lightboxImg') return;
  img.classList.add('article-image');
  img.addEventListener('click', () => {
    lightboxImg.src = img.src;
    lightboxCaption.textContent = img.alt || '';
    lightbox.classList.add('open');
  });
});

lightbox.addEventListener('click', () => {
  lightbox.classList.remove('open');
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') lightbox.classList.remove('open');
});

// ============ 代码块复制按钮 ============
document.querySelectorAll('.code-block pre').forEach(pre => {
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.textContent = '复制';
  btn.addEventListener('click', async () => {
    const code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
    try {
      await navigator.clipboard.writeText(code);
      btn.textContent = '已复制 ✓';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = '复制';
        btn.classList.remove('copied');
      }, 1500);
    } catch (err) {
      // 降级方案
      const textarea = document.createElement('textarea');
      textarea.value = code;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      btn.textContent = '已复制 ✓';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = '复制';
        btn.classList.remove('copied');
      }, 1500);
    }
  });
  pre.parentElement.appendChild(btn);
});

// ============ 章节折叠按钮 ============
document.querySelectorAll('body > h2').forEach(h1 => {
  const btn = document.createElement('span');
  btn.className = 'collapse-btn';
  btn.textContent = '▼';
  btn.title = '折叠/展开';
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const collapsed = h1.classList.toggle('collapsed');
    let next = h1.nextElementSibling;
    while (next && next.tagName !== 'H2') {
      next.style.display = collapsed ? 'none' : '';
      next = next.nextElementSibling;
    }
  });
  h1.appendChild(btn);
});

// ============ 键盘快捷键：j/k 跳章节 ============
const h1Sections = Array.from(document.querySelectorAll('body > h2'));
document.addEventListener('keydown', (e) => {
  // 输入框聚焦时不响应
  if (e.target.matches('input, textarea')) return;

  if (e.key === 'j' || e.key === 'ArrowDown' && e.shiftKey) {
    e.preventDefault();
    let next = h1Sections.find(s => s.getBoundingClientRect().top > 120);
    if (next) next.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else if (e.key === 'k' || e.key === 'ArrowUp' && e.shiftKey) {
    e.preventDefault();
    let prev = null;
    for (const s of h1Sections) {
      if (s.getBoundingClientRect().top >= 120) break;
      prev = s;
    }
    if (prev) prev.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});

// ============ 平滑滚动（点击目录链接）============
document.querySelectorAll('.toc a').forEach(a => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.getElementById(a.dataset.target);
    // scroll-margin-top 已避开顶部导航；不要再 scrollBy 偏移，否则会打断平滑滚动
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});
