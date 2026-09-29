# 概念页面

学习中遇到一个概念，把它写成一篇讲解页面。概念可以来自任何领域。读者没有这个领域的背景，页面从问题讲起：为什么需要它，它是什么，怎么起作用，什么时候不成立。每个概念生成一篇完整说明和一篇概览。

用户点名要了解某个概念时，直接从第 1 步做到交付。不用对话里的讲解代替页面。`wiki/<name>/` 里已有旧页面或旧的 `research/concept.md`，也按本指南在这个目录重写，不复述旧稿。不要先删掉旧目录，也不要问是恢复旧稿还是重写。

## 步骤

1. 收集资料：提出这个概念的原始文献、标准、权威教材，讲具体实现时再加上固定版本的官方文档或源码，写进 `wiki/<name>/research/concept.md` 的「资料」一节。教程和博客只用来找线索，不作为依据。
2. 定范围：讲什么、不讲什么，写进「范围」。名字有歧义时（同名缩写、不同领域的同名概念），在这里定下用哪个含义。
3. 查前置概念：读懂这个概念之前要先懂的概念，逐个看 `wiki/` 下有没有页面。有就在正文第一次用到时链接过去；没有就先按本指南把它写出来，需要几层写几层。
4. 写 `concept.md`：格式和写法（含代码块）见[正文文件](concept/write.md)，示范见 `guides/examples/concept/`。`wiki/<name>/research/` 里已有 `review-*.md` 时先读完。经对照资料确认的事实错误，新稿里不能再出现。写完运行 `python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md --run`，生成两个页面。
5. 审查 `concept.md`：每轮按[审查](concept/review.md)派 20 个独立的 `tclaude -p` 进程。通过条件见同一份文件。
6. 检查页面：重新运行第 4 步的命令，再运行 `python3 .dojo/scripts/ci-01-validate.py wiki/<name>/index.html wiki/<name>/overview.html`，然后在浏览器里打开两个页面，看公式能不能正常显示，图有没有错位，折叠块能不能打开。
7. 修改：页面内容改 `concept.md` 后重新生成，不手改生成出的 html。样式和模板的问题改对应的模板或样式，改完回到第 6 步。什么时候再派审查，见[审查](concept/review.md)。整篇重写时回到第 4 步重写并重新生成，再按第 5 步派一轮。

## 交付

- `wiki/<name>/index.html` 和 `overview.html`。首页元数据写在 `index.html` 的 head 里
- `wiki/<name>/research/`：`concept.md` 和每轮的审查记录
- 回复里写交付说明：生成位置、主要资料、审查了几轮、新写的前置概念页、已知不足
