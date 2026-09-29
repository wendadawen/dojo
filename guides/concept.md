# 概念页面

学习中遇到一个概念，把它写成一篇讲解页面。概念可以来自任何领域。读者没有这个领域的背景，页面从问题讲起：为什么需要它，它是什么，怎么起作用，什么时候不成立。每个概念生成两个页面：完整说明 `index.html` 和概览 `overview.html`。完整说明比概览细，细节收在折叠块里，写法见[正文文件](concept/write.md)的「正文与折叠块」一节。

用户点名要了解某个概念时，按「步骤」从第 1 步做到交付。

## 步骤

1. 收集资料：原始文献、标准和权威教材。讲具体实现时，再加上固定版本的官方文档或源码。写进 `wiki/<name>/research/concept.md` 的「资料」一节。教程和博客只用来找线索，不作为依据。
2. 定范围：讲什么、不讲什么，写进「范围」。名字有歧义时（同名缩写、不同领域的同名概念），在这里定下用哪个含义。
3. 查前置概念：读懂这个概念之前要先懂的概念，逐个看 `wiki/` 下有没有页面。有就在正文第一次用到时链接过去；没有就先按本指南把它写出来，需要几层写几层。
4. 写 `concept.md`：格式和写法见[正文文件](concept/write.md)，示范见 `wiki/engram/`。写完运行 `python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md --run`，生成两个页面。
5. 审查 `concept.md`：每轮按[审查](concept/review.md)派审查者，按那份文件判断这一轮是否通过。
6. 检查页面：运行第 4 步的生成命令，再运行 `python3 .dojo/scripts/ci-01-validate.py wiki/<name>/index.html wiki/<name>/overview.html`。然后在浏览器里打开两个页面，看公式能不能正常显示，图有没有错位，折叠块能不能打开。
7. 修改：页面内容的问题改 `concept.md` 后重新生成。改完按第 6 步检查页面，再按[审查](concept/review.md)判断要不要派下一轮。

## 交付

- `wiki/<name>/index.html` 和 `overview.html`。
- `wiki/<name>/research/`：`concept.md` 和每轮的审查记录。
- 回复里只写交付说明：两个页面的路径、主要资料、审查了几轮、新写的前置概念页、已知不足。
