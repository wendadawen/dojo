# Muon 优化器独立审查（第二轮）

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：ae60cb8b276c90c46675cfc1ae780ee6966f1b13
- 时间：2026-08-09

## 问题

- [重要·技术] index.html 概念定位框「前置概念」字段及第 1 章「动量更新矩阵为什么需要正交化」：两处均称 Newton-Schulz 正交化页面为「占位页，待生成」，但 `wiki/newton-schulz/index.html` 文件实际存在（修改时间晚于本页）。应删除「占位页 / 待生成」措辞，将已有的 `<a href="../newton-schulz/index.html">` 链接保留为有效引用。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html「RMS 对齐缩放因子」段及来源说明 [F7]：RMS 对齐缩放因子 $\gamma=0.2\sqrt{\max(m,n)}$ 标注来源为 Moonshot AI《Muon is Scalable for LLM Training》（2025），推导参考苏剑林博客（kexue.fm）。WebSearch 获取的二手来源（newton.com.tw）提及月之暗面 2025 年 2 月改进含「调整参数更新尺度」，但未直接确认 $0.2$ 系数。补充 Moonshot 原文链接或明确标注 $0.2$ 的出处。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html「RMS 对齐缩放因子」段 PyTorch 2.12 默认超参表：列出 `torch.optim.Muon` 的 lr=1e-3、weight_decay=0.1、momentum=0.95 等默认值，标注来源为「PyTorch 2.12 文档」。WebSearch 无法验证 PyTorch 2.12 版本是否存在该 API 及这些默认值。核对 PyTorch 官方文档并补充版本发布链接。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：进入修复
