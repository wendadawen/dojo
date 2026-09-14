# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-6.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**6** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**16** 条
- 有理由不改：**0** 条

## 修复与判定说明

通读 research/ 下 6 轮 review（review-1..6），汇总 22 条轻微问题：6 条本轮修掉，16 条判定为「已不存在」（历轮已修）。实际修复（均已复验）：(1) vLLM 首现未解释 → index L181「vLLM（开源 LLM 推理框架）」、overview L52 同步；(2) HBM 未展开 → index L141「HBM（高带宽显存）」（用词对齐 gpu-execution-model 前置页）；(3) SM 未展开 → index L141「SM 即流式多处理器」；(4) β 记号与 Leviathan 原文相反（本页 β=拒绝概率、原文 β=接受率）→ 在 [F3] 补注「互为补，对照原文时勿混淆」；(5) 无来源术语「superstep」→ 删去引号括注，改为「每轮做五件事」；(6) SVG 图内 div 缺 class="dg-label" → 7 处 foreignObject div 按 style-guide §11 补 class（页面自带 .flow-svg foreignObject div 样式继续生效，渲染不变）。未改无关内容，未动任何数字与引文编号（正文/summary/overview/图注的 2×-3×、2-2.5×、1.4×-1.8×、3.07×、c=0.04、γ≈9 等保持一致）。链接核查：页面引用的 gpu-execution-model / standard-attention / eagle-speculative / kimi-k3 / index 均存在，无指向 research/ 或不存在文件的路径；alt/aria 无 $...$。
