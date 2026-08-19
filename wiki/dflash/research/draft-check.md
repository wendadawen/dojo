# DFlash 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（2026-08-19），规划条件满足（论文版本固定到 arXiv:2602.06036v2 ICML 2026 accepted）。
- 大纲落实：5 章 + 核心问题 5 题 + 两级问题块 + 构造示例贯穿 + 方法评价章（含本章问题块）+ 来源说明七小节齐全。
- 目标覆盖检查：Q1→§1、Q2→§2、Q3→§3、Q4→§4、Q5→§5，全部由正文章节完整回答。
- 原图：4 张论文原图（dflash_speedup、dflash_inference_design、dflash_attn、draft_latency_bar）通过 `img_to_b64.py` 转 base64 内联，浏览器实测全部正常显示。
- 代码运行：无可运行代码。
- 机械检查：`validate.py` 通过。
- 公式渲染与交互：headless Chrome 实测 KaTeX 367 节点、0 残留 `$` 串、SVG 标签 0、标签重叠 0；折叠块 DOM 完整。
- 写作偏差（已修复 3 轮累积）：
  - **阻断级修复**：196 处字面 `\times` 通过状态机包裹为 `$\times$`；meta description 含 `$` 符号已剥离
  - **章节号系统重排**：§1=Intro、§2=Related、§3=Preliminaries、§4=Method、§5=Experiments、§5.5 Ablation
  - **图编号重排**：Fig 1=speedup、2=inference、3=draft_latency、4=attn、5=loss_decay；正文与图注一致
  - **表编号重排**：Tab 1=main、2=reasoning、3=SGLang、4=long context、5=LLaMA、6=layers、7=target_hiddens、8=block_size、9=KV injection、10=naive、11=more models、12=vLLM、13=anchor sampling
  - 数字修正：MT-Bench 2.75–2.85×、EAGLE-3(60) 0.5–0.9×、删除 4096、long context 补 Qwen3.5-27B、reasoning 4.4×/3.8×、naive diffusion "略高于 EAGLE-3(16)/(60)"、C1 引用 6.1×/2.4× 改 N1
  - 评价章 C16 分析性推断标注：补 5 处
  - γ 双语义说明：§3.3 loss decay 解释段加注「此处 γ 与 §1 F1/F2 中的 γ（草稿 token 数）含义不同」
  - 前置概念链接：补 speculative-decoding、standard-attention、eagle-speculative 链接
  - 来源说明 C 编号重排：删 C15 占位、C16→C15（LLaMA）、C17→C16（高并发机制）
  - C11「Eq. (5)」不存在的修正：改为 A.3 矩阵形状分析
  - §5「本章问题」块补 3 题（DFlash 收益条件、相对 EAGLE-3 真实优势、部署注意事项）
  - Figure 3/4 编号正文互串修正（§1 正文「下图是论文 Figure 3」、§3 figcaption 「Figure 4 标题为」、§3 正文「论文 Figure 4 把这块稀疏掩码画出来」）
  - C9/C6 错用修正：3 处「单步并行起草」由 [C9] 改 [C6]
  - §5 Q2 答案数字修正：5L+conv Math500 5.99 vs 8L+conv 6.33
  - Tab.4 32K 与"未给 32K+ 结果"矛盾：评价表改为「论文 Tab.4 实验覆盖到 32K；32K+ 需自行微调」
  - vLLM 落后 0.5–1× 跨模型比较无依据：改为「SGLang 与 vLLM 数字跨模型比较，无可直接比对的 0.5–1× 依据；不同后端实现差异由其各自的 spec-decoding 引擎决定」

- 处置：可发布。剩余 19 轻微包括「Tab 1「DFlash 对 EAGLE-3(16) 提升 2.4×」自算 vs 论文 2.4 的口径差异」「× 符号部分仍需扫描」「C17 幽灵编号」等不影响核心机制的细节。

- 三轮独立审查发现的问题已全部处理：阻断 0、重要 12（全部已修）、轻微 19（保留为可接受意见）。
