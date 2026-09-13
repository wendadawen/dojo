<!-- review-meta
round: 6
page: wiki/kv-cache/index.html
reviewed_content_sha256: fb4302a6ed5eb6a6
-->
# KV cache 审查记录（第 6 轮）

- 页面版本：227b07bbdc09e218dafa81bf71bb3c2c5df9486f
- 审查时间：2026-09-13 21:13
- 审查者：独立子代理（未参与写作，也未参与前序审查与修复）
- 已完整阅读章节：开篇与「核心问题」；1. 注意力为什么需要缓存——K/V 与查询无关（含 4 token 表格、折叠块、本章问题）；2. prefill 与 decode——缓存产生的两个阶段（含流程图与本章问题）；3. 缓存有多大——每 token 字节数公式（含折叠块、本章问题）；4. 为什么显存成为瓶颈——动态缓存与权重的争夺（含表格、本章问题）；来源与范围说明（论断 C / 公式 F / 外部数字 N、构造示例、辅助解释、简化条件）。全文含折叠块与图注逐段通读。

## 问题

- [重要·技术] 「主要依据」段（第 62 行）与来源章节 C2（第 298 行）：页面两处把 Strata 论文标注为 arXiv:2508.18572v2，但该版本不存在，arXiv 上仅有 v1，按所标位置无法打开来源。｜引文依据：HTTP 请求 https://arxiv.org/abs/2508.18572v2 返回 404，https://arxiv.org/abs/2508.18572 仅列 Version 1（2025-08-26 提交）；v1 §2.1 原文 "During prefill, the model typically processes both (i) new tokens from the user query and (ii) context tokens ... In the subsequent decode phase, the model generates tokens autoregressively, continually reusing and extending the KV cache."（即 C2 所需内容在 v1 中）｜修复要求：把第 62、298 行两处 "arXiv:2508.18572v2" 改为实际存在的版本号 arXiv:2508.18572v1（与 wiki/strata 页一致）；若作者确实引用了未公开的 v2 文件，则改为可定位的 v1。｜修复：｜复验：
- [轻微·格式] 公式符号不统一（第 211、215、238 行，dojo:summary 第 7 行）：层数变量在定义处（第 211 行公式与第 215 行符号表，及来源章节第 301 行）写作 $L_{\text{layers}}$，但在第 3 章本章问题解答（第 238 行）写作 $L$（"$2\cdot L\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$"），行内公式亦出现 $L$；KV 头数在 dojo:summary 写 $H_{\mathrm{kv}}$、正文写 $H_{\text{kv}}$。同一变量全页未保持一种写法。｜引文依据：不适用｜修复要求：全页（含 dojo:summary）统一为 $L_{\text{layers}}$ 与 $H_{\text{kv}}$，使每个变量只有一种写法。｜修复：｜复验：

## 本轮已核对、未构成问题的来源论断（核对依据）

- N2（Llama-3.1-8B 配置）：原仓库 meta-llama/Llama-3.1-8B-Instruct 需鉴权（401），经同权重镜像 NousResearch/Meta-Llama-3.1-8B-Instruct config.json 读取——num_hidden_layers 32、num_attention_heads 32、num_key_value_heads 8、hidden_size 4096、head_dim 128、max_position_embeddings 131072、torch_dtype "bfloat16"，与页面第 90、225、238 行一致。
- N1（OPT-13B 800 KB/token、单请求 1.6 GB）：vLLM 论文 §3 原文 "the KV cache of a single token demands 800 KB of space ... the memory required to store the KV cache of one request can be as much as 1.6 GB"，算式 2×5120×40×2；与页面第 225、260 行一致。
- C4/N3（显存分布）：vLLM §1 原文 "Approximately 65% of the memory is allocated for the model weights ... Close to 30% of the memory is used to store the dynamic states of the requests"，Figure 1 图例标注 KV cache 段约 30%；与页面第 252、262 行一致。
- C5（请求完成即丢弃缓存）：vLLM §4.3 "Once a request finishes its generation, its KV blocks can be freed"；SGLang §1 "the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls"；与页面第 268 行一致。
- N4（40GB≈0.3M token）：Strata §1 原文 "40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B"；与页面第 254 行一致。
- 复算全部通过：2×32×8×128×2=131,072 B=128 KB；20,000×131,072 B≈2.44 GiB；131,072(128K token)×128 KB=16 GiB；2×5120×40×2=800 KB，2048×800 KB≈1.6 GB；40×2^30/131,072≈0.33M；1+2+3+4=10、n(n+1)/2 对 n=20000 为约 2 亿；正文与 dojo:summary、overview.html 的数字（128 KB、2.44 GB、16 GB、800 KB、0.3M、65%/30%）三处一致，无相互矛盾。
- 链接与功能：standard-attention、prefix-caching、hetero-pd、paged-attention、strata、hisparse 六个被引页面均真实存在（无「（待生成）」占位）；hisparse 所引 arXiv:2608.07009v1 存在（2026-08-07 提交）；结构图为 HTML（无 SVG `<text>`、无等宽框线图），无 img alt 属性内容（不含 `$...$`）；页面无 Unicode 数学字符；python3 .dojo/scripts/validate.py wiki/kv-cache/index.html 返回 "validation ok"。
- 表述维度：未发现元话语（"下面来看""需要注意的是"）、会话指代（我/我们/你）、调试复现叙事、临场评价；"本文/本页"自称符合 style-guide §12；"场景"仅作「手册场景」的普通名词，未术语化。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复（改两处 arXiv 版本号；统一 $L_{\text{layers}}$ 与 $H_{\text{kv}}$ 写法）
