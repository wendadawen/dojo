<!-- review-meta
round: 4
page: wiki/qwen3-5-dataflow/index.html
reviewed_content_sha256: a26219f9778e6241
-->
# Qwen3.5-397B-A17B 前向数据流审查记录（第 4 轮）

- 页面版本：68c70b027df4a20761b011b346898ed993a7cebe（工作树 index.html）
- 审查时间：2026-09-13 20:24
- 审查者：独立子代理（未参与写作与前序轮次）
- 适用规范：dojo:type=dataflow → `guides/model-dataflow.md`（表述维度并入 `guides/concept/check.md` 第 2.2 节第 12 项）
- 已完整阅读章节：1. 关键规格（含家族变体）、2. 交互式数据流、3. 要点（整体结构 / GDN / 全注意力层 / MoE 路由 / MTP 草稿层 / 长上下文开销）、4. 视觉编码器与多模态融合（4.1–4.8）、5. 与 Qwen3.8-Flash-Next 的架构对比、6. 核对方式、来源与范围说明；并逐条读取了 7 个视图的全部节点 tooltip（含公式与说明）与图例。
- 外部来源获取：`huggingface.co/Qwen/Qwen3.5-397B-A17B/config.json`（raw）、同仓库模型卡、`Qwen/Qwen3.5-{122B-A10B,35B-A3B,27B,4B}` config、`Qwen/Qwen3.8-Flash-Next` config、`huggingface/transformers@36deb0b5` 的 `models/qwen3_5/modeling_qwen3_5.py` 与 `models/qwen3_5_moe/modeling_qwen3_5_moe.py`、`models/qwen3_next/modeling_qwen3_next.py`、`vllm-project/vllm` 的 `qwen3_next_mtp.py`。`qwen.ai` 博客页在本环境无法取得正文，相关条目按"定位不到"处理（见下）。

## 问题

- [重要·技术] 3 要点「整体结构」第 2 条 vs 6 核对方式「端到端数据流」：同一台本机缩小模型，前处写"每层输入输出均为 [1,T,4096]"，后处写该模型"隐藏维 64""残差全程单流 64 宽"，两处互斥。引文依据：第 168 行"本机缩小模型逐层打点确认：每层输入输出均为 [1,T,4096]，无中间扩宽"；第 287 行"等比缩小模型（8 层 = [GDN×3, 全注意力]×2，隐藏维 64，4Q/2KV 头…）…逐层形状记录确认残差全程单流 64 宽"。｜修复要求：把第 168 行打点结论改为该缩小模型的实际形状 [1,T,64]，或改为"等效推断到 4096 宽"并注明依据是源码/权重而非缩小模型打点；两处形状口径必须一致。｜修复：｜复验：

- [重要·来源] 3 要点「MoE 路由」第 1 条与 5 对比表「MoE」行：称 Qwen3.8-Flash-Next"保留了 norm_topk_prob 开关"，本页来源表未列该断言的任何来源，且对照数据所依据的 `Qwen/Qwen3.8-Flash-Next/config.json` 中不存在该字段。引文依据：该 config 顶层键为 architectures / image_token_id / language_model_only / model_type / text_config / tie_word_embeddings / transformers_version / video_token_id / vision_config / vision_end_token_id / vision_start_token_id，全文检索 "norm_topk_prob" 无命中（同一方法在 `Qwen3.5-397B-A17B/config.json` 中可正常命中 `attn_output_gate: true`，说明检索有效）。｜修复要求：给出 Qwen3.8-Flash-Next 保留该开关的可定位依据（建模代码文件路径 + 行号，或 config 字段），否则删除两处括注/表格单元中的该归因。｜修复：｜复验：

- [重要·来源] 3 要点「MoE 路由」第 2 条与 2 视图「MoE 内部」aux loss 节点：断言"DeepSeek-V4 / Kimi-K3 的无辅助损失 sigmoid+bias 路由…后者推理时用偏置修正负载"，属对第三方模型的机制归因，本页来源表（7 行）与正文均未给出任何来源，也未链接站内对应页。引文依据：不适用（无来源可定位）。｜修复要求：补上可定位来源（官方报告/源码路径与行号），或删除该对比句；若保留应降级为不指向具体型号的一般性表述。｜修复：｜复验：

- [轻微·技术] 1 关键规格表「QK 归一化」行与「来源与范围说明」表：依据写作"源码 Qwen3NextAttention.__init__"，但本代 q_norm/k_norm 定义在 qwen3_5 自己的类里，来源表"（组件继承自 qwen3_next/ 与 qwen3_vl_moe/）"对注意力类不成立。引文依据：`models/qwen3_5/modeling_qwen3_5.py` 中 `class Qwen3_5Attention(nn.Module)`（自包含，不引用 Qwen3NextAttention），`self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)`。｜修复要求：把该行依据改为 `Qwen3_5Attention.__init__`，并把来源表的"组件继承自 qwen3_next/"限定到实际继承的组件（视觉侧继承 qwen3_vl_moe 等），不要整类归并。｜修复：｜复验：

- [轻微·规范] 2 交互式数据流：7 个视图的节点、公式、说明全部写在 `<script>` 的 `VIEWS` 对象里，`<div id="cy">` 为空容器，无脚本时视图内容完全不呈现。引文依据：`guides/model-dataflow.md`「视图：脚本失效时页面仍要能读：视图内容写在 HTML 里，脚本只负责切换显隐」，发布前检查「交互视图在无脚本时仍可读」。｜修复要求：把每个视图的节点/边数据改为写在 HTML 中（如各视图一个 `<template>`/隐藏块），脚本仅负责显示隐藏；或在 `index.html` 内联一份无脚本可读的静态节点-连线文本块供降级显示。｜修复：｜复验：

- [轻微·表述] 1 关键规格「家族变体」表首行：以"（本页）"作自我指代。引文依据：`<tr><td>397B-A17B（本页）</td>…`。｜修复要求：改为"397B-A17B（本文档主线型号）"或直接写"397B-A17B"，去掉对页面自身的指代。｜修复：｜复验：

- [轻微·表述] 2 交互式数据流 引言段：以图表说明替代直接进入路径，并用名词化术语。引文依据：第 151 行"下图按七个视图给出实际前向路径。标签页切换粒度：…该交互为阅读增强项：脚本失效时图表不可见，但第 3–6 节正文已给出全部结论与数值。"。｜修复要求：删去"下图按七个视图给出实际前向路径"这类引导语，保留必要的操作说明与降级说明；"该交互为阅读增强项"改为直陈（如"脚本失效时图表不显示；第 3–6 节给出全部结论与数值"）。｜修复：｜复验：

- [轻微·表述] 1 关键规格表前的说明句：描述本页表格自身结构，属元话语。引文依据：第 115 行"下表每一行都可回溯到具体依据。「config 键」列给出官方 config.json 中的字段名，「核对」列说明该数字如何被独立验证。"。｜修复要求：删去或压缩为一句陈述列义（如"「config 键」列为官方 config.json 字段名，「核对」列为独立验证方式"），不出现"下表每一行都可回溯…"式自我描述。｜修复：｜复验：

- [轻微·技术] 2 视图「多模态融合」posv 节点公式引入全文未定义的符号。引文依据：`f:'H = \\mathrm{arange}(h/2) + s,\\ W = \\mathrm{arange}(w/2) + s'`，该 `s` 未在页面任何处说明含义（应为该段在位置轴上的起点偏移）。｜修复要求：在节点说明里定义 `s`（如"s 为该图段在位置轴上的起始偏移"），或改用与正文一致的记号。｜修复：｜复验：

- [轻微·来源] 「来源与范围说明」表「官方架构与规模表述」行：博客只给标题与日期，未给 URL 与节次/行号，本环境无法定位正文核对。引文依据：`官方博客「Qwen3.5: Towards Native Multimodal Agents」（qwen.ai，2026-02-16）`（页面"官方博客表述为「Gated DeltaNet + Gated Attention 混合注意力 + 高稀疏 MoE」"依赖此条）。｜修复要求：补 URL 与节次/行号；若无法提供，把该引文（含 110 行引号内表述）降级为对模型卡/config 的表述或删除。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 7
- 处置：修复

本轮已回源核对且通过的项（供复验参考）：总参 403,397,928,944 = 语言主干 396,346,350,336 + 视觉 456,010,480 + MTP 6,595,568,128；命名口径 396.80B；单 token 激活 16,331,922,176（含词嵌入查表 17,349,040,896）；60 层 / hidden 4096 / 512 专家 top-10 / moe_intermediate_size 1024 / shared 1024 / GQA 32Q-2KV 头维 256 / GDN 16 键头-64 值头-128 维 / conv kernel 4 / full_attention_interval 4 / partial_rotary_factor 0.25 / mrope_section [11,11,10] / rope_theta 1e7 / max_position_embeddings 262144 / attn_output_gate=true / attention_bias=false / tie_word_embeddings=false / mtp_num_hidden_layers 1 / mtp_use_dedicated_embeddings=false / mamba_ssm_dtype float32 / router_aux_loss_coef 0.001 / 视觉 depth 27-hidden 1152-patch 16-merge 2-temporal 2-intermediate 4304-out 4096-num_position_embeddings 2304-hidden_act gelu_pytorch_tanh-deepstack 空 / 特殊 token 248053-248054-248056-248057，均与官方 config.json 逐字段一致。家族变体表 122B-A10B（48/3072/32-2/64/256 专家 top-8 I=1024）、35B-A3B（40/2048/16-2/32/256 专家 top-8 I=512）、27B（64/5120/24-4/48/稠密 17408）、4B（32/2560/16-4/32/稠密 9216/tie=true）四行与各自官方 config 一致。各分项加总（路由专家 60×6,442,450,944=386,547,056,640=95.82%；单层 MoE 激活 140,513,280；视觉六分组合计 456,010,480；KV/GDN 状态 0.938/7.500/30.000 GiB 与 0.179 GiB；激活 512 归一下界 0.010000）复算全部吻合。注意力输出门确为"逐头逐 token 标量"（`attn_output * torch.sigmoid(gate)`，gate 已 reshape 到 [*, heads]，与页面表述一致）；MoE 重归一化在 `Qwen3_5MoeTopKRouter.forward` 中无条件执行、代码中无 norm_topk_prob（页面 3.5 侧表述正确）；MTP 拼接 `[norm(embedding); norm(hidden)]` 与 vLLM `Qwen3NextMultiTokenPredictor.forward` 逐句一致。内部链接 16 个站内概念页全部存在；`.dojo/scripts/validate.py` 返回 `validation ok`；正文引用的 `research/measured.md` 为 `.md`（该目录下仅 `.md`）。
