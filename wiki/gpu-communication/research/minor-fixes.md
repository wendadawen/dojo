# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-3.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**5** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**9** 条
- 有理由不改：**0** 条

## 修复与判定说明

读完 research/ 下 review-1/2/3.md，汇总去重后共 14 条轻微问题（第1轮8条、第2轮3条含2条与第1轮重复、第3轮5条）。实修 5 条（全部来自第3轮，前两轮未复报、现页面仍在）：(1) reduce_scatter 段「它的两个阶段分别对应 all-reduce 的前半段」逻辑不成立 → 「先算后切，即 all-reduce 的第一阶段（reduce-scatter）」；(2) HCA 的 Adapter「指转接两种规格」无来源推断 → 删去该半句；(3)「换…是唯一能实现维度互换的原语」无条件「唯一」→「是用于『维度互换』的原语」；(4) 来源栏把「UCX 为默认传输后端」记在 github.com/ai-dynamo/nixl（该仓库无 default 字样）→ 改引 NVIDIA Dynamo 文档《KV Cache Transfer》(docs.dynamo.nvidia.com/dynamo/additional-resources/tensor-rt-llm-details/kv-cache-transfer, 实测 HTTP 200)，nixl 仓库保留作后端清单出处；(5) CSS 空规则 @media (prefers-color-scheme: dark) { } 与残留缩进空行 → 删除。判定「已不存在」9 条：第1轮8条（『我』会话指代、『后文所有术语都围绕这条主线展开』元话语、『艺名/秒切』口语、NVSwitch『8条线』未标简化、GPUDirect『官方表述：』直引包装、步数只写 N-1 未写 2(N-1)、UCX『可选后端之一』、三处『容易混淆』元话语）与第2轮中2条重复项，现页面均已按修复要求改妥；第2轮余下1条（来源缺可定位 URL）关键论断（环形公式/TCCL/GPUDirect/NIXL）均已补 URL，仅『PCIe 与 NVLink 带宽量级』一条属量级对比且已注明『具体数值随代际变化』，第3轮复验已接受，不改。未做无关重构：CSS 块其余重复规则未动（与共享 dojo-note.css 重复，但非本轮所指问题）。页面无 research/ 或 .md 路径、无含 $ 的 alt、无 overview 需同步；summary/description 未改动故无需同步。文件：/Users/wendadawen/code/dojo/wiki/gpu-communication/index.html。
