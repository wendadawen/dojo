# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-3.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**8** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**13** 条
- 有理由不改：**2** 条

## 修复与判定说明

通读 research/review-1/2/3.md 三份记录，汇总出 14 条轻微问题（第 1 轮 9、第 2 轮 1、第 3 轮 4），逐条回验当前页后：13 条判定「已不存在」，1 条实修；另按本任务「不得出现指向不存在文件路径」的硬约束修掉 7 条来源标注缺陷。故 fixed=8、skipped=13。
【实修 8 条】① R3-4（轻微·来源）：attn 视图 woa 节点 d 仍写「按 8 组各降到 1024 维，避免一次处理 32768 维」，把无来源支持的设计动机写成事实，且与 §3 要点已改口径（「分组的设计动机未在来源中说明」）自相矛盾；改为「按 8 组各降到 1024 维。真实权重形状 [8192, 4096]；分组的设计动机未在来源中说明。」②–⑧ 指向仓库中不存在文件的路径（成因：13ead44 起 research/ 只保留 .md，本页引用的脚本与 .out 存档已登记进 wiki/deepseek-v4-1/research/measured.md 后删除）：§7 的 wiki/deepseek-v4-1/research/（空目录）与 ckpt/verify_dataflow_shapes.out；来源表四行里的 official/inference/config.json、official/inference/model.py、ckpt/headers.json、research/official/tech_report.txt、research/*.py + research/ckpt/*.out；以及规格表与 §7 共 10 处已移除的实测脚本名（count_params.py、verify_cache_size.py、verify_active_params.py、verify_csa2_modes.py、verify_rope_yarn.py、verify_sparse_attn_window.py、verify_gate_formula.py、verify_sinkhorn.py、probe_engram_layout.py、verify_halfstack_diff.py）与「同上脚本」衔接语。全部改为「实测/运行期实测」表述，实测清单统一指向实际存在的 wiki/deepseek-v4-1/research/measured.md（该清单逐条登记了本页引用的全部脚本与 .out）；官方源码名（model.py/kernel.py/inference/config.json 及行号）作为外部来源定位保留不动，与已转干净的姊妹页 deepseek-v4-dataflow 的做法一致。
【已不存在 13 条】（均为历轮修复或页面再生成后的副作用，当前页已满足修复要求，未再改动）R1：hc1 公式多出的 1/4 因子（现无平均因子）、候选池引文 565–567（现为 565–570）、来源表 L 行号括注（现给 model.py/kernel.py 机制行号）、attn 边 kvn→cmp（现为 hc1→cmp「经 Compressor 独立投影」）、gt/cmp 图例同色（COLOR.gt 已改 #b7791f、cmp 保持 #e0922f）、正文 Unicode 数学字符（U+2212 计 0，×/→ 仅存于 VIEWS 画布字段，按已记录接受理由保留）、样式表缺换行（页内样式已外链到 libs/，本页无该规则）、每 token 激活行的 decode 说明（已补「decode 为两半之和 15.4687B」）、导语 prefill 归因（已加「配合解码器滑动窗口的有界重放」）；R2：sparse 图例多余的「可达集 #8a93a3」（已并入「索引器打分与可达集」）；R3：overview 图例「全局 KV(可下钻)/层段(可下钻)」（现无该标注，且 6 条图例与节点 drill 一一对应）、attn 边（同上）、CSA2 缩写未给全称（导语已补「Compressed Sparse Attention 2，压缩稀疏注意力第 2 版」）。
数字与引文编号一致性未受影响：改动只涉及来源标注措辞与一处动机断言，未触及任何数值；summary/description/正文/图注中的 890 B、288 B、2048×8=16384、报告行 320 / 314–317 / 565–570 全部保持一致。
## 不改的条目及理由

- R1-1 的附带要求「系数写作 sigmoid(...)+ε」中的 +ε 未采纳：被指缺陷（公式多出的 1/4 平均因子）在当前页不存在（现为 \tilde{x} = \textstyle\sum_i \mathrm{sigmoid}(\mathrm{mixes}\cdot\mathrm{scale}+\mathrm{base})_i \odot x_i），按「问题已不存在就不要重改」处理；该式与页面其余示意公式（hc2 的 x_i <- x_i + λ_i y 等）保持同一粒度。
- 暗/亮模式切换的 JS 注释仍写「带 localStorage 持久化 + highlight.js 主题切换」（页面实际用 Prism）：该注释是 96 个页面共有的模板文案，非本页问题，三轮审查均未提出，按不重构原则不作本页单点修改。
