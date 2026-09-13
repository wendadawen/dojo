<!-- review-meta
round: 6
page: wiki/standard-attention/index.html
reviewed_content_sha256: 34a5d71ca31fd18a
-->
# 标准 Transformer 注意力审查记录（第 6 轮）

- 页面版本：f3dc6435a7aca0262a4f5cd63d9d64d3（工作树，HEAD 36538b0）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 注意力要解决什么问题——从 RNN 的"逐步传递"到"直接查询" / 2. 缩放点积公式——每个符号与每一步 / 3. 为什么除以 √d_k——缩放因子的方差推导与不缩放的后果 / 4. 多头注意力——拆分子空间、拼接、参数量等价 / 5. 复杂度、瓶颈与边界——标准注意力不能做什么 / 来源与范围说明（含全部折叠块与图注）

本轮回源与复算范围：Vaswani et al. 2017（arXiv:1706.03762）§3.2.1 脚注原文 "assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q·k=…, has mean 0 and variance d_k"、§3.2.1 对比句 "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k"、§3.2.2 "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."、§4 Table 1（Recurrent O(n·d²)/O(n)/O(n)；Convolutional O(k·n·d²)/O(1)/O(log_k n)；Self-Attention O(n²·d)/O(1)/O(1)）、§3.2.2 与 Table 3 超参（d_model=512、h=8、d_k=d_v=64），均与页面表述一致，引文可定位。FlashAttention（Dao et al. 2022）"3× speedup on GPT-2 seq 1K / 15% end-to-end on BERT-large seq 512" 与页面一致；Linear Attention（Katharopoulos et al. 2020）复杂度 O(n·d²) 与页面一致。页面数字复算全部通过：softmax([0.7071,0])=[0.670,0.330]、softmax([1,0])=[0.731,0.269]、AV 各元素、3×3 遮罩三行归一（0.401/0.599、0.258/0.316/0.426）、2048²=4.19×10⁶、32768²≈1.07×10⁹、e²³≈9.74×10⁹、ln256=5.545、1/256=0.004；复算表（d_k=4/64/1024，未缩放最大权重 0.171/0.748/0.939、熵 3.98/0.75/0.15；缩放后 0.044/0.044/0.043、熵 5.07/5.05/5.05）与页面 0.170/0.749/0.936、3.97/0.75/0.16、0.043/0.043/0.043、5.05/5.05/5.05 一致。validate.py 通过；C/F/N 编号与来源章节双向对应齐全；../rope/ 与 ../mla/ 链接目标存在；无 alt 含 `$...$`；无 Unicode 数学字符（`×` 属 validate.py 明确豁免的排版字符）。

## 问题

- [重要·技术] index.html 第 322 行（§3「不缩放的后果」）：闭式 "$8\sqrt{2\ln n}$" 与紧随其后的复算数字 11 / 23 不符，读者代入 n 会得到与标注数值差约 18%–44% 的结果，构成同一处"式—数不一致"。｜引文依据：本页给式 $8\sqrt{2\ln n}$，复算 8√(2 ln 8)=16.31、8√(2 ln 256)=26.64；正文却写"n=8 时约 11，n=256 时约 23"。蒙特卡洛（q,k 各分量 iid N(0,1)、d_k=64、20 万次）得 E[max logit]=11.33（n=8）、22.54（n=256），与 11 / 23 吻合而与闭式不符——差额来自高斯极值期望的修正项，页面未说明该修正项。｜修复要求：写明 8√(2 ln n) 只是随 n 增长的主项、对有限 n 会高估，或改用含修正项的期望表达式，使式与数自洽（例："量级随 8√(2 ln n) 增长；按上述设定复算，n=8 约 11、n=256 约 23"）。｜修复：｜复验：
- [轻微·技术] index.html 第 322 行：以"最大 logit 与典型值的差值"（≈23）论证"某个 key 几乎独占权重"。softmax 平移不变，主导程度取决于最大与次大 logit 之差而非最大 logit 的绝对值；本页自身复算表在 d_k=64 行给出的未缩放最大权重为 0.749，与"几乎独占"不符。｜引文依据：本页第 341 行 "64｜0.749｜0.75"；第 322 行 "e^{23}\approx 9.7\times 10^9，让某个 key 几乎独占权重"。｜修复要求：改为与复算表一致的表述（如"最大权重升到约 0.75、熵降到约 0.75，分布已明显集中"），或注明该句描述的是更大 d_k 的极限。｜修复：｜复验：
- [轻微·表述] index.html 第 302 行："本文把脚注展开。"——以"本文"为主语的自我指代（元话语）。｜引文依据：不适用｜修复要求：改为不自我指代的写法，如"该推导在脚注中给出，下面展开"。｜修复：｜复验：
- [轻微·表述] index.html 第 597、598 行（来源与范围说明 F3、F4）："本页展开为完整推导""本页用于支撑 C4"以"本页"为主语。｜引文依据：不适用｜修复要求：改为客观陈述，如"完整推导见正文 §3""本页给出该推导用于支撑 C4"之外的表述，避免以"本页"作主语。｜修复：｜复验：
- [轻微·表述] index.html 第 496 行与第 500 行：§4 末句与 §5 首句近乎重复（"……决定了哪些问题必须由后续变体解决"连出两次），跨标题连读时明显冗余。｜引文依据：不适用｜修复要求：删除或改写其中一句（第 500 行首句可与第 496 行合并）。｜修复：｜复验：

## 结论

- 处置：修复（1 条重要 + 4 条轻微待关闭；无阻断）
- 统计：阻断 0 / 重要 1 / 轻微 4