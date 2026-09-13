<!-- review-meta
round: 5
page: wiki/quantization-basics/index.html
reviewed_content_sha256: a185d32b701189a0
-->
# 量化基础审查记录（第 5 轮）

- 页面版本：c1c4a6355f29d1bbc7e9996054f4505be388b865（index.html 工作树哈希；overview.html 93791ccb5d9425de8eb9de51cf7681e897e50375）
- 审查时间：2026-09-13 20:22 CST
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何文件）
- 已完整阅读章节（含折叠块与图注，按顺序）：核心问题；1. 量化要解决什么问题——压缩比与精度损失的权衡（含本章问题）；2. 均匀量化怎么把浮点变成整数——仿射变换与对称特例（2.1 对称量化特例、2.2 手算、本章问题）；3. 量化误差从哪来——离散级粗、舍入与裁剪（含本章问题）；4. 对称/非对称 与 量化粒度（4.1 对称 vs 非对称、4.2 粒度、本章问题）；5. PTQ vs QAT 与 浮点 block-wise 量化（5.1 PTQ、5.2 QAT、5.3 MXFP4、本章问题）；来源与范围说明。另核对了 overview.html。

## 来源核对记录（本轮逐条回源，均给出引文依据）

- C1/F1 仿射量化公式：Gholami 2021 §III-B 原文为 Q(r)=Int(r/S)−Z、r̃=S(Q(r)+Z)。页面记 x_q=clip(round(x/s)+z)、x̂=s(x_q−z)，并声明「本文 z=−Z，两式等价」——代入验证：round(r/S)+z = Int(r/S)−Z ⇒ z=−Z；x̂=s(x_q−z)=S(Q(r)+Z)，等价成立。
- C2/F2 对称 restricted range：§III-C 原文「in "restricted range" S is chosen as max(|r|)/(2^{n−1}−1), which only uses the range of [-127,127]」，与页面 s=max(|x|)/(2^{b−1}−1)、范围 [−2^{b−1}+1, 2^{b−1}−1] 一致。
- C3/F3 非对称：§III-C 原文 S=(β−α)/(2^b−1)、β=r_max、α=r_min；q_max−q_min=2^b−1，等价。z=round(q_min−α/s) 的推导（折叠块 L310-311）复算正确。
- C5 舍入与裁剪：§III-B 原文「the recovered real values r̃ will not exactly match r due to the rounding operation」；三来源分类页面已自标注为教学归纳，处理正确。
- C6 离群值：§III-C（左栏，紧接其后为 Summary (Symmetric vs Asymmetric Quantization)）原文「this approach is susceptible to outlier data in the activations. These could unnecessarily increase the range and, as a result, reduce the resolution of quantization」——章号 §III-C 经版面核对无误。
- C7 PTQ：§IV Figure 4 原文「In PTQ, a pre-trained model is calibrated using calibration data (e.g., a small subset ...)」，与页面一致；128–512 校准样本已标注为「常见工程值，非硬性规则」。
- C8/F4 QAT 与 STE：§III-G 原文「STE essentially ignores the rounding operation and approximates it with an identity function」，支持 ∂ŵ/∂w≡1。
- C9/N2 与 N4 MXFP4/E2M1：emergentmind.com/topics/mxfp4 原文「The E2M1 codebook yields the set {0,±0.5,±1,±1.5,±2,±3,±4,±6}」、块 32、E8M0、E2M1(1+2+1)——与页面 N4、5.3 描述一致；136 bit/32=4.25 bit 复算正确。
- C10 gpt-oss 部署事实：NVIDIA TensorRT-LLM 仓库 examples/models/core/gpt_oss/README.md 原文「GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the other weights are in bf16.」——与页面一致。
- N1 数字：NVIDIA Developer Blog 原文「If the Llama2 7B model is stored in FP16/BF16, each parameter occupies 2 bytes, resulting in a total memory usage of approximately ~14 GB」；7×10^9×2=14 GB、INT8≈7 GB、INT4≈3.5 GB 复算正确。
- 代码（5.2 折叠块）：抽出 <code class="language-python"> 实际执行，输出与页面「预期输出」逐行一致（E1 x_q=[2,4,7]/error=[0.4,-0.2,0.0]；E2 scale=7.1429/x_q=[0,0,1,7]/error=[-1.2,-3.4,1.5429,0.0]；E3 块 A scale=0.8、块 B scale=7.1429）。
- 手算复算：2.2 [1.2,3.4,5.6]（s=0.8）、3 章 [1.2,3.4,5.6,50.0]（s≈7.143）、本章问题 [2.0,4.0,6.0]（s=6/7）全部复算正确；无同一页内两处矛盾；无指向仓库中不存在文件路径的引用（正文/来源说明均无 research/ 或 .md 路径）。
- validate.py wiki/quantization-basics/index.html → validation ok。
- overview.html 与 index.html 相互链接；被引用的 ../mxfp4-qat/index.html 真实存在；无「（待生成）」占位。全页无会话指代（未出现「我/我们/你」）。

## 问题

- [轻微·格式] 2.2 手算折叠块（L211）：`Python 内置 <code>round</code> 用" banker's rounding"` 引号与 `banker's` 之间多一个半角空格｜引文依据：不适用｜修复要求：删除该多余空格，改为 `用"banker's rounding"`｜修复：｜复验：
- [轻微·格式] 5.2「辅助解释」callout（L527）：`<b>辅助解释</b>。 可以` 句号后多一个半角空格，与全页中文标点后不留空格的排版不一致｜引文依据：不适用｜修复要求：删除「。」后的空格｜修复：｜复验：
- [轻微·表述] 开篇（L74）、各章末（L153/240/294/376/576）、结尾（L635）：以「本文/本章/下一章」为主语的自我指代与结构预告属元话语，如「本文回答：均匀量化如何编码浮点、误差从何而来…」（L74）、「本章明确了…但还没说…——下一章给出均匀仿射量化公式并手算」（L153）、「本文从"为什么量化"出发，依次给出了…」（L635）｜引文依据：不适用｜修复要求：把结构预告改写为承上启下的实质衔接——直接陈述下一章要解决的问题本身，删去「本文回答／本文…依次给出了／本章…下一章给出」这类以文稿自身为主语的表述｜修复：｜复验：
- [轻微·表述] 5.2 STE 折叠块（L420）：「注意 STE 令导数为 1 不等于"量化没有误差"」中的「注意」为祈使式元话语（与规范所列「需要注意的是」同类）｜引文依据：不适用｜修复要求：改为陈述句，直接说明 STE 令导数为 1 的性质与边界，去掉「注意」｜修复：｜复验：
- [轻微·技术] 5.3（L550）：「MXFP4 在大模型部署中已成为事实低位宽方案」是无来源支持的判断写成结论；C9/N2 只支持 MXFP4 的格式定义（块 32/E8M0/E2M1），C10 只支持 OpenAI gpt-oss 一例｜引文依据：TensorRT-LLM gpt_oss README「GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the other weights are in bf16.」（仅单例部署事实，无「已成为事实方案」之据）｜修复要求：降级为可核对表述，如「MXFP4 已被 OCP 规范收录，并用于 gpt-oss 等模型的 MoE 权重存储」，删除「已成为事实低位宽方案」这一定性判断｜修复：｜复验：
- [轻微·事实] 1 章开篇（L119，overview.html 第 2 节同）：「部署到 24 GB 的消费级显卡已几乎装不下」把 14 GB 权重 + KV cache/激活的余量夸大；24 GB 卡容纳 7B FP16 权重（约 14 GB）后尚余约 10 GB，普通上下文与批量的 KV cache/激活通常可容纳，「几乎装不下」属未标注的估计被写成事实｜引文依据：NVIDIA Developer Blog 原文仅给「~14 GB」内存占用，未称 24 GB 装不下｜修复要求：改为带条件的表述（如「显存余量吃紧」）或补充具体条件（长上下文/大批量推理时），并将该判断标注为估计｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 机制/公式/数字/代码本轮全部回源核对通过，未发现核心结论错误、来源不支持或页内矛盾；剩余 6 项均为表达与排版层面，不影响正确性与主线理解。
- 处置：修复（改完 6 项轻微后即可发布）；三项校验（来源、代码执行、validate.py）均通过。