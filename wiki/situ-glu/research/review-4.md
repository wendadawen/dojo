<!-- review-meta
round: 4
page: wiki/situ-glu/index.html
reviewed_content_sha256: fb55d143fbc9b6df
-->
# SiTU-GLU 审查记录（第 4 轮）

- 页面版本：f5fbc89aff81345cc52d5b214976eea3d3ce32de
- 审查时间：2026-09-13
- 审查者：独立子代理
- 适用规范：guides/concept/check.md（dojo:type=concept）
- 来源获取：Kimi K3 Technical Report（arXiv 2607.24653），正文 §2.3.2 与附录 §B 为核对对象；另核对 Shazeer 2020《GLU Variants Improve Transformer》。原文经 https://arxiv.org/html/2607.24653 抓取后逐段比对。
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么需要给 SwiGLU 加上界——激活爆炸从哪里来 / 2. SiTU-GLU 的公式与手算——定义、符号与三个边界点 / 3. 近原点像 SwiGLU、远点饱和——两个性质怎么同时成立 / 4. 为什么是 softcap 而不是 clip——饱和区里梯度差别 / 5. 在 K3 中的使用位置与不解决——两处使用与四条边界 / 来源与范围说明（含全部折叠块与表格）
- 机械验证：.dojo/scripts/validate.py 返回 validation ok；页面无 Unicode 数学字符裸写；GLU 前置链接 ../../wiki/glu/index.html 真实存在；overview.html 与 index.html 互链。

## 已核对通过的来源事实（供复验）

- F1/§2.3.2 定义式与 [β1 tanh(Wgx/β1)⊙Sigmoid(Wgx)]⊙[β2 tanh(Wux/β2)] 逐字一致（Eq.12）；F2 softcap(x,β)=βtanh(x/β) 一致。
- β1=4（gate）、β2=25（up）、上界 100 一致（§2.3.2 末段 + §B Eq.19）。
- F3 局部展开 βtanh(z/β)=z+O(z³/β²) 与 §B Eq.(18) 一致；"β1,β2→∞ 逐点收敛到 SwiGLU"与 §B 原文一致。
- C8 引文 "Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior." 与 §B 末段逐字一致。
- C9 引文 "coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic" 与 §2.3.2 逐字一致。
- 三件套（RMSNorm §2.3.1 / SiTU-GLU §2.3.2 / QB §2.3.3）、896 专家、2.8T 规模、"a chain of nearly four consecutive matrix multiplications"、KDA–MLA 混合注意力，均与 §2.3、§3 Table 1 一致。
- 手算复算全部吻合：g(0)=0,u(0)=0,y(0)=0；y(10)=3.9463×9.4987≈37.485（SwiGLU 99.9955）；y(100)=4×24.9832≈99.933（SwiGLU 10000）；折叠块 x=2→3.2493、x=50→96.403；4tanh(0.125)=0.497412 与 z−z³/(3β²)=0.497396 吻合；4e⁻⁵⁰≈7.7×10⁻²²、4e⁻⁸≈1.34×10⁻³ 与页面一致。

## 问题

- [重要·技术] §1 第三段（"K3 §2.3.2 第二段给出两个理由"）与本章问题 2 解答：页面把"GLU 的 sigmoid 门虽然有界，但值分支 $W_u x$ 仍然无界——乘积仍无界"作为来源明确给出的理由之一，并写明"两点依据均为 K3 §2.3.2 第二段（C3）"。来源该段只给出一条明确理由（GLU 不保留 Swish 正侧近似线性响应），并未陈述 GLU 值分支无界。"值支无界→乘积仍无界"虽由 GLU 定义 $ \sigma(W_g x)\odot W_u x$ 可推得，但属页面自身推断，被写成了来源结论。｜引文依据：K3 §2.3.2 "The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish."（该段无任何关于 GLU 值分支无界或 GLU 乘积无界的表述）｜修复要求：把"值支无界"一条明确标注为本页依据 GLU 定义的推断，或改写为"来源只给出一条理由（丢失 Swish 正侧响应）"并删去对 §2.3.2 的"两个理由"归因。｜修复：｜复验：

- [重要·可读性] §4 第 5 段"这里要做一个边界澄清（误解 5）"，及 §5"收尾的几条常见误解（与开头'最容易误解'对应）"：开头"最容易误解"块只有 4 条且未编号，§4 却引用"误解 5"；§5 的对照清单有 5 条（比开头多出"$\beta_1=4,\beta_2=25$ 是普适最优"）。§3 的"误解 3"按开头清单编号（第 3 条"只套门支就够了"），§4 的"误解 5"按 §5 清单编号（第 5 条"softcap 在饱和区等价于 hard clamping"），两处编号口径互不一致；且 §5"与开头'最容易误解'对应"的说法与事实不符（多出一条）。读者按编号到开头块查找"误解 5"会落空。｜引文依据：不适用｜修复要求：统一误解编号口径——把"$\beta_1=4,\beta_2=25$ 是普适最优"补入开头"最容易误解"块使两处条目一一对应并统一编号，或删去"误解 5"引用与"与开头对应"的表述。｜修复：｜复验：

- [轻微·技术] §3"输出上界"段、§3 折叠块、F4、N1、结尾段：用 $|\mathrm{SiTU\text{-}GLU}(x)|\le\beta_1\beta_2=100$ 表示向量输出的界，符号 $|\cdot|$ 作用在向量上不严谨；来源用的是无穷范数。｜引文依据：K3 §B Eq.(19) "‖SiTU-GLU(x)‖∞ ≤ β1β2 = 100"｜修复要求：改用 $\|\mathrm{SiTU\text{-}GLU}(x)\|_\infty$，或明确写成"每个输出坐标的绝对值"。｜修复：｜复验：

- [轻微·技术] §5 使用位置第二条、核心问题 5 解答、C10、N2：把 §3 Table 1 的模型级行"Activation Function: SwiGLU → SiTU-GLU"表述为具体使用位置"Dense FFN"。来源全文未出现"Dense FFN"字样（仅有 Dense Layers 行与 activation function 行），该定位是页面对模型级行的收窄解读，按来源原义应为"模型 FFN 激活函数整体由 SwiGLU 换为 SiTU-GLU"。｜引文依据：K3 §3 Table 1 行 "Activation Function | SwiGLU | SiTU-GLU | –"｜修复要求：按表格原义表述为"模型级 Activation Function 行"，或标注为推断。｜修复：｜复验：

- [轻微·技术] §1 第二段：写"SwiGLU 是当前最常用的版本"。这是无来源支持的最高级判断，来源只到"被广泛采用"。｜引文依据：K3 §2.3.2 "SwiGLU has subsequently become a widely adopted FFN design in large language models"｜修复要求：降级为"被广泛采用"，或标注为判断。｜修复：｜复验：

- [轻微·技术] §5 误解 2 条目内的英文引文：写作 "preserves the local response of SwiGLU while controlling both factors"，与来源措辞不一致（来源为 preserve ... in the product），引文被改写。｜引文依据：K3 §2.3.2 "allowing SiTU-GLU to preserve the local response of SwiGLU while controlling both factors in the product."｜修复要求：按原文逐字引用，或去掉英文引号改为中文转述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（两处重要问题需关闭后发布；无核心结论错误，公式、数字、引文主干均已回源核对通过）