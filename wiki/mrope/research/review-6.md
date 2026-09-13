<!-- review-meta
round: 6
page: wiki/mrope/index.html
reviewed_content_sha256: a03fea74d55be628
-->
# MRoPE 审查记录（第 6 轮）

- 页面版本：08e234736a84d1558a4016335ccee0c5eeb798cb
- 审查时间：2026-09-13 21:17
- 审查者：独立子代理
- 已完整阅读章节：核心问题、常见误解、1. 文本有序、图像有格——一维位置轴装不下多模态（含本章问题）、2. 位置 id 变三元组——三种模态的分配规则（含代码块与本章问题）、3. 一个头维装三条轴——分段与交错两种槽位排布（含折叠补充与本章问题）、4. 位置轴上的省账——推进量与长序列外推（含本章问题）、来源与范围说明；另读 overview.html。全文逐段通读，含全部折叠块与图注。

## 来源核对依据（本轮实际打开的外部材料）

- arXiv:2409.12191v2 §2.1「Model Architecture」原文：M-RoPE "deconstructing the original rotary embedding into three components: temporal, height, and width"；文本 "utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE"；图像 "temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components"；视频 "temporal ID increments for each frame"；同节 "reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference"。→ 对应 C1–C7，页面表述一致。
- §3.3.2（Ablation Study）Table 8：骨干 Qwen2-1.5B + ViT-L；NextQA 43.9→46.0、STAR 55.5→57.9、RealWorldQA 54.5→53.7、InfoVQA 50.8→50.3；原文 "Compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks"。→ 对应 N1，数字逐项核对一致（降幅项 RWQ/InfoVQA 亦成立）。
- Figure 5 图注原文："Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video. With the help of M-RoPE, the model demonstrated robust performance when the inference length exceeded the maximum training length of 16384 tokens."（正文另述 80K tokens）→ 对应 N2，页面「Qwen2-VL-72B 在 Video-MME 中等时长视频、16K 训练 / 80K 推理」逐项一致。
- 论文小节编号经核对为 2.1 Model Architecture、3.3.2（Ablation Study），页面所引 §2.1 / §1 / §3.3.2 均存在且内容对位。
- Qwen2-VL-7B-Instruct config.json（huggingface.co）：rope_scaling.type = "mrope"、mrope_section = [16,24,24]；hidden_size 3584、num_attention_heads 28 → head_dim 128。→ 页面 §3 分段例 $[16,24,24]$、64 槽位、cos/sin 维度 $[0,31]/[32,79]/[80,127]$ 与 $mrope\_section\times2=[32,48,48]$ 逐项一致。
- transformers qwen2_vl modeling_qwen2_vl.py：`mrope_section = mrope_section * 2`；`cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)`——分段实现的「乘 2 后按 i%3 取段」描述与源码一致。
- 页面代码块实机运行（python3），输出与「预期输出」逐字一致：`视觉 token 数 = 196, 推进量 = 14` / `视觉段首/末: (8, 8, 8) (8, 21, 21)` / `图后首文本: (22, 22, 22) (而非 (204, 204, 204))` / 四行 grid 输出含 `(1,66,120): 1980 个 token, 推进 60`。
- 复算：11+11+10=32；交错槽位 T{0,3,…,30}=11、H{1,4,…,31}=11、W{2,5,…,29}=10；16+24+24=64、32+48+48=128；196=14²、784=28²、1764=42²、1980=33×60、1980/60=33.0；8+196+5=209 且 22+5−1=26；8+1764+5=1777 且 8+42+5−1=54。全部相符。
- 链接：../rope/、../positional-encoding/、../vit/、../qwen3-8-flash-next-dataflow/、../qwen3-5-dataflow/index.html、overview.html 均真实存在；页面无「（待生成）」占位、无指向 dojo 仓库中不存在文件的路径。alt 属性中无 `$...$`。
- .dojo/scripts/validate.py wiki/mrope/index.html → validation ok。

## 问题

- [重要·技术] §2「本章问题」第 2 题解答折叠块（index.html L144）：末句把「退化回一维」的主语写成了视觉，与全文核心事实相反。｜引文依据：页面 L144 原文「MRoPE 的做法是让文本也用三元组（三值相同），统一表示下视觉才自然退化回一维」；对照 L164「文本三值相同意味着 MRoPE 在纯文本序列上功能等价于 1D-RoPE」及 L238「三值相同时三个分量给出的旋转相位一致……MRoPE 对纯文本无损」——退化回 1D 的是文本（$t=h=w$），视觉恰恰是保留二维（$h/w$ 按网格）的一方，不会退化回一维。｜修复要求：将「视觉」改为「文本」，使该句与 L164/L238 的结论一致（即：统一表示让文本自然退化回一维 RoPE）；若本意是「视觉不必被拆成独立体系」，需改写为不产生「视觉退化回一维」歧义的表述。｜修复：｜复验：
- [轻微·格式] §1 与 §2 符号复用，同一变量承担两种含义，违反「符号全文单义」。｜引文依据：L115「纵向相邻对应 $\Delta=W$（$W$ 为网格列数）」中 $W$=网格列数；L166 公式「$W=\mathrm{arange}(g_w/\text{merge})+s$」及 L273 槽位表列名中 $W$=宽分量，而网格宽在 §2 又记作 $g_w$；同理 L125「每个 token 带 $(t,h,w)$：右邻是 $w{+}1$」中 $h,w$ 为分量坐标，而 L95/L107/L309/L332「$\max(h,w)/\text{merge}$」中 $h,w$ 实指网格边长，同量在 L174/L335 又写作 $\max(g_h,g_w)/\text{merge}$。｜修复要求：统一记法——网格边长一律用 $g_h,g_w$，分量坐标一律用 $h,w$；§1 的网格列数改用 $g_w$（或明示符号表），使 $\max(h,w)/\text{merge}$ 与 $\max(g_h,g_w)/\text{merge}$ 合并为同一写法。｜修复：｜复验：
- [轻微·可读性] §4 正文（L325）使用未加解释的缩写「RWQ」，读者无法据此回源。｜引文依据：论文 §3.3.2 Table 8 该基准名为 RealWorldQA（54.5→53.7），页面写作「个别基准如 RWQ、InfoVQA 小幅下降」，「RWQ」为页面自造缩写且全文未定义。｜修复要求：写作「RealWorldQA」，或首次出现时注明「RealWorldQA（RWQ）」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（重要问题须关闭；两处轻微问题顺带修正）
