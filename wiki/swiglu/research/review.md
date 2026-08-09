# SwiGLU 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照 ar5iv 原文）
- 页面版本：index.html `178e2e01274e56bfa8c9a0c0377d3b6cc1bfcb79`；overview.html `b5cac9a48d94fcf2570233b6558dd0397c21d402`
- 时间：2026-08-09 15:05 CST
- 审查依据：`guides/concept/check.md` 段 A 盲读 + 段 B 对照来源
- 来源：WebSearch "SwiGLU GLU variants Shazeer 2020 arxiv 2002.05202" + ar5iv (https://ar5iv.labs.arxiv.org/html/2002.05202) 逐条核对 Eq.(4)/(5)/(6)、Table 1、§2/§3.1/§4 原文

## 审查范围说明

按任务包禁止读取 `research/` 目录、仓库其他页面、修改两份文档。因此 check.md 段 A 末"逐题核对 scope.md 学习目标"一项无法执行（scope.md 在 research/ 内）。改为核对页面自身 `index.html`"读完你能回答"5 条学习目标是否由正文完整回答——5 条均被正文覆盖（目标 2"手算小例子"受下方阻断问题影响，答案含数字错误）。

## 问题

- [阻断·技术] index.html §"SwiGLU 的公式、Swish 定义与手算"（教学示例，及 §"从 GLU 到 SwiGLU"折叠块"同输入下 GLU 与 SwiGLU 的逐维对照"、§"来源与教学说明·教学示例"）：教学示例声明 $W=\mathrm{diag}(1,-1)$、$x=[1.0,0.5]$，但 $xW$ 计算写作 $[1.0\times1+0.5\times0,\;1.0\times(-1)+0.5\times0]=[1.0,-1.0]$。按 $\mathrm{diag}(1,-1)$ 标准行向量乘法，第二分量应为 $x[0]\cdot W[0][1]+x[1]\cdot W[1][1]=1.0\times0+0.5\times(-1)=-0.5$，即 $xW=[1.0,-0.5]$；页面把对角元 $-1$ 错放到 $W[0][1]$ 位置（相当于用了 $W=[[1,-1],[0,0]]$ 而非 $\mathrm{diag}(1,-1)$）。下游 $\mathrm{Swish}(xW+b)\approx[0.7311,-0.2689]$、SwiGLU 输出 $[0.7311,-0.1345]$、GLU 对照门 $[0.7311,0.2689]$ 与输出 $[0.7311,0.1345]$ 全部基于 $xW=[1.0,-1.0]$ 展开，与声明权重不一致。学习目标 2 明确要求"手算一个小例子"，读者按声明 $W$ 复算得 $xW=[1.0,-0.5]$ 会与页面 $[1.0,-1.0]$ 冲突而卡住。修法：将 $W$ 改为 $\mathrm{diag}(1,-2)$（则 $xW=[1.0\times1,\;0.5\times(-2)]=[1.0,-1.0]$，下游全部自洽，"让第二维输入翻号"仍成立），并同步修正教学说明与折叠块中对 $W$ 的描述（$\mathrm{diag}(1,-1)\to\mathrm{diag}(1,-2)$）；或保留 $W=\mathrm{diag}(1,-1)$ 重算全部数字为基于 $xW=[1.0,-0.5]$（$\mathrm{Swish}(-0.5)\approx-0.1888$ 等）。 ｜ 修复：采用方案一，将 $W$ 改为 $\mathrm{diag}(1,-2)$。修改 4 处：正文教学示例声明（$W$ 描述）、$xW$ 计算式（中间步骤改为 $1.0\times0+0.5\times(-2)$，结果仍为 $[1.0,-1.0]$）、折叠块对照声明、教学说明。下游数字（Swish、GLU、SwiGLU 输出）均基于 $xW=[1.0,-1.0]$，无需改动。 ｜ 复验：

- [重要·技术] index.html §"从 GLU 到 SwiGLU" ASCII 图图注：图注称"两图都按 Shazeer 记法画（激活在 $W$ 分支、$V$ 为值分支）"，但 GLU 图中 $\sigma$ 画在 $V$ 分支（$xV+c\to\sigma(\cdot)$，标注为门分支），即 Dauphin 记法；与 Shazeer Eq.(4) $ \mathrm{GLU}(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c)$（$\sigma$ 在 $W$ 分支）矛盾。SwiGLU 图（$\mathrm{Swish}$ 在 $W$ 分支）才是 Shazeer 记法。正文 §"SwiGLU 的公式"又正确指出"GLU 概念页用 Dauphin 记法（$\sigma$ 在 $V$ 分支）、SwiGLU 用 Shazeer 记法（$\mathrm{Swish}$ 在 $W$ 分支）"，与图注"两图都按 Shazeer"自相矛盾。学习目标 4 含"说明 Shazeer 记法中 $W$ 与 $V$ 分别用于哪个分支；与 Dauphin 记法差在哪里"，图注错误直接误导该目标。修法：将图注改为"GLU 图按 Dauphin 记法（$\sigma$ 在 $V$ 分支，与 GLU 概念页一致），SwiGLU 图按 Shazeer 记法（$\mathrm{Swish}$ 在 $W$ 分支）；两记法相差 $W\leftrightarrow V$ 标签，因 $\otimes$ 可交换而等价"。 ｜ 修复：已将图注从"两图都按 Shazeer 记法画"改为"GLU 图按 Dauphin 记法画（σ 在 V 分支），SwiGLU 图按 Shazeer 记法画（Swish 在 W 分支）；两记法相差 W↔V 标签，因 ⊗ 可交换而等价"。与正文 §"SwiGLU 的公式"对两记法的说明一致。 ｜ 复验：

- [轻微·技术] index.html §"来源与教学说明·核心论断与来源" C3：C3 将 "we use $\beta=1$ in our experiments" 作为 Shazeer §1 末段直接引文。ar5iv 核对显示论文正文未单独出现该句，$\beta=1$ 是通过 Eq.(6) 直接使用 $\mathrm{Swish}_1$（及 Eq.(3) $\mathrm{FFN}_{\mathrm{Swish}}$ 用 $\mathrm{Swish}_1$）隐式体现。论点本身（实验固定 $\beta=1$）正确，但引文形式不精确，与页面 meta 声称"通过 ar5iv HTML 版逐条核对"不符。修法：将 C3 改为"由 Eq.(6) 使用 $\mathrm{Swish}_1$ 可知实验固定 $\beta=1$；§3.1 实验设置未提 $\beta$ 调参"，删除伪引号。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §"从 GLU 到 SwiGLU" 形状对照表：Swish 取值范围写作"$(-\epsilon,+\infty)$"，$\epsilon$ 未定义，且开区间暗示下确界不可达；但正文已正确说明 Swish 最小值 $\approx-0.278$ 在 $z\approx-1.278$ 处取得（应为闭的下界）。小白读者看到 $-\epsilon$ 不知所指。修法：改为"$\approx[-0.278,+\infty)$"或"最小 $\approx-0.278$，正侧无界"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §"SwiGLU 的公式" 符号说明（$b,c$ 条）："Shazeer §2 在 FFN 部署中省略偏置（见 S3）"中"S3"与全文其它处一致使用的"§3"记法不一致，读者不知 S3 指什么。修法：改为"见 §3"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html meta blockquote（"主要依据"段）：已发布页面元信息块链接到 `../../wiki/glu/research/evidence.md`（另一概念的 `research/` 内部工作文件）。`research/` 通常为未发布工作产物，从公开页面链接到内部工作文件可能不当（且本文档自身禁止审查者读 `research/`）。修法：确认项目约定；若 `research/` 不对外发布，改为链接 GLU 概念页 `index.html` 或移除该具体文件链接。 ｜ 修复： ｜ 复验：

## 段 B 对照来源核查记录（供复验参考）

已逐条核对，与 ar5iv 一致的项目（无需修复）：

- C1/F2 SwiGLU 定义 $\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$：= Shazeer Eq.(5)，逐字符一致。
- C5/F3 $\mathrm{FFN}_{\mathrm{SwiGLU}}=(\mathrm{Swish}_1(xW)\otimes xV)W_2$：= Shazeer Eq.(6)，一致；确认 $\mathrm{Swish}_1$（$\beta=1$）且无偏置。
- C6 $d_{ff}$ 缩 $2/3$ 引文：与 Shazeer §2 末段原文一致（页面用省略号省去中间括注，语义不变）。
- C7 实验设置 $d_{model}=768$、12 层编码/解码、$d_{ff}=3072\to2048$、segment-filling、524,288 步：与 ar5iv §3.1 一致。
- C8/N1 Table 1 八行 log-perplexity（524,288 步）：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633——**八个数字全部与 ar5iv Table 1 完全一致**，GEGLU 最优、SwiGLU 紧随。
- C9 "divine benevolence" 引文：与 Shazeer §4 末段原文逐字符一致。
- C11/F4/F5 $\tfrac83 d$ 推导：$\tfrac23\times4d=\tfrac83 d$，算术正确；LLaMA-7B $d=4096$、$d_{ff}=11008$（$\tfrac83\times4096\approx10922.67$ 向上取整到 256 倍数）正确。
- 参数量等式 $3d\cdot\tfrac83 d=8d^2=2d\cdot4d$：算术正确（$d=4096$ 时 $=134{,}217{,}728$）。
- Swish 边界值 $\mathrm{Swish}(0)=0$、$\mathrm{Swish}(1)\approx0.7311$、$\mathrm{Swish}(-1)\approx-0.2689$、最小值 $\approx-0.278$ @ $z\approx-1.278$：复算一致。
- 经验增益 ReLU→SwiGLU 降 $0.041$（$1.677-1.636$）、GEGLU vs SwiGLU 差 $0.003$（$1.636-1.633$）：算术正确。
- 前置链接有效性：`../../wiki/glu/index.html`、`../../wiki/situ-glu/index.html`、`../../index.html`、overview↔index 互链——目标文件均存在（仅查存在性，未读内容）。
- overview.html 与 index.html 论断一致，未发现额外事实错误。

未能在本次核对中验证（受来源访问限制，不列为问题）：C12 SiTU-GLU 源自 Kimi K3 Technical Report §2.3.2——需读 situ-glu 概念页/K3 报告，超出允许输入范围；论断本身与 overview 一致，留待该页质检。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：进入修复。阻断（教学示例 $W$ 与 $xW$ 不一致）与重要（GLU 图注记法标注错误）须先关闭；4 条轻微逐条修复或写明接受理由。学习目标 2、4 分别被阻断与重要问题影响，修复后应能闭环。scope.md 学习目标核对因 `research/` 访问禁令未执行，已改用页面自述学习目标核对（5 条均被正文覆盖）。
