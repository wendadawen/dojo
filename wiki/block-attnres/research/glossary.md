# Block AttnRes：术语表

登记全文所有首次出现的术语、缩写和符号。名称、首次出现位置、定义或含义。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| 残差连接 / Residual Connection | S1 | 把一层的输入直接加到该层（或若干层）的输出上，使信息可以绕过中间的非线性变换跨层直连。本页前置概念，详见 `wiki/residual-connection/`。 |
| 标准残差 | S1 | 本页中对"等权累加"形式残差连接的简称，即 $h_{l+1}=h_l+F_l(h_l)$。 |
| 等权累加 | S1 | 标准残差把所有前层输出以系数 1 累加进当前残差流的性质。 |
| RNN-over-depth 瓶颈 / 深度方向的 RNN 瓶颈 | S1 | 标准残差在深度方向把所有历史压成单一流 $h_l$，与 RNN 在时间维度把所有历史压成 hidden state 的结构类比。 |
| AttnRes / Attention Residuals / 注意力残差 | S1 | 用 attention 替代等权累加的跨层信息流机制；本页概念的核心。 |
| Full AttnRes / Full Attention Residuals | S2 | AttnRes 的完整形式：每层对此前所有层输出做 attention，开销 $O(Ld)$ 内存。 |
| Block AttnRes / Block Attention Residuals | S3 | AttnRes 的分块形式：$L$ 层分 $N$ 个 block，块内求和成单表征，块间 attention，开销 $O(Nd)$ 内存。 |
| 序列方向 attention / 标准 self-attention | S1 | 作用在 token 维度（同一层、不同位置）的注意力机制；本页仅作为类比，不展开。 |
| softmax | S2 | 把任意实数分数转换成一组正数并让和恰好为 1 的函数 $p_i=e^{z_i}/\sum_j e^{z_j}$；本页用于把内积分数转成权重。 |
| RMSNorm | S2 | 按均方根归一化：$\mathrm{RMSNorm}(x)=x/(\|x\|_2/\sqrt d)$；本页用于 softmax kernel 中归一化 key，防止幅值大的层主导权重。 |
| pseudo-query / 伪查询 | S2 | AttnRes 中每层自带的可学习参数向量 $q_l=w_l\in\mathbb{R}^d$，与 key 内积决定权重；不依赖该层输入。 |
| softmax kernel | S2 | AttnRes 中衡量 pseudo-query 与 key 相似度的函数 $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$；输出正数供 softmax 归一化。 |
| key / 键 | S2 | AttnRes 中被检索的来源向量；Full AttnRes 中 $k_i=v_i=h_1$（$i=0$）或 $f_i(h_i)$（$1\le i\le l-1$）。 |
| value / 值 | S2 | AttnRes 中加权求和的来源向量；与 key 同源（$k_i=v_i$）。 |
| 权重 / attention 权重 / $\alpha_{i\to l}$ | S2 | 层 $l$ 对来源 $i$ 的 softmax 权重 $\alpha_{i\to l}=\phi(q_l,k_i)/\sum_j\phi(q_l,k_j)$。 |
| block / 块 | S3 | Block AttnRes 中 $S$ 层为一组的分组单位；$L$ 层分 $N$ 个 block。 |
| block 级表征 / 块级表征 / $b_n$ | S3 | block $n$ 内所有层输出的求和 $b_n=\sum_{j\in B_n}f_j(h_j)$；$b_0=h_1$ 为 embedding。 |
| partial sum / 部分和 / $b_n^i$ | S3 | block $n$ 内前 $i$ 层的求和 $b_n^i=\sum_{j\in B_n,\,j\le i}f_j(h_j)$；作为"当前流"参与块间 attention。 |
| 候选集合 / candidate set | S3 | Block AttnRes 中每层做 attention 的来源集合，按 Eq.(10) 取 $[b_0,\dots,b_{n-1}]$ 或 $[b_0,\dots,b_{n-1},b_n^{i-1}]$。 |
| 当前流 / current stream | S3 | 当前 block 的 partial sum $b_n^{i-1}$；与历史块快照并列作为候选来源。 |
| 历史块快照 / block snapshot | S4 | K3 中在块边界层（`layer_idx % 12 == 0`）存入的 block 级表征；共 8 个（含 embedding）。 |
| partial block / 部分块 | S4 | K3 中最后一个 block（仅 9 层，不足 12 层）；与 7 个完整 block 共同构成 8 个 block。 |
| 加权三次 / 三次加权 | S4 | K3 中 AttnRes 加权的位置：每个 attention 子层前一次、每个 MLP 子层前一次、模型末尾 final norm 前一次。 |
| output AttnRes / 输出 AttnRes | S4 | K3 中模型末尾 final norm 前的第三次 AttnRes 加权，聚合所有 $N$ 个 block 表征。 |
| K3 / Kimi K3 | S1 | Moonshot AI 的 Kimi K3 模型，2.78T 参数 MoE，主干 93 层；本页概念的具体实例化对象。 |
| KDA / Kimi Delta Attention | S4 | K3 中的线性注意力模块；本页只在配置表中使用，不展开。 |
| Gated MLA / Gated Multi-Head Latent Attention | S4 | K3 中的全局注意力模块；本页只在配置表中使用，不展开。 |
| Stable LatentMoE | S4 | K3 中的稀疏混合专家 FFN；本页中作为"MLP 子层"的同义词使用。 |
| MLP 子层 | S4 | K3 中每个 attention 子层后的 Stable LatentMoE 子层；AttnRes 在其前加权一次。 |
| attention 子层 | S4 | K3 中每个 decoder layer 的 KDA 或 Gated MLA 子层；AttnRes 在其前加权一次。 |
| `attn_res_block_size` | S4 | K3 官方 `config.json` 中的字段，值为 12，表示 Block AttnRes 的 block 大小（每 block 层数）。 |
| `config.json` | S4 | HuggingFace 上 `moonshotai/Kimi-K3` 仓库的官方配置文件；本页事实来源之一。 |

## 符号

| 符号 | 首次出现 | 定义或含义 |
|---|---|---|
| $L$ | S1 | 网络总层数（K3 中 $L=93$）。 |
| $N$ | S1 | Block AttnRes 的 block 数（K3 中 $N=8$）。 |
| $S$ | S1 | 每个 block 的层数，$S=L/N$（K3 中 $S=12$）。 |
| $d$ | S1 | hidden 维度（K3 中 $d=7168$）。 |
| $h_l$ | S1 | 第 $l$ 层的残差流（层 $l$ 的输入）；标准残差中 $h_{l+1}=h_l+F_l(h_l)$。 |
| $h_1$ | S2 | token embedding，AttnRes 中作为 $i=0$ 的 key/value（$b_0=h_1$）。 |
| $f_l(h_l)$ | S2 | 第 $l$ 层的输出（层 $l$ 对 $h_l$ 做变换后的结果，在 K3 中是 attention 或 MLP 子层的输出）。 |
| $F_l$ | S1 | 第 $l$ 层的非线性变换函数（标准残差记号，来自前置概念页）。 |
| $q_l$ | S2 | 层 $l$ 的 pseudo-query，$q_l=w_l\in\mathbb{R}^d$，层自带可学习参数。 |
| $w_l$ | S2 | 层 $l$ 的可学习参数向量，$q_l=w_l$。 |
| $k_i$ | S2 | 第 $i$ 个来源的 key，Full AttnRes 中 $k_i=v_i$。 |
| $v_i$ | S2 | 第 $i$ 个来源的 value，Full AttnRes 中 $k_i=v_i$。 |
| $\phi(q,k)$ | S2 | softmax kernel，$\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$。 |
| $\alpha_{i\to l}$ | S2 | 层 $l$ 对来源 $i$ 的 attention 权重。 |
| $\mathrm{RMSNorm}(x)$ | S2 | $x$ 按均方根归一化，$\mathrm{RMSNorm}(x)\approx x/(\|x\|_2/\sqrt d)$。 |
| $\|x\|_2$ | S2 | 向量 $x$ 的 $\ell_2$ 范数。 |
| $B_n$ | S3 | block $n$ 的层索引集合。 |
| $b_n$ | S3 | block $n$ 的完整求和 $b_n=\sum_{j\in B_n}f_j(h_j)$；$b_0=h_1$。 |
| $b_n^i$ | S3 | block $n$ 内前 $i$ 层的 partial sum $b_n^i=\sum_{j\in B_n,\,j\le i}f_j(h_j)$。 |
| $i$ | S3 | block $n$ 内的层序号（$i=1$ 为 block 第一层，$i\ge 2$ 为后续层）。 |
| $n$ | S3 | block 索引（$n=0$ 为 embedding block，$n=1,\dots,N$ 为实际 block）。 |
| $O(Ld)$ / $O(Nd)$ | S3 | 内存复杂度记号；$L$ 为层数，$N$ 为 block 数，$d$ 为 hidden 维度。 |
| `layer_idx` | S4 | K3 实现中的层索引（0-based）；块边界层满足 `layer_idx % 12 == 0`。 |
