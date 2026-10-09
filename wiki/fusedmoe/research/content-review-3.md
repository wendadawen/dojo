# 正文审查 第 3 轮

## 事实（A）

- 推不出｜长页《一个专家在算什么》｜「第一层的输出里前半叫门、后半叫值，激活函数作用在前半（门）上」——标注的 `fused_moe/activation.py L47-L54` 写的是 `gate*activation(up)`，即激活作用在**后半（up）**，与页面相反｜该处原文：「Returns True if activation expects gate*activation(up) pattern. Gated activations expect input tensor with 2x the output size, where the first half is the gate and second half is the up projection.」；写「激活作用在前半」的确切公式在另一个文件 `layers/activation.py` 的 `SiluAndMul` docstring（L119：`x -> silu(x[:d]) * x[d:]`）｜改标注：改引 `repo/vllm/vllm/model_executor/layers/activation.py L116-L119`（句子本身与源码一致，不必改字）。
- 位置标错／推不出｜长页《累积与量化》｜「第二层之前的中间结果也压一次」的依据已在第 2 轮补为 L1815-L1821，本轮复核通过，不再计。其余各段（含偏置、掩码、调优表）经复核与标注一致。

## 事实（B）

- 推不出｜长页《两种输入激活格式》｜「训练和推理里」这一领域限定不在标注里｜blog L15-L27 与 moe_kernel_features L11 只讲输入激活格式取决于 All2All Dispatch、两种格式的形状，无训练/推理之分｜删「训练和」，或补依据。
- 推不出｜长页《适用条件与边界》｜原因句「排在前面几轮的对齐本身有开销，而每块只用一次权重」与后果句「每个分块对应一路（一个 token 的一次专家选择），块内只有一行有效，权重按这一路选中的专家取」都不在标注里｜L1554-L1566 只写条件 `num_tokens * top_k * 4 <= global_num_experts`、注释「SPARSITY_FACTOR is a heuristic margin……」、「Skips moe_align_block_size and activates the sorted_token_ids is None path」；「每块一行有效／按该路专家取权重」在 L374-L376（naive_block_assignment：each block corresponds to a single token）、L409-L414｜删去无依据的原因句，或补依据到 L374-L376、L409-L414。

## 复算

无。复算了：短页与长页的每专家平均 $Mk/E$ 行（$64\times2/64=2$）；对齐例子（桶大小 $2,1,1$、补齐后 $6$、`sorted_token_ids=[0,2,1,4,3,4]`、`expert_ids=[0,1,2]`、`num_tokens_post_padded=6`）；补齐上界（$4+3\times1=7$）；权重分段（每专家 $I\times H=4$ 个元素，起点 $4e$）；各次 GEMM 形状（$(M,H)\cdot(2I,H)\to(M,2I)$，激活减半为 $(M,I)$，$(M,I)\cdot(H,I)\to(M,H)$）。第 2 轮把隐藏维/中间维记号改为 $H$／$I$ 后，形状推导仍自洽，未发现算错。

## 文风

- AI 味｜长页·开头段「$M$ 小、$E$ 大时这个行数小得可怜。」｜「小得可怜」是情绪化的评价词，没给出量｜例子库第 7 类｜删掉，行数小这件事由下一句的具体数字承担。
- AI 味｜长页·开头段「方阵乘起来才摊得满的硬件，遇到这种长条矩阵会把时间花在把权重搬进显存，算力空转。」｜「算力空转」是对前半句的评价式复述｜例子库第 7 类｜删掉。

## 读者

- 读不通｜长页·《一个专家在算什么》｜「第一层输出的维度取两倍中间维」与上一段「先把 $H$ 维升到中间维 $I$」、第二层输入 $I/2$ 三处的「中间维」对不上｜缺一个含义唯一、前后一致的「中间维」定义｜给第一层输出维和激活后的维各起一个固定名字。
- 读不通｜长页·《moe_align_block_size》末段与《对齐后的分组 GEMM》｜上一节的「分块」是按专家对齐的整段行、由 `expert_ids` 逐个索引；这一节又把「分块」写成行列两方向切出的输出小块，读不出 `expert_ids` 对应哪一种｜缺一句说明｜把「按专家对齐的行块」和「GEMM 的输出小块」用两个不同的词。
- 读不通｜长页·导语「第一批算的是 $M$ 个 token……」｜前面在讲「每个组合各自调用一次矩阵乘」，突然冒出「第一批」｜缺先行词｜直接写明是第一次矩阵乘。

## 处理

- 事实 A1：采纳。标注改引 `repo/vllm/vllm/model_executor/layers/activation.py L116-L119`（`SiluAndMul` 给出 `silu(x[:d]) * x[d:]` 的确切公式）；句子不改（与源码一致）。
- 事实 B1：采纳。删「训练和」，改为「token 按专家凑到一起这一步叫 All2All 分发」。
- 事实 B2：采纳。删去「每块只用一次权重」这一无依据原因，改为「它换来的『一次算很多行』并没有用到」；后果句补依据 L374-L376、L411-L414。
- 文风 2 条：全部采纳。「小得可怜」改「很小」；「算力空转」删，改为「时间会花在把权重搬进显存，而不是花在乘加上」。
- 读者 1：采纳。记号改为：隐藏维 $H$，激活后的中间维 $I$，第一层输出宽 $2I$（激活后 $I$，第二层权重 $(E,H,I)$）；全文统一。
- 读者 2：采纳。把按专家对齐的整段行改称「行块」，GEMM 里按行列切出的小块仍叫「分块」，并点明「`expert_ids` 按行块索引，同一个行块细分出的几个分块共用同一个专家」。
- 读者 3：采纳。删「第一批」，改为「这一次有 $M$ 个 token」。
