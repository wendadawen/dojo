# 第 3 轮

源码位置核查通过：276 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1（路径覆盖）

- 措辞｜HYV4Attention.forward：attn｜`sparse_indices` 这一步没画也没交代｜modeling_hy_v4.py:448、474｜在 attn 的折叠说明里写清 indices 就是 sparse_indices、这条路为 None
- 事实错误｜HYV4MoE.forward：add；HYV4MLP.forward：down；HYV4HyperConnection.forward：mix；HYV4HyperHead.forward：hmix｜同一个源码调用被写了两处，折叠又展开了一遍别处已画出的算子｜modeling_hy_v4.py:599、496、628、661｜折叠只引用同图已画出的张量
- 其余三项（没多画、if/for 没画成节点、自建类都有图）无

## 审查者 2（节点与形状）

- 事实错误｜HYV4Experts.forward：edown｜第二行右边写 bf16，与同一行算出的 cur2（fp32）、iadd 的输入（fp32）矛盾｜modeling_hy_v4.py:567-568｜右边改 fp32 [m, 6144]
- 写法｜HYV4Experts.forward：hit｜第二行左边写了 mask.sum 之后的形状，不是连进来的 mk 的形状｜modeling_hy_v4.py:559-560｜改 int64 [256 + 1, 8, B×T]
- 写法｜HYV4Indexer.forward：tki｜第一行写源码里不存在的变量名｜modeling_hy_v4.py:257-258｜改用产生它的表达式
- 写法｜HYV4Attention.forward：idx｜第二行左边把切片前的掩码形状写上了｜modeling_hy_v4.py:439｜改 bf16 [B, T, T]
- 写法｜HYV4Experts.forward：pos｜字母 m 首次出现时第三行没写含义｜plan.md:81｜写到 pos 的第三行
- 写法｜HYV4RotaryEmbedding.forward：cos、sin｜第一行漏了末尾 dtype 转换｜modeling_hy_v4.py:117-118、120｜补 `.to(...)`
- 写法｜HYV4Experts.forward：tw｜源码列指 559，该行没有这个参数｜modeling_hy_v4.py:555｜改指 555

## 审查者 3（连线与结构）

- 事实错误｜HYV4Experts.forward：edown｜同审查者 2｜同审查者 2｜同审查者 2
- 事实错误｜HYV4MoE.forward 连线 hfl → exp：形状与 dtype｜与上一行重复，且 exp 用的是展平前的形状｜modeling_hy_v4.py:594-598｜改成 hs → exp：形状（orig_shape）
- 事实错误｜HYV4RotaryEmbedding.forward 连线｜cos/sin 只取 dtype，device 是 fr 用的｜modeling_hy_v4.py:115、117-118｜device 挪到 fr
- 写法｜全篇｜同为「算子用到的参数张量」，bias、base、scale、sinks 画成了节点，fn、hc_fn、weight 等只在折叠里交代，两套画法｜modeling_hy_v4.py:511、616｜见「修改」里的说明

## 审查者 4（公式与面板）

- 措辞｜HYV4Model.forward：nrm｜公式里 h 与同图节点 h 同名不同量｜modeling_hy_v4.py:55-60｜改用 hidden_states
- 措辞｜多图｜符号 h 不是源码变量名｜modeling_hy_v4.py:939、60｜统一写 hidden_states
- 措辞｜HYV4DecoderLayer.forward：adda、addf、面板｜post 同图两处指不同量｜modeling_hy_v4.py:695、711｜addf 注明 post 来自 fhc
- 措辞｜HYV4HyperConnection.forward：post；DecoderLayer：ahc、fhc｜公式用 hc_magnitude，源码名是 hc_post_magnitude｜modeling_hy_v4.py:637｜按源码写
- 措辞｜HYV4Model：rope、hch、nrm；DecoderLayer：aln、fln｜公式把被调类的内部算式抄了进来｜modeling_hy_v4.py:114-120、655-664、55-60｜只写调用本身
- 措辞｜HYV4Model：hch；HyperConnection：norm、mix；HyperHead：hnorm、hmix｜用 variance_epsilon/hc_eps，代码里都是 self.eps，与别处不统一｜modeling_hy_v4.py:71-73、662｜统一或写明
- 措辞｜HYV4MoE.forward 面板「混合专家」｜用符号 x，源码里没有｜modeling_hy_v4.py:593-600｜改 hidden_states
- 措辞｜HYV4Experts.forward：eup｜「两个 clamp 的上界都是 swiglu_limit」与说明不一致｜modeling_hy_v4.py:573-574｜改成 gate 只夹上界、up 上下都夹

## 审查者 5（读者与措辞）

- 措辞｜HYV4Experts.forward：edown/cur2｜dtype 互相打架｜计划｜同审查者 2
- 措辞｜HYV4Experts.forward：mask｜「承接 one_hot 对齐用的 0」指代不清｜计划｜换成更清楚的说法
- 措辞｜HYV4Experts.forward：where｜expert_idx 突然出现，没说它是循环变量｜modeling_hy_v4.py:561｜在折叠说明里补
- 措辞｜HYV4Attention.forward：expkv｜448 只出现不解释｜计划｜改「448 = 192 + 256」
- 措辞｜HYV4Indexer.forward：am｜同一件加性掩码在两图形状不同，L 没定义｜计划｜第三行补切片与 L 的含义
- 措辞｜config 表：use_mla｜MLA 没给全称｜计划｜补 Multi-head Latent Attention
- 措辞｜config 表：learnable_sink｜sink 没给上下文｜计划｜补说明
- 措辞｜摘要｜漏掉旋转键那 64 维｜计划第 4 行｜补上
- 措辞｜HYV4RotaryEmbedding 面板｜公式跳过 cat 那步｜计划｜改写
- 措辞｜HYV4MoE：shp 与 add｜同一个调用出现两次，像算了两遍｜计划｜add 说明里点明就是 shp
- 措辞｜前提｜「之后每 4 层」起点不清｜计划第 21 行｜改成「第 5 层起每 4 层」

## 修改

- 节点：Experts 的 edown 改 fp32、hit 左边改 mk 的形状、pos 写 m 的含义、tw 源码列改 555；Indexer 的 tki 改用表达式；Attention 的 idx 左边改 [B, T, T]；RotaryEmbedding 的 cos/sin 补 `.to(...)`；HyperConnection 的 out、HyperHead 的 hout 前面已补。
- 连线：MoE 的 exp 改连 hs（形状）；RotaryEmbedding 的 device 挪到 fr，cos/sin 只写 dtype；Experts 的 where 折叠补 expert_idx 的来历。
- 折叠：MoE 的 add 改成引用 hsh；MLP 的 down 改成引用 mv；HyperConnection 的 mix、HyperHead 的 hmix 保持展开式；Model 的 rope、hch、nrm 与 Decoder 的 aln、fln 改成只写调用；HyperConnection 的 post、Decoder 的 ahc/fhc 公式改用 hc_post_magnitude；Experts 的 mask 说明改清楚、eup 说明改准。
- 面板：config 表补 MLA 全称与 learnable_sink 说明；摘要补旋转键的 64 维；RotaryEmbedding 面板补 cat；MoE 面板的 x 改 hidden_states；Decoder 面板的 post 注明来自 fhc；前提改 indexer_types 的写法；Attention 面板的 mask 说明只留 eager。

不成立（写出理由）：

- 审查者 3 关于参数张量两套画法：按 plan.md，权重形状写在折叠的说明里；只有 e_score_correction_bias、base、scale、sinks 这几个直接参与计算的量画了节点，其余权重保持在折叠里，不再改。
