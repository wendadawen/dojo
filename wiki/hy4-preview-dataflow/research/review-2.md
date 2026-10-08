# 第 2 轮

源码位置核查通过：276 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1（路径覆盖）

- 事实错误（配置）｜config.json 表：num_key_value_heads｜写 64，config.json 里是 8；64 是 `__post_init__` 改写后的运行值｜sources/config.json；configuration_hy_v4.py:140｜照抄 8，并注明源码会改写成 64
- 路径覆盖五项：1 每一步都画了、2 没有多画、3 if/for 没画成节点、4 自建类都有图且用到它的节点都点开、5 公式没有前提外的分支。均「无」。

## 审查者 2（节点与形状）

- 事实错误｜HYV4TopkRouter.forward：hs、view｜输入写成展平后的 [B×T, 6144]，但 `self.gate(hidden_states)` 是在展平之前调的｜modeling_hy_v4.py:596-597、514｜hs 改 [B, T, 6144]，view 改 [B, T, 6144] → [B×T, 6144]
- 事实错误｜HYV4Experts.forward：cur2、iadd｜current 乘的是 fp32 的 top_k_weights，第 567 行末尾是 fp32，转回发生在 568 行｜modeling_hy_v4.py:567-568｜cur2 改 fp32 [m, 6144]，iadd 第三个输入改 fp32
- 写法｜HYV4Model.forward：tk｜第二行用了未定义的维度字母 topk，与别处的 2048 不统一｜config.json｜改 int32 [B, T, 2048]
- 写法｜HYV4Experts.forward：hit、hitn、where、pos、tok｜字母 n、m 首次出现在算子第二行，没在张量第三行写明含义｜plan.md:81｜hitn 第三行写 n 的含义，tok 第三行写 m 的含义
- 写法｜HYV4MLP.forward：gv、uv、av｜第一行写了源码里不存在的局部变量名 gate、up｜modeling_hy_v4.py:496｜改成 gate_proj(x)、up_proj(x)、act_fn(gate_proj(x))
- 写法｜HYV4HyperConnection / HyperHead / RotaryEmbedding：out、hout、cos、sin｜第一行漏了末尾的 `.to(...)`，第二行右边却写成转换后的 dtype｜modeling_hy_v4.py:638,640；663-664；117-118,120｜第一行补 `.to(...)`
- 写法｜HYV4Attention.forward：attn｜第二行左边用「等」省掉了其余输入形状｜modeling_hy_v4.py:466-477｜逐个列出
- 写法｜HYV4Indexer.forward：idx｜第二行左边用「等」｜modeling_hy_v4.py:435-442｜逐个列出

## 审查者 3（连线与结构）

- 事实错误｜HYV4Model.forward：ly；HYV4DecoderLayer.forward：attn；HYV4Attention.forward：idx｜position_ids 是实参却没有节点和连线｜modeling_hy_v4.py:848、701、440｜三张图补 pid 节点与连线（本页在 ly、attn 折叠里写明这个参数只为兼容保留、没有计算，idx 已补连线）
- 事实错误｜HYV4TopkRouter.forward：hs、view｜同审查者 2｜modeling_hy_v4.py:596-597｜同审查者 2
- 事实错误｜HYV4HyperConnection.forward：bsplit、ssplit｜base、scale 没有接进来，算子不可达｜modeling_hy_v4.py:631-632｜补 base、scale 张量节点与连线
- 写法｜HYV4MLP.forward：mul、down｜第一行照抄整条嵌套表达式，式子里的 x 没有连线｜modeling_hy_v4.py:496｜补 x → mul、x → down
- 写法｜HYV4HyperConnection.forward：mix；HyperHead.forward：hmix｜第一行出现 flat，没有连线｜modeling_hy_v4.py:628、661｜补 fl → mix、hfl → hmix
- 写法｜HYV4Experts.forward：zero｜hfl → zero 只用了形状与 dtype，没写｜modeling_hy_v4.py:557｜线上写「形状与 dtype」
- 写法｜跨图 id 复用｜一大批同名 id 在不同图里指不同量｜plan.md:86｜见「修改」的说明

## 审查者 4（公式与面板）

- 事实错误｜config.json 表：num_key_value_heads｜同审查者 1｜同审查者 1｜同审查者 1
- 措辞｜HYV4Model.forward：hch｜折叠公式里的 mixes 没说明是什么｜modeling_hy_v4.py:661｜式后补 mixes 的定义
- 措辞｜HYV4DecoderLayer.forward：fhc｜折叠「公式」只写了「同 attn_hc」一句，没写公式｜modeling_hy_v4.py:621-640｜照 ahc 写出完整式子
- 事实错误｜HYV4RMSNorm.forward 面板｜rms_norm_eps 记 1e-05，但 q_a_layernorm、kv_a_layernorm 没传 eps、用代码默认值 1e-6｜modeling_hy_v4.py:47、347、355、675-676、790｜在配置或说明里注明

## 审查者 5（读者与措辞）

- 措辞｜HYV4ForCausalLM 面板「介绍」｜「算到下一个 token 的 logits」与描述、logits 节点冲突｜计划第 3、91、119 行｜改成「算出全部位置的 logits」
- 措辞｜HYV4DecoderLayer 连线｜只写 mlp 点开 → HYV4MoE，与折叠矛盾｜计划第 280、296 行｜见「修改」
- 措辞｜HYV4Attention 连线｜rope 点开指向 HYV4RotaryEmbedding，但折叠说 rope 是普通函数｜计划第 431、452 行｜删掉这条点开
- 措辞｜前提｜「每 4 层一个 full，其余为 shared，第 0、1 层也是 full」读起来像第 0、4、8…层｜计划第 21、57 行｜改成「第 0、1 层与之后每 4 层（第 5、9…77 层）」
- 措辞｜全文｜用了 B 但没交代 B 是什么｜计划第 18 行｜前提加一句「B 是批大小」
- 措辞｜HYV4Model：tk｜topk 未定义｜计划第 145 行｜改 2048
- 措辞｜HYV4Model：hch｜mixes 没定义｜计划第 199 行｜同审查者 4
- 措辞｜HYV4HyperConnection：post｜源码名 hc_post_magnitude，折叠写 hc_magnitude｜计划第 923、971 行｜加注是同一个量
- 措辞｜HYV4Indexer：qsplit｜head_dim 同名不同值｜计划第 43、493、494 行｜注明索引器的 head_dim 是 index_head_dim
- 措辞｜config 表｜gated_mla、use_mla、gating_type 没给上下文｜计划第 46-49 行｜加注 MLA、DSA、elementwise 的含义
- 措辞｜HYV4UnweightedRMSNorm：out｜说「调用方再乘回 flat」，本图里这个量叫 x｜计划第 1126 行｜改写
- 措辞｜HYV4ForCausalLM：logits｜「slice(-0, None)」读起来像负数｜计划第 91 行｜写「slice(-0, None) 就是 slice(0, None)」

## 修改

- config 表：num_key_value_heads 改成 8 并注明源码会改写；use_mla、use_dsa、gated_mla、gating_type 加注；前提加 B 的含义、改 indexer_types 的写法。
- 节点：TopkRouter 的 hs/view 改 [B, T, 6144]；Experts 的 cur2 改 fp32、hitn/tok 第三行写 n、m；MLP 的 gv/uv/av 改表达式；HyperConnection 的 out、HyperHead 的 hout、RotaryEmbedding 的 cos/sin 第一行补 `.to(...)`；Attention 的 attn、Indexer 的 idx 第二行左侧列全形状；Model 的 tk 改 2048。
- 连线：HyperConnection 补 base、scale 两个张量节点与其出入线；MLP 补 x → mul、x → down；HyperConnection/HyperHead 补 fl/hfl → mix/hmix；Experts 的 hfl → zero 线上写「形状与 dtype」；Attention 删掉 rope 点开。
- 折叠：Model 的 hch 补 mixes 定义；Decoder 的 fhc 写出完整公式；HyperConnection 的 post 注明 hc_post_magnitude 即 hc_magnitude；Indexer 的 qsplit 注明 head_dim 是 index_head_dim；UnweightedRMSNorm 的 out 改写。
- 面板：第一张图介绍改成「算出全部位置的 logits」；RMSNorm 面板注明两处 layernorm 用代码默认 eps 1e-6；Decoder 的 mlp 说明改成写明两张图各自怎么点到。
- 跨图 id：本模块输入与上一层同名时按来源图统一，不再逐一改无关节点（图内查询侧的量本来就用 T；HyperHead 图已加 h 前缀与 HyperConnection 图区分；Experts 的 up/down 已改 eup/edown；Experts 与 TopkRouter 的 hs 改成与 MoE 图一致的输入名）。其余同名 id 都是只在各自图上出现、不会被读者跨图代入的量，暂不再改，读者按每张图自带的第二、三行读。

不成立的（写出理由）：

- 审查者 3 关于 position_ids 的连线：`position_ids` 在 `HYV4DecoderLayer.forward`、`HYV4Attention.forward` 里都只往下传、本层不参与计算，只在索引器里用；本页在 Decoder 的 attn、Attention 的 idx 折叠里写明，未给每张图都补一个 pid 节点，以免同一张图上出现两个语义相同的 int64 张量。
