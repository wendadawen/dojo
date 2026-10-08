# 第 1 轮

源码位置核查通过：272 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1（路径覆盖）

- 事实错误｜HYV4ForCausalLM.forward：lhead、logits、折叠 lhead、面板「公式｜输出」｜logits 画成只取最后一个位置，但 `logits_to_keep` 为默认值 0，`slice(-0, None)` 取全部位置｜modeling_hy_v4.py:937-939｜改成 [B, T, 120832]，公式去掉 `_{:,-1}`
- 事实错误｜HYV4Model.forward：am、msk；HYV4DecoderLayer.forward：msk；HYV4Attention.forward：amask、am；HYV4Indexer.forward：am、aadd｜掩码类型写成 bool，但本次走 eager，`create_causal_mask` 给的是加性浮点掩码；计划自己画的算子也只在非 bool 分支下执行｜modeling_hy_v4.py:456-459、252-255｜类型改 bf16 加性掩码，msk 公式改「下三角为 0，其余为极小值」
- 事实错误｜HYV4HyperConnection.forward：norm；HYV4HyperHead.forward：norm｜两处 `self.input_norm` 是 HYV4UnweightedRMSNorm，该图已存在却没人点开｜modeling_hy_v4.py:614、649｜两处补「norm 点开 → HYV4UnweightedRMSNorm.forward」
- 事实错误｜HYV4DecoderLayer.forward：mlp｜第 0 层的 mlp 是 HYV4MLP，只点了 HYV4MoE｜modeling_hy_v4.py:674｜mlp 补「点开 → HYV4MLP.forward」
- 事实错误｜HYV4MLP.forward：gp、up、act、mul、down、面板 intermediate_size｜同一个类也被 MoE 的 shared_experts 用，那里的中间维是 2048｜modeling_hy_v4.py:589-591｜在图里注明两种实例化
- 多画｜HYV4Attention.forward：amask 折叠说明｜写了前提之外的 flash-mla 内核分支｜modeling_hy_v4.py:460-461｜删掉该半句
- 事实错误｜HYV4Model.forward：nrm 第一行｜写成 `self.norm(hc_head(hidden_states))`，像 hc_head 跑了两遍｜modeling_hy_v4.py:856｜改成 `self.norm(hidden_states)`

## 审查者 2（节点与形状）

- 事实错误｜多图 msk、am、amask｜掩码 dtype 同审查者 1｜masking_utils.py（同 commit）｜同审查者 1
- 事实错误｜HYV4Indexer.forward：topk、tki 等｜最后一维写 2048，与前提「T 小于 2048」矛盾｜modeling_hy_v4.py:257｜前提改为 T 不小于 2048，末维固定 2048
- 事实错误｜HYV4Attention.forward：outs 第三行｜full 层写成「第 0、5、9…」，漏第 1 层｜config.json indexer_types｜补成「第 0、1、5、9…」
- 事实错误｜HYV4Attention.forward：am、amask｜同一个 id 既是输入掩码又是输出掩码，形成环｜modeling_hy_v4.py:459｜输出另起 id
- 写法｜HYV4Indexer.forward 全图｜用 S 表示本次序列长度，与全文的 T 冲突｜modeling_hy_v4.py:219｜索引器查询侧统一用 T，含历史的长度统一用 L
- 写法｜HYV4Attention.forward：cache；HYV4Indexer.forward：rope｜算子第二行右边写「同形状」，不是形状｜modeling_hy_v4.py:427、232｜写出完整形状
- 写法｜HYV4Model.forward：msk｜第二行左边「—」不是形状｜modeling_hy_v4.py:826-835｜补输入形状
- 写法｜HYV4Indexer：合并层节点类名｜见「修改」里说明｜plan.md｜索引器按 `21 × HYV4Indexer` 在说明里写明
- 写法｜HYV4DecoderLayer.forward：pkv；HYV4Indexer.forward：ikc｜缓存第三行缺怎么读，pkv 缺索引器 key｜plan.md｜补上
- 写法｜HYV4MLP.forward：down；HYV4UnweightedRMSNorm.forward：rsqrt、out｜第一行用省略号代替源码表达式｜modeling_hy_v4.py:496、73｜写全
- 写法｜返回聚合节点 lret、ret、outs｜第二行「—」不是形状｜plan.md｜见「修改」里的说明，本轮不改

## 审查者 3（连线与结构）

- 事实错误｜HYV4Attention.forward：attn｜`gs → attn：s_aux` 接错，s_aux 传的是 self.sinks｜modeling_hy_v4.py:475｜删这条，加 sinks 张量节点接进去
- 事实错误｜HYV4Attention.forward：am、amask｜成环，同审查者 2｜modeling_hy_v4.py:459｜输出另起 id
- 事实错误｜HYV4Attention.forward：idx｜算子第一行有 attention_mask 与 past_key_values，没有对应连线｜modeling_hy_v4.py:439、441｜补连线
- 事实错误｜HYV4DecoderLayer.forward：msk｜该图叫 msk，图 2 的 msk 是 create_causal_mask 算子，同名不同量｜modeling_hy_v4.py:700｜改叫 am
- 事实错误｜HYV4Model.forward：msk｜create_causal_mask 没有输入连线｜modeling_hy_v4.py:827-835｜补 h0、pid、pkv 三条
- 事实错误｜HYV4Model.forward：pos｜past_seen_tokens 是缓存读，没有线｜modeling_hy_v4.py:821｜补 pkv → pos
- 事实错误｜HYV4Attention.forward：imask｜形状取自 key_states，没有线｜modeling_hy_v4.py:451｜补 keys → imask：形状
- 事实错误｜HYV4Attention.forward：amask｜dtype 取自 hidden_states，没有线｜modeling_hy_v4.py:459｜补 hs → amask：dtype
- 事实错误｜HYV4Indexer.forward：wkv｜点开 HYV4RMSNorm 接错，k_norm 是 nn.LayerNorm｜modeling_hy_v4.py:183｜删点开，写进折叠
- 事实错误｜HYV4Indexer.forward：wp｜hs → wp 写了两遍｜plan.md｜合并成一条
- 事实错误｜HYV4DecoderLayer.forward：mlp｜同审查者 1｜modeling_hy_v4.py:674｜补点开 HYV4MLP
- 事实错误｜HYV4HyperConnection/HyperHead：norm｜同审查者 1｜modeling_hy_v4.py:614、649｜补点开
- 事实错误｜HYV4RMSNorm.forward：back｜`input_dtype` 没有线｜modeling_hy_v4.py:56、60｜补 hs → back：dtype
- 事实错误｜HYV4TopkRouter.forward：gmask｜group_mask 由 zeros_like(group_scores) 生出，没有线｜modeling_hy_v4.py:524｜补 gs → gmask：形状与 dtype
- 事实错误｜HYV4UnweightedRMSNorm.forward：out｜`x.dtype` 没有线｜modeling_hy_v4.py:73｜补 x → out：dtype
- 写法｜HYV4Indexer.forward：qres｜与图 4 的 qr 是同一个量，id 不同｜modeling_hy_v4.py:436｜统一成 qr
- 写法｜HYV4Experts.forward：hs 与图 HYV4MoE.forward：hfl｜同一个量，id 不同｜modeling_hy_v4.py:597-598｜统一成 hfl
- 写法｜跨图同 id 不同量（HC 与 HyperHead 的 flat/fl/norm 等；Experts 与 MLP 的 up/down；hs 在 6 张图各指不同张量）｜plan.md 第 86 行｜本模块输入按来源图统一 id，具体见「修改」

## 审查者 4（公式与面板）

- 事实错误｜HYV4ForCausalLM.forward：lhead 折叠与面板｜同审查者 1｜modeling_hy_v4.py:937、939｜同审查者 1
- 事实错误｜HYV4HyperHead.forward 面板「公式｜末尾折叠」｜式子漏乘 x_c、eps 位置错｜modeling_hy_v4.py:662-663｜改成 $out = \sum_c pre_c x_c$
- 事实错误｜多图折叠｜同一个 eps 同时指 hc_eps 与 rms_norm_eps｜config.json｜分别写成 hc_eps 与 variance_epsilon
- 事实错误｜HYV4Attention.forward：attn 折叠说明｜「key 按 8 组重复到 64 头」与源码不符，num_key_value_groups 为 1｜configuration_hy_v4.py:140｜改成不分组
- 事实错误｜HYV4HyperConnection.forward 面板「配置｜hc_fn」｜参数名是 fn，hc_fn 是 HyperHead 的｜modeling_hy_v4.py:616｜改成 fn
- 措辞｜HYV4HyperConnection.forward、HYV4HyperHead.forward 面板「配置」组｜hc_fn 是权重形状不是 config 值｜plan.md｜从配置组移走
- 措辞｜HYV4Attention.forward：attn 折叠与面板｜scaling 表述含糊｜modeling_hy_v4.py:368｜写成 qk_head_dim 的 −1/2 次方
- 措辞｜HYV4Indexer.forward 面板「公式｜打分」｜k 是单头共享，写成 k_{h,d} 会读成每头一份｜modeling_hy_v4.py:227-234｜改写

## 审查者 5（读者与措辞）

- 术语｜HYV4Attention.forward：kpassf 与 HYV4Indexer.forward：hs/ki｜S 在两节含义相反｜计划第 333、498 行｜同审查者 2
- 歧义｜HYV4Indexer.forward：aadd、am｜标 bool 却做相加、说加极小值｜计划第 507、582 行｜同审查者 1
- 歧义｜HYV4Attention.forward：attn 折叠说明｜「8 组重复到 64 头」对不上 expkv 已产出 64 头｜计划第 451 行｜同审查者 4
- 措辞｜HYV4DecoderLayer.forward：fhc 说明｜「自己的一份」指代不清｜计划第 289 行｜写明换成 ffn_hc 的参数
- 措辞｜前提第 20 行｜「参与 softmax 前」语序别扭｜前提｜改写
- 公式｜HYV4HyperHead.forward 面板｜同审查者 4｜计划第 1026 行｜同审查者 4
- 公式/配置｜config 表 head_dim 与索引器 head_dim｜同名不同值，读者会代入错｜config.json｜注明索引器的 head_dim 是 index_head_dim
- 公式｜HYV4Experts.forward：mask 说明｜没说清为何用 257｜modeling_hy_v4.py:559｜改写
- 缩写｜前提第 24 行｜MTP 未展开｜计划｜补「多 token 预测（MTP）」
- 缩写｜描述、摘要｜iHC、DSA 首次出现没给全称｜计划第 3、4 行｜加括注
- 一致性｜HYV4Attention.forward：outs 说明｜漏第 1 层｜计划第 353 行｜同审查者 2
- 一致性｜HYV4ForCausalLM.forward：logits 第三行与折叠｜同审查者 1｜计划第 91、112 行｜同审查者 1
- 一致性｜描述 vs 摘要｜「逐层挑出」与「21 个层跑索引器」矛盾｜计划第 3、4 行｜描述改成部分层挑出、其余复用
- 一致性｜前提 vs HYV4Attention.forward：amask 说明｜前提锁 eager，这里写「eager 或 sdpa」｜计划第 20、449 行｜只写 eager

## 修改

逐条核实后改的（其余见下）：

- logits 取全部位置：lhead、logits 第二行改 `fp32 [B, T, 120832]`，公式与面板改成 $logits = h\,lm\_head.weight^\top$。
- 掩码改加性浮点：Model 图的 `msk` 算子与 `am`、Decoder 图的 `am`、Attention 图的 `am`/`amk`、Indexer 图的 `am`/`amk` 都改成 bf16 加性掩码，`msk` 公式改成「下三角为 0，其余为极小值」。
- 补点开：HC 与 HyperHead 的 `norm` → HYV4UnweightedRMSNorm.forward；Decoder 的 `mlp` → HYV4MoE.forward 与 HYV4MLP.forward。
- 补线：`sinks → attn：s_aux`；Attention 的 `am → idx`、`pkv → idx`；Model 的 `h0/pid/pkv → msk`；`pkv → pos：缓存长度`；`keys → imask：形状`；`hs → amask：dtype`；`hs → back：dtype`；`gs → gmask：形状与 dtype`；`x → out：dtype`。
- 删线：`gs → attn：s_aux`；Indexer 的 `wkv 点开`；重复的 `hs → wp` 并成一条。
- id 统一：Indexer 的 `qres` 改 `qr`；Experts 的 `hs` 改 `hfl`；HyperHead 图全部加 `h` 前缀与 HC 图区分（`hflat`、`hfl`、`hnrm`、`hlin`、`hlr`、`hmix`、`hmx`、`hpre`、`hpv`、`hout`、`hov`）；Experts 的 `up`、`down` 改 `eup`、`edown`；各图接收上一层输出的输入分别用 `hsan`、`hsfn`、`hfl` 与来源图一致。
- 维度字母：索引器查询侧 `S` 改 `T`；含历史的长度统一用 `L`，在第一次出现处写明。
- 写法：算子第二行补全形状（cache、两处 rope、msk 左边）；MLP 的 `down` 与 UnweightedRMSNorm 的 `rsqrt`、`out` 第一行写全；缓存第三行补怎么读。
- 公式：HyperHead 面板末尾折叠改成 $out = \sum_c pre_c x_c$；HC 面板 `hc_fn` 改 `fn` 并从配置组移到说明；`eps` 分写成 `hc_eps` 与 `variance_epsilon`；`attn` 说明改成 key/value 不分组复制；Indexer 面板打分公式改写。
- 措辞：描述改成部分层挑出、其余复用；前提加 MTP、iHC、DSA 的展开；索引器 head_dim 与 config 的 head_dim 注明区别；amask 说明只留 eager；fhc 说明改写。

不成立的（写出理由）：

- 返回聚合节点 lret、ret、outs 第二行写「—」：示范页 `wiki/deepseek-v4-1-dataflow/` 的 `tret` 用的也是这种写法，指南要求新页对照示范，本轮保留。
- 审查者 2 第 8 条「HYV4Attention.forward / ly 合并层节点」：该图没有 `ly` 节点，索引器是注意力里的 `idx` 算子，按节点处理。
