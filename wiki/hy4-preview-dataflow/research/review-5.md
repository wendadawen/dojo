# 第 5 轮

源码位置核查通过：279 个节点。有结论的席位 5 个。5 席都没有事实错误；有写法与措辞问题，改完通过。

## 审查者 1（路径覆盖）

- 版本口径（同一份计划的早期副本）｜hyv4model/idx、hyv4attention/ptk→tkidx｜shared 层的 `topk_indices = prev_topk_indices` 没有可见节点｜modeling_hy_v4.py:443-446｜已在两图各补 prev_topk_indices 张量（ptk），Attention 图连到 idx
- 其余四项（每步都画、没多画、if/for 不画节点、自建类都有图、公式无前提外分支）无

## 审查者 2（节点与形状）

- 排版｜HYV4HyperConnection.forward 面板｜公式行与配置行挤成一行，表格断裂｜计划第 991 行｜拆成两行
- 写法｜HYV4Attention.forward：idx｜写「[B, T, T] 与 [B, T, L] 却都写 64」，实际两条都是 64；切片的「下三角、其余极小值」只为对应浅色线｜modeling_hy_v4.py:399、459｜两条统一写 [B, T, L]
- 写法｜HYV4Attention.forward：imask｜「箭头两边都是形状」未满足，左边只有输入形状、没有 key_states｜modeling_hy_v4.py:451-453｜左边补 keys 的形状
- 写法｜HYV4Model.forward：tk｜「随后的 shared 层经 prev_topk_indices 复用」写进 tk 的说明，与 ptk 分工不清｜modeling_hy_v4.py:844｜tk 只讲本层产出，ptk 讲入口
- 其余写法（算子第三行、形状箭头、类型、交替、缓存节点）无

## 审查者 3（连线与结构）

- 事实错误｜（同一份计划早期副本）HYV4Model.forward：shp 连到 hs｜——本轮当前版本已无此线，不成立
- 写法｜HYV4Model.forward：hmm，hsh 未连线｜charcs 已连线｜不成立
- 写法｜HYV4Model.forward：am 第一行写成 dict 名｜modeling_hy_v4.py:835、846｜改成 causal_mask_mapping["indexed_attention"]
- 其余（缓存、多输出、口开目标、id 唯一）无

## 审查者 4（公式与面板）

- 措辞｜hyv4decoderlayer：ahc、fhc｜公式把被调类 HYV4HyperConnection 的内部算式整段写了进来，与本计划其它被调类只写调用本身不一致｜modeling_hy_v4.py:695、711｜改成只写调用
- 措辞｜hyv4decoderlayer：ahc｜说明里把 input_norm 的 self.eps 叫成 variance_epsilon｜modeling_hy_v4.py:614、71-73｜改成 self.eps / rms_norm_eps
- 其余（算子自带公式、算子/张量折叠、面板配置、数值）无

## 审查者 5（读者与措辞）

- 措辞｜HYV4Model：tk 与 ptk｜两行互相指回 ly，读不出这是循环的出口与入口｜计划第 145、146 行｜各补一句「相对 ly 这个循环，出口/入口」
- 措辞｜HYV4DecoderLayer：attn 说明｜「shared 层不用自己算的 topk」会被读成 shared 层自己算了｜计划｜改成「shared 层不跑索引器，直接用 ptk 传来的 topk」
- 公式｜HYV4MoE：gate｜公式行混入了元描述｜计划｜公式只留算式，元描述并入说明
- 措辞｜HYV4MLP：out｜第一行是算子名 down_proj，会被当成算子｜计划｜第三行注明是返回值
- 措辞｜HYV4DecoderLayer：ahc 说明｜说明里的 out 在本图没有对应节点｜计划｜改用本图节点名 hsa
- 公式/配置｜HYV4RotaryEmbedding 面板与 config 表 head_dim｜同时列 qk_rope_head_dim 与 head_dim，读者分不清 d 取哪一个｜计划｜在 fr 说明与 config 表注明 head_dim 就是 qk_rope_head_dim
- 措辞｜HYV4Model：am｜L 没在本图解释｜计划｜在 am 第三行补 L 的含义
- 缩写｜介绍｜iHC、DSA 已在描述/摘要解释，无
- 一致性｜HYV4ForCausalLM：model 节点与折叠｜返回是一个输出还是元组，前后不一｜计划｜节点第二行补第二个输出，折叠与说明统一
- 一致性｜介绍｜「残差流」与「残差通路」两种叫法｜计划｜首次出现处点明是同一件事
- 一致性｜config 表 use_dsa 与摘要｜DSA 的展开两处不同｜计划｜统一

## 修改

- 面板/表格：修好 HyperConnection 被挤成一行的那条；Attention 的 am、amask、idx 掩码统一写 [B, T, L]；imask 左边补 keys 的形状；Model 的 am 改下标形式并补 L 的含义。
- 折叠：Decoder 的 ahc、fhc、Model 的 hch、mlp 都改成只写调用；MoE 的 gate 公式去元描述、add 与 shp 改用同图节点名；MLP 的 out 第三行注明是返回值；Decoder 的 ahc 说明改用 hsa。
- 措辞：Model 的 tk 与 ptk 各补一句循环出口/入口；Decoder 的 attn 说明改成「shared 层不跑索引器」；介绍把「残差通路」与「残差流」点明为同一件事；config 表 head_dim 与 fr 说明点明 d 就是 head_dim；use_dsa 展开与摘要统一。
- 这样改下来，5 个席位提出的都是写法与措辞，没有事实错误。改完重跑 `build_page.py` 与 `check_source.py` 都通过；第 5 轮之后不再改动事实内容，余下的机械核查在 `guides/dataflow/check.md` 下另记。

## 仲裁（轮数上限后核对最后一批改动）

第 5 轮是轮数上限。之后的改动交独立子代理核对，只答有没有引入源码里没有的内容、漏步或多步、接错线，有没有违背前几轮裁定。

核对结果：10 类改动里 7 类通过，3 处必须改：

- Attention 图 `ptk → idx` 接错线：`prev_topk_indices` 从没传进 `self.indexer`（源码 444-446 是 `topk_indices = prev_topk_indices`）。改成 `ptk → tkidx`。
- Decoder 图 `mlp` 公式 `mo, tk = mlp(hsfn)` 引入源码没有的第二返回值：HYV4MoE.forward 只返回一个张量，`tk` 来自 self_attn。改成 `mo = mlp(hsfn)`。
- Experts 图新增的 `hfl → edown：取行` 源码里不存在：edown（源码 567）只用 current 与 down_proj[expert_idx]。删掉。

作者改完后再交仲裁复改，三处全部通过；无违背前几轮裁定之处（`lret`/`ret`/`outs` 的「—」、不画 pid 节点、`mlp` 只点开 HYV4MoE、参数张量处理、config 表 `num_key_value_heads` 都未被动到）。计划通过。

## 生成器改版后的补记

第 5 轮之后、交付之前，`build_page.py` 被改成在生成时也核两条：张量第二行必须写数据类型和形状（返回聚合节点也不例外），不同图里同一个 id 的形状必须一致。示范页 `wiki/deepseek-v4-1-dataflow/` 同时改成返回节点用 `×` 连写各分量形状、跨图同名 id 各自独立。

作者照新版重跑，改了两类：

- 返回聚合节点补齐分量形状：`lret`、`ret`、`outs`（两处）、`hret` 的第二行改成 `fp32 [B, T, 120832]`、`bf16 [B, T, 4, 6144] × int32 [B, T, 2048]` 这种写法。
- 跨图同名但形状不同的 id 改名：`ao→aoh`、`qr2→iqr2`、`kr2→ikr2`、`am→iam`（索引器图）、`hf→hfr`、`gs→grps`（TopkRouter 图）、`x→mx0`（MLP 图）、`hfl→hhfl`（HyperHead 图）、`hf→rmhf`（RMSNorm 图）、`x/mv/ov→ux/umv/uov`（UnweightedRMSNorm 图）、`x→rex`（RotaryEmbedding 图）、`outs→routs`（TopkRouter 图）、`ret→hret`（HyperConnection 图）。

改完 `build_page.py`、`check_source.py`（计划与页面两遍）、`site/check.py`、几何自检全部通过。


