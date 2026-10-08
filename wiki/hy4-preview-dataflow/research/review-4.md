# 第 4 轮

源码位置核查通过：279 个节点。有结论的席位 5 个。这一轮有事实错误，不通过。

## 审查者 1（路径覆盖）

- 少画｜HYV4DecoderLayer.forward：attn；HYV4Attention.forward：idx｜shared 层的 `topk_indices = prev_topk_indices` 是路径上确实执行的一步，只在文字里提了一句，没有节点和连线｜modeling_hy_v4.py:443-446、688、705｜补 prev_topk_indices 节点与连线
- 其余四项（每步都画、没多画、if/for 不画节点、自建类都有图）无

## 审查者 2（节点与形状）

- 事实错误｜HYV4Attention.forward：am、amask；HYV4DecoderLayer.forward：am｜掩码最后一维写 T，与 keys、vals、im、amk 的 L 不一致；masked_fill 也不会把最后一维从 T 变 L｜modeling_hy_v4.py:451-453、459｜掩码最后一维一律写 L
- 事实错误｜HYV4TopkRouter.forward：corr｜把两个操作数都写成 [B×T, 256]，实际第二个是 [256]，靠广播｜modeling_hy_v4.py:517｜改成 [B×T, 256] 与 [256]
- 写法｜HYV4Model.forward：am｜第一行写 causal_mask_mapping，但它是 dict｜modeling_hy_v4.py:835、846｜改成 causal_mask_mapping["indexed_attention"]
- 写法｜HYV4MoE.forward：hsh｜第一行写 hidden_states，但共享专家返回值没有变量名，且同图三处都叫 hidden_states｜modeling_hy_v4.py:599｜改 self.shared_experts(residuals)
- 写法｜HYV4Model.forward：pos｜第一行少了 `.unsqueeze(0)`，源码标与 pid 不一致｜modeling_hy_v4.py:822-823｜补上
- 写法｜HYV4RMSNorm.forward：hs 等｜形状一律写 6144，但 q_a_layernorm、kv_a_layernorm 点开时是 2048、512｜modeling_hy_v4.py:347、355、430｜注明维度随调用点变
- 写法｜HYV4Attention.forward：cache；HYV4Experts.forward：where｜L、m、n 首次出现在算子第二行，算子第三行按规则必须为空｜modeling_hy_v4.py:427、565、560｜在前提或面板里先写清
- 写法｜HYV4HyperConnection.forward 面板｜「折叠」行和「hc_mult」行挤成同一行，表格断裂｜计划｜拆成两行

## 审查者 3（连线与结构）

- 事实错误｜HYV4MLP 图：mul、down｜x → mul、x → down 说这两个算子取用 x，实际 mul 的输入是 av、uv，down 的输入是 mv｜modeling_hy_v4.py:496｜删掉这两条
- 事实错误｜HYV4DecoderLayer：attn；HYV4Attention｜prev_topk_indices 没有节点和连线｜modeling_hy_v4.py:688、705、444-446｜同审查者 1
- 事实错误｜HYV4Experts：eup、edown｜两者用 expert_idx 切专家片，没有连线｜modeling_hy_v4.py:559-567｜补 hitn → eup、hitn → edown
- 写法｜HYV4Indexer：wp｜hs → wp 标「形状与 dtype」，但它是数据输入｜modeling_hy_v4.py:244-248｜改成「输入数据，权重另取 dtype」
- 写法｜HYV4Model：rope｜h0 → rope 漏标注 device 与 dtype｜modeling_hy_v4.py:838｜补上
- 写法｜HYV4RMSNorm：dt｜hs → dt 标「dtype」，但它是数据转 fp32｜modeling_hy_v4.py:57｜改成「数据转 fp32」
- 写法｜HYV4Attention：attn｜attn_weights 不取，折叠说明里没写｜modeling_hy_v4.py:466、481｜补一句
- 缓存、点开目标、节点 id、交替性四项无

## 审查者 4（公式与面板）

- 措辞｜HYV4HyperConnection.forward 面板｜折叠行与 hc_mult 行挤成一行，表格断裂｜计划｜拆成两行
- 措辞｜HYV4DecoderLayer：ahc、fhc｜公式把被调类 HYV4HyperConnection 的内部算式整段写了进来，别的被调类只写调用本身｜modeling_hy_v4.py:695、711、621-640｜改成只写调用
- 措辞｜HYV4DecoderLayer：ahc 说明｜把 input_norm 的 eps 叫成 variance_epsilon，属性名其实是 self.eps｜modeling_hy_v4.py:71-73、53｜改成 self.eps
- 其余（公式符号、数值）无

## 审查者 5（读者与措辞）

- 排版｜HYV4HyperConnection.forward 面板｜同审查者 4｜同审查者 4｜同审查者 4
- 术语｜HYV4HyperConnection：ahc｜同一 eps 两行里两个名字｜计划｜统一
- 符号未定义｜HYV4Model 面板「末尾折叠」｜c、pre_c、x_c 没定义｜计划｜写明并统一名称
- 指代｜HYV4MLP：down 说明｜「这个图的 add」指代不明，本页没有 add｜计划｜改成「HYV4MoE 图的 add」
- 前后不一致｜HYV4Attention 面板「稀疏注意力」｜说 mask 只在 Top-2048 外加极小值，与 amask 折叠不符｜计划｜改成在因果掩码基础上再稀疏
- 符号未定义｜HYV4TopkRouter 面板｜公式用 x，本页叫 hidden_states｜计划｜改成 hidden_states

## 修改

- 面板：修好 HyperConnection 被挤成一行的那条表格；Attention 的 mask 说明改准；TopkRouter 的 x 改 hidden_states；Model 的末尾折叠补 c、pre_c、x_c 的定义；MLP 的 down 说明改成「HYV4MoE 图的 add」。
- 节点：掩码最后一维统一写 L；TopkRouter 的 corr 改成带广播的形状；Model 的 am 第一行改下标形式、pos 补 `.unsqueeze(0)`、源码列统一；MoE 的 hsh 改表达式；RMSNorm 的 hs 注明维度随调用点变、面板加一行说明；Indexer 的 wp 边注改准。
- 连线：MLP 删掉 x → mul、x → down；Experts 补 hitn → eup、hitn → edown；Model 的 h0 → rope 补标注；RMSNorm 的 hs → dt 改注。
- 补画：Decoder 与 Attention 图各补一个 prev_topk_indices 张量节点（Decoder 的 ptk 接 attn，Attention 的 ptk 接 tkidx），与 tk 区分开——tk 是本层算出的，ptk 是上一层传来的。
- 折叠：Decoder 的 ahc、fhc 改成只写调用；关注 attention 的 attn_weights 说明。
