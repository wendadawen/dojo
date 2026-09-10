var VIEWS = {
// ---------- 视图1:整体总览 ----------
overview: {
  title:'整体总览 (40 层)',
  rankSep:42, nodeSep:24,
  legend:[['#1f9d6b','输入/输出'],['#c0392b','编码器层(可下钻)'],['#7a45c9','解码器层 / 草稿头(可下钻)'],['#e0922f','压缩条目(可下钻)'],['#b7791f','全局 KV'],['#8a93a3','层段']],
  nodes:{
    emb:  {label:'embed\nid → 5120', t:'io', io:'token id [1,T] -> [1,T,5120]', f:'x = \\mathrm{Embedding}(\\mathrm{ids})', d:'词嵌入。vocab_size=129280、dim=5120；真实权重 embed.weight 形状 [129280, 5120] BF16。'},
    e0:   {label:'Layer 0\n纯窗口注意力', t:'enc', drill:'attn', io:'[1,T,5120] -> [1,T,5120]', f:'\\text{SWA only}', d:'层 0 的 compress_ratios 为 0：不生产也不复用全局 KV，只有滑动窗口注意力（窗口 128）。'},
    e1:   {label:'Layer 1\n窗口 + Engram', t:'enc', drill:'engram', io:'[1,T,5120] -> [1,T,5120]', f:'h \\leftarrow h + \\mathrm{Engram}(h)', d:'全模型两个 Engram 注入层之一（另一个是层 14）。24 行 n-gram 哈希查表结果经稠密投影写回残差流。'},
    e2:   {label:'Layer 2\nFull：生产主 KV (r=2)', t:'cmp', drill:'sparse', io:'[1,T,5120] -> [1,T,5120]', f:'C_j = \\mathrm{RMSNorm}(\\sum_{t=jr}^{(j+1)r-1}\\mathrm{softmax}(s_t)v_t)', d:'CSA2 的 Full 层：自算主 KV、索引器 K 与 Top-K。压缩比 2，是全模型四个 source 层中的第一个，其缓存被层 2–7 共用。'},
    e3:   {label:'Layers 3..19\n同模式再重复', t:'grp', io:'[1,T,5120] -> [1,T,5120]', f:'\\text{Full} \\to \\text{Reuse} \\times 5', d:'层 8、14 是另外两个 Full 层（各生产一份 r=2 的主 KV），其余层复用同组缓存。层 19 的输出即 H_{L/2}。'},
    hmid: {label:'H(L/2)\n编码器末层隐状态', t:'gt', io:'[1,T,5120]', f:'H_{L/2} = \\text{out}(\\text{Layer 19})', d:'解码器全局 KV 的输入。参考实现中层 20 的注意力输入就是它，因此「层 20 的压缩器读自己的输入」等价于「用 H_{L/2} 投影」。'},
    d20:  {label:'Layer 20\nFull：生产主 KV (r=1)', t:'dec', drill:'attn', io:'[1,T,5120] -> [1,T,5120]', f:'C_{20} = H_{L/2} W_{20}^{KV}', d:'解码器的 Full 层，唯一执行 CED 投影的层。压缩比 1（每 token 一条 288 字节条目），其缓存被层 20–39 共用。'},
    d21:  {label:'Layers 21..39\n含 4 个 Reindex 层', t:'grp', io:'[1,T,5120] -> [1,T,5120]', f:'\\text{Reuse} \\to \\text{Reindex} \\to \\text{Reuse}', d:'层 24、28、32、36 是 Reindex 层：复用主 KV 与索引器 K，但在候选池内自己重算 Top-K；其余层连 Top-K 一起复用。'},
    head: {label:'lm_head\n5120 → 129280', t:'io', io:'[1,T,5120] -> [1,T,129280]', f:'\\mathrm{logits} = W_{lm}\\,x', d:'输出头。真实权重 head.weight 形状 [129280, 5120] BF16。'},
    mtp:  {label:'DSpark\n3 个草稿块', t:'dspark', drill:'dspark', io:'[1,T,5120] -> 5 个草稿位置', f:'\\text{draft} \\times 3\\ \\text{blocks}', d:'块式推测解码。输入取自层 37、38、39 的注意力输入，不改变主干缓存。点击下钻看草稿链。'}
  },
  edges:[
    ['emb','e0','5120'],['e0','e1','5120'],['e1','e2','5120'],['e2','e3','5120'],
    ['e3','hmid','5120'],['hmid','d20','5120（同时喂 KV 投影）'],['d20','d21','5120'],
    ['d21','head','5120'],['d21','mtp','5120']
  ]
},

// ---------- 视图2:注意力内部 ----------
attn: {
  title:'注意力内部（层 2 / 层 20 的 Full 层结构相同，仅压缩比不同）',
  rankSep:40, nodeSep:20,
  legend:[['#8a93a3','残差流读写'],['#3f6fd0','注意力主体'],['#b7791f','位置与门控'],['#16a085','缓存'],['#e0922f','压缩条目']],
  nodes:{
    in:  {label:'残差流入\n4×5120', t:'grp', io:'[1,T,20480]', f:'x = (x_1,x_2,x_3,x_4)', d:'超连接把残差流复制成 4 份，每份 5120 宽。'},
    hc1: {label:'attn 超连接\n读出', t:'grp', io:'[1,T,20480] -> [1,T,5120]', f:'\\tilde{x} = \\textstyle\\sum_i \\mathrm{sigmoid}(\\mathrm{mixes}\\cdot\\mathrm{scale}+\\mathrm{base})_i \\odot x_i', d:'读出系数由上一层 FFN 的超连接混合提供（层 0 为 one-hot）；本层 hc_attn_fn（形状 [24, 20480]）产出的 post/comb 系数用于本子层写回，其 attn_pre 供本层 FFN 读出。'},
    wqa: {label:'wq_a\n5120 → 1280', t:'attn', io:'[1,T,5120] -> [1,T,1280]', f:'q = W_{q_a}\\tilde{x}', d:'查询低秩投影。q_lora_rank=1280；真实权重形状 [1280, 5120]。'},
    qn:  {label:'q_norm\n1280', t:'attn', io:'[1,T,1280] -> [1,T,1280]', f:'q \\leftarrow \\mathrm{RMSNorm}(q)', d:'低秩后再归一化。q_norm.weight 形状 [1280]。'},
    wqb: {label:'wq_b\n1280 → 32768', t:'attn', io:'[1,T,1280] -> [1,T,64,512]', f:'64 \\times 512 = 32768', d:'展开成 64 个查询头、头维 512。真实权重形状 [32768, 1280]。'},
    rope:{label:'RoPE\n仅末尾 64 维', t:'gt', io:'[1,T,64,512]', f:'\\mathrm{rope\\_head\\_dim}=64', d:'只旋转头维的末尾 64 维（源码用 x[..., -rd:] 切片）。频率表按层选择：层 0、1（compress_ratio=0）用 base 10000 且关 YaRN；层 2–39（compress_ratio>0）用 base 160000 且开 YaRN，查询、窗口 KV 与压缩条目共用同一份表。'},
    wkv: {label:'wkv\n5120 → 512', t:'attn', io:'[1,T,5120] -> [1,T,512]', f:'1\\ \\text{KV head} \\times 512', d:'只有一个 KV 头、头维 512，64 个查询头共享这一份键值。真实权重形状 [512, 5120]。'},
    kvn: {label:'kv_norm\n512', t:'attn', io:'[1,T,512] -> [1,T,512]', f:'k,v \\leftarrow \\mathrm{RMSNorm}(\\cdot)', d:'键值归一化。kv_norm.weight 形状 [512]。'},
    win: {label:'窗口 KV\nFP8，128 槽', t:'buf', io:'[bsz, 128, 512]', f:'128 \\times 512 \\times 1\\ \\mathrm{B} = 64\\ \\mathrm{KiB/层}', d:'最近 128 个位置的原始键值，环形缓冲、不随序列长度增长。保持 FP8。'},
    cmp: {label:'压缩条目\nTop-512 条', t:'cmp', drill:'sparse', io:'[1,T,512,512]', f:'288\\ \\mathrm{B}/\\text{条目} \\times 512', d:'该层所属压缩比组的条目中被索引器选中的 512 条。位置取组起点 j·r。压缩条目由 Compressor 的独立投影产生（自带 wkv 与 norm），不复用窗口 KV 的 wkv/kv_norm。点击下钻看选择过程。'},
    sink:{label:'汇聚点\n每头 1 个标量', t:'gt', io:'[64] F32', f:'+ e^{\\mathrm{sink}-m}\\ \\text{只进分母}', d:'每头一个可学习标量，加在 softmax 分母上，不携带值向量。真实张量 attn.attn_sink 形状 [64]。'},
    sa:  {label:'稀疏注意力\n窗口 + 压缩', t:'attn', io:'窗口 128 + 压缩 512 -> [1,T,64,512]', f:'o = \\frac{\\sum_t e^{q\\cdot k_t/\\sqrt{d}-m}v_t}{\\sum_t e^{q\\cdot k_t/\\sqrt{d}-m} + e^{\\mathrm{sink}-m}}', d:'两部分槽位拼进同一次调用，softmax 在并集上做一次归一化。索引为 $-1$ 的占位槽位分子分母都不贡献。'},
    woa: {label:'wo_a\n32768 → 8192（8 组）', t:'attn', io:'[1,T,64,512] -> [1,T,8,1024]', f:'o\\_groups=8,\\ o\\_lora\\_rank=1024', d:'按 8 组各降到 1024 维，避免一次处理 32768 维。真实权重形状 [8192, 4096]。'},
    wob: {label:'wo_b\n8192 → 5120', t:'attn', io:'[1,T,8192] -> [1,T,5120]', f:'y = W_{o_b}o', d:'拼回隐藏维。真实权重形状 [5120, 8192]。'},
    hc2: {label:'写回四流\n× 系数', t:'grp', io:'[1,T,5120] -> [1,T,20480]', f:'x_i \\leftarrow x_i + \\lambda_i y', d:'子层输出按每流系数写回残差流。'}
  },
  edges:[
    ['in','hc1','20480'],['hc1','wqa','5120'],['wqa','qn','1280'],['qn','wqb','1280'],
    ['wqb','rope','32768'],['hc1','wkv','5120'],['wkv','kvn','512'],
    ['kvn','win','FP8 128 槽'],['hc1','cmp','经 Compressor 独立投影'],
    ['rope','sa','q'],['win','sa','128 位置'],['cmp','sa','512 条目'],['sink','sa','分母'],
    ['sa','woa','32768'],['woa','wob','8192'],['wob','hc2','5120']
  ]
},

// ---------- 视图3:压缩与稀疏选择 ----------
sparse: {
  title:'压缩条目与两级稀疏选择',
  rankSep:38, nodeSep:18,
  legend:[['#16a085','索引器打分与可达集'],['#b7791f','边界处理'],['#1f9d6b','选择结果'],['#3f6fd0','注意力']],
  nodes:{
    pool0:{label:'全部可达压缩条目\n随上下文线性增长', t:'idxr', io:'[1,T,N_reach,r=2 层约 50 万条]', f:'n_{\\mathrm{reach}}(i) = \\lfloor (i+1)/r \\rfloor', d:'只有「整组 token 都已出现」的条目才可达。1M 上下文、r=2 时约 50 万条；不可达条目在打分前置 $-\\infty$。'},
    score:{label:'索引器打分\n32 头 × 128 维', t:'idxr', io:'[1,T,32,128] x [n,r=128] -> [1,T,n]', f:'I_{q,j} = \\sum_h w_{q,h}\\mathrm{ReLU}(q_h\\cdot k_j)', d:'每头的贡献是 ReLU 内积再乘该头权重，按头求和。索引器 K 由压缩条目投影而来，每条 68 字节（128 维 FP4 + 每 32 通道一个 E8M0）。'},
    blk:  {label:'块级打分\n块分 = 块内最大', t:'idxr', io:'[1,T,n] -> [1,T,n/8]', f:'\\mathrm{block}(b) = \\max_{j\\in b} I_{q,j}', d:'把位置切成大小 8 的块，每块取块内最大索引分作为块分。'},
    pin:  {label:'钉住最新块', t:'gt', io:'—', f:'\\text{pin the newest block}', d:'最新块只填了一部分，若不钉住会被更早的完整块以更大块分压过，导致 query 看不到紧邻的历史。'},
    top2048:{label:'选出 2048 块', t:'idxr', io:'[1,T,n/8] -> 2048 块', f:'\\text{top-}2048', d:'只在建池层（层 20）执行一次；后续 Reindex 层复用这个池。'},
    cand:{label:'候选池\n2048 × 8 = 16384', t:'sel', io:'16384 个候选位置', f:'2048 \\times 8 = 16384', d:'候选位置集合。短上下文下可达条目本身少于 16384，此时候选池不起缩减作用。'},
    top512:{label:'位置级 Top-512', t:'sel', io:'16384 -> [1,T,512]', f:'\\text{top-}512', d:'在池内按索引分取前 512 条。Reindex 层重做这一步，Reuse 层直接复用上一次的选择。'},
    sa:   {label:'拼进稀疏注意力', t:'attn', io:'窗口 128 + 压缩 512', f:'\\mathcal{T} = \\text{窗口} \\cup \\text{Top-}512', d:'两部分槽位在同一次调用里做 softmax，窗口没有额外权重加成——谁的分高谁拿得多。'}
  },
  edges:[
    ['pool0','score','逐条打分'],['score','blk','聚合'],['pin','blk','保留最新块'],
    ['blk','top2048','块分排序'],['top2048','cand','展开 8 位置'],['cand','top512','池内排序'],
    ['top512','sa','512 条'],['pool0','sa','可达性决定候选范围']
  ]
},

// ---------- 视图4:MoE 内部 ----------
moe: {
  title:'MoE 与超连接写回',
  rankSep:38, nodeSep:20,
  legend:[['#8a93a3','残差流读写'],['#b7791f','门控'],['#3f6fd0','专家'],['#1f9d6b','选择']],
  nodes:{
    in:  {label:'残差流入\n读出后 5120', t:'grp', io:'[1,T,20480] -> [1,T,5120]', f:'\\tilde{x}', d:'由本层 hc_attn_fn 产出的 attn_pre 系数从 4 份残差流读出 5120 维——即「读入 FFN 的系数来自本层注意力产出」这一次序关系。'},
    norm:{label:'ffn_norm', t:'grp', io:'[1,T,5120]', f:'x \\leftarrow \\mathrm{RMSNorm}(x)', d:'MoE 前的归一化。'},
    gate:{label:'gate 打分\n384 专家', t:'gt', io:'[1,T,5120] -> [1,T,384]', f:'\\mathrm{score} = \\sqrt{\\mathrm{softplus}(xW^\\top)}', d:'真实权重 gate.weight 形状 [384, 5120]。另有 gate.bias 与 gate.bias_vl 各 [384]——图像 span 的 token 用另一套偏置。'},
    bias:{label:'修正偏置\n只影响选择', t:'gt', io:'[384] F32', f:'\\mathrm{idx} = \\mathrm{topk}_6(\\mathrm{score}+b)', d:'无辅助损失的负载均衡偏置：只参与 topk 排序，不进入权重计算，因此不改变被选中专家的输出尺度。'},
    sel: {label:'top-6 选择', t:'sel', io:'384 -> 6', f:'w = \\frac{\\mathrm{score}_{\\mathrm{idx}}}{\\sum \\mathrm{score}_{\\mathrm{idx}} + 10^{-20}} \\times 1.5', d:'选中 6 个路由专家；权重在选中集合内归一化后乘 route_scale=1.5。'},
    rexp:{label:'6 个路由专家\n各 2304 × 5120 × 3', t:'moe', io:'[1,T,5120] -> 6 × [1,T,2304]', f:'\\mathrm{w1},\\mathrm{w3}{:}\\,[2304,5120]\\ \\ \\mathrm{w2}{:}\\,[5120,2304]', d:'每个专家约 35.39M 参数（fp4 量化存储，I8 打包）。384 个专家全存但每 token 只用 6 个。'},
    shared:{label:'共享专家\n1 个，每 token 必过', t:'moe', io:'[1,T,5120] -> [1,T,2304]', f:'\\text{shared expert}', d:'与路由专家同形（约 35.39M），不参与稀疏选择。'},
    sum: {label:'加权相加', t:'moe', io:'6 × [1,T,2304] + [1,T,2304] -> [1,T,5120]', f:'y = \\sum_{i\\in\\mathrm{idx}} w_i E_i(x) + E_{\\mathrm{shared}}(x)', d:'路由专家按权重加权，共享专家恒权相加。'},
    hc:  {label:'写回四流', t:'grp', io:'[1,T,5120] -> [1,T,20480]', f:'x_i \\leftarrow x_i + \\lambda_i y', d:'MoE 输出按每流系数写回四份残差流。'}
  },
  edges:[
    ['in','norm','5120'],['norm','gate','5120'],['bias','gate','384'],
    ['gate','sel','384 -> 6'],['sel','rexp','6 个专家'],['norm','shared','5120'],
    ['rexp','sum','加权'],['shared','sum','恒权'],['sum','hc','5120']
  ]
},

// ---------- 视图5:Engram 注入 ----------
engram: {
  title:'Engram：n-gram 哈希查表（层 1 与层 14）',
  rankSep:36, nodeSep:18,
  legend:[['#c0392b','哈希与投影'],['#16a085','嵌入表'],['#b7791f','门控'],['#8a93a3','残差流']],
  nodes:{
    ctx:  {label:'上下文片段\n长度 2 / 3 / 4', t:'engram', io:'token id 序列', f:'n\\text{-gram},\\ n \\in \\{2,3,4\\}', d:'从当前位置往前取三种长度的连续片段。'},
    hash: {label:'哈希映射\n3 × 8 = 24 个下标', t:'engram', io:'-> 24 个表下标', f:'(4-1) \\times 8 = 24', d:'每种片段长度用 8 个不同的哈希函数映射，共 24 个下标。'},
    tab:  {label:'嵌入表\n两层合计约 196B', t:'buf', io:'[384006168, 256] 与 [384016682, 256]', f:'\\approx 196\\ \\mathrm{B}\\ \\text{参数}', d:'engram_vocab_size=16000000、head_dim=256。表很大，但每 token 只按 24 个下标取值，因此不计入激活参数。'},
    row:  {label:'取回 24 行\n每行 256 维', t:'engram', io:'-> [24, 256]', f:'24 \\times 256 = 6144', d:'每 token 实际取回 6144 个表元素，相对 196B 的表可以忽略。'},
    proj: {label:'稠密投影\n每层 157.33M', t:'engram', io:'[1,T,24,256] -> [1,T,5120]', f:'\\approx 157.33\\ \\mathrm{M}', d:'把取回的向量投影回隐藏维。这一步是稠密的，只要该层启用 Engram 就整份参与计算——这是 Engram 计入激活参数的部分。'},
    gate: {label:'门控\n带符号平方根后 sigmoid', t:'gt', io:'[1,T,5120]', f:'g = \\sigma(\\mathrm{copysign}(\\sqrt{|d|}, d))', d:'门控值取残差流与查表结果的归一化点积，先取绝对值开方、保留原符号，再经 sigmoid。与训练内核一致。'},
    add:  {label:'写回残差流', t:'grp', io:'[1,T,5120]', f:'x \\leftarrow x + g \\odot \\mathrm{proj}', d:'按门控加权写回。图像 span 的 token 跳过这一步。'}
  },
  edges:[
    ['ctx','hash','片段'],['hash','tab','24 个下标'],['tab','row','按行取值'],
    ['row','proj','[24,256]'],['gate','add','门控'],['proj','add','投影结果'],['add','ctx','写回后继续前向']
  ]
},

// ---------- 视图6:DSpark 草稿链 ----------
dspark: {
  title:'DSpark：块式推测解码草稿链',
  rankSep:36, nodeSep:18,
  legend:[['#7a45c9','主干层与草稿块'],['#8a93a3','拼接与投影'],['#b7791f','判定头'],['#556070','草稿输出']],
  nodes:{
    h37: {label:'层 37 注意力输入', t:'dec', io:'[1,T,4,5120]', f:'h_{37}', d:'取注意力子层的输入（不是输出），此时残差流仍是 4 份。'},
    h38: {label:'层 38 注意力输入', t:'dec', io:'[1,T,4,5120]', f:'h_{38}', d:'同上。'},
    h39: {label:'层 39 注意力输入', t:'dec', io:'[1,T,4,5120]', f:'h_{39}', d:'同上。'},
    mean:{label:'按 hc 维取均值\n4 → 1', t:'grp', io:'[1,T,4,5120] -> [1,T,5120]', f:'h.\\mathrm{mean}(\\mathrm{dim}=2)', d:'在超连接副本维（大小为 4）上取均值，把三层各自压成 5120 维。'},
    cat: {label:'沿特征维拼接\n3 → 15360', t:'grp', io:'3 × [1,T,5120] -> [1,T,15360]', f:'3 \\times 5120 = 15360', d:'三层的均值结果拼成一个 15360 维向量，作为一次性的草稿输入（不再自回归）。'},
    proj:{label:'main_proj\n15360 → 5120', t:'dspark', io:'[1,T,15360] -> [1,T,5120]', f:'W{:}\\,[5120, 15360]', d:'真实权重 main_proj.weight 形状 [5120, 15360]，与三层拼接的宽度一致。'},
    b1:  {label:'草稿块 1\n5 个位置', t:'dspark', io:'-> [1,5+V]', f:'\\text{block size} = 5', d:'块内位置用噪声 token（id 128799）占位，一次前向产出 5 个草稿位置。草稿块自身是 MoE：128 专家取 top-3。'},
    b2:  {label:'草稿块 2\n5 个位置', t:'dspark', io:'-> [1,5+V]', f:'\\text{block size} = 5', d:'同上。共 3 个 MTP 块（n_mtp_layers=3）。'},
    b3:  {label:'草稿块 3\n5 个位置', t:'dspark', io:'-> [1,5+V]', f:'\\text{block size} = 5', d:'最后一个块，额外带 Markov 头与置信头。'},
    markov:{label:'Markov 头\n秩 256', t:'gt', io:'[1,5,256]', f:'\\mathrm{rank} = 256', d:'dspark_markov_rank=256，给草稿序列内部的相邻关系建模。'},
    conf:{label:'置信头', t:'gt', io:'[1,5]', f:'\\text{confidence}', d:'输出每个草稿位置的接受置信度，决定这次接受几个草稿。'},
    out: {label:'草稿 token\n交主干校验', t:'out', io:'5 个草稿位置', f:'\\text{draft} \\to \\text{verify}', d:'草稿交给主干一次性校验。DSpark 不产生主干缓存，只提高一次前向平均能确认的 token 数。'}
  },
  edges:[
    ['h37','mean','取均值'],['h38','mean','取均值'],['h39','mean','取均值'],
    ['mean','cat','5120'],['cat','proj','15360'],['proj','b1','5120'],
    ['b1','b2','草稿'],['b2','b3','草稿'],['b3','markov','末块'],['b3','conf','末块'],
    ['markov','out','修正'],['conf','out','接受数']
  ]
},
};

var COLOR={io:'#1f9d6b',enc:'#c0392b',dec:'#7a45c9',attn:'#3f6fd0',moe:'#3f6fd0',
  grp:'#8a93a3',out:'#556070',cmp:'#e0922f',idxr:'#16a085',gt:'#b7791f',
  sel:'#1f9d6b',buf:'#16a085',engram:'#c0392b',dspark:'#7a45c9'};
