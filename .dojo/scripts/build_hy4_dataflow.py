#!/usr/bin/env python3
"""生成 Hy4 preview 全屏可交互数据流页（源码核对版）。

页面只含数据（JSON）+ 引擎脚本；渲染、连线、拖动、缩放由
libs/dojo-flow.js 负责。

事实来源（三方交叉核对，缺一不写）：
  - 源码：transformers models/hy_v4/modeling_hy_v4.py 与 configuration_hy_v4.py
  - 配置：tencent/Hy4-preview → config.json
  - 权重：tencent/Hy4-preview → model.safetensors.index.json（2006 张量）
  另用部署侧实现 vLLM vllm/models/hy_v4/nvidia/* 与 SGLang
  sglang/srt/models/hunyuan_v4.py 确认检查点键名与运行时结构的对应关系。

每个节点都带 src（源码行号或权重键），可逐条回查。

    python3 .dojo/scripts/build_hy4_dataflow_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("wiki/hy4-preview-dataflow/index.html")

# 行号基准：transformers main 的 modeling_hy_v4.py
H = "modeling_hy_v4.py:{}"


def n(nid, name, shape=None, kind="op", src=None, note=None, drill=None, sub=None,
      detail=None):
    # src 仍作为构建期的可读标注保留在脚本里，供核对与复核；但不写进页面数据，
    # 节点下方不展示源码位置。
    node = {"id": nid, "name": name, "kind": kind}
    if shape:
        node["shape"] = shape
    # detail 是节点第三行：权重张量名 / 配置开关→行为 / 分层构成统计。
    # 这三类是参考图里信息密度最高的内容，全部可回查源码或 config.json。
    if detail:
        node["detail"] = detail
    if note:
        node["note"] = note
    if drill:
        node["drill"] = drill
    if sub:
        node["sub"] = sub
    return node


def e(a, b, label=None):
    edge = {"from": a, "to": b}
    if label:
        edge["label"] = label
    return edge


# ============================================================
# 视图数据
#
# 版式规则（对齐参考标准）：
#   1. 算子框（kind=op，蓝色）名 = 源码里被调用的模块/函数名；
#      张量框（kind=tensor，白色）名 = 源码里的变量名。
#   2. 算子与张量严格交替：每个算子后面跟一格它的输出。
#   3. 算子框第二行放它的规格（权重形状 / in → out / einsum 下标），
#      张量框第二行放 dtype + 形状；第三行放可回查的依据（权重键、开关、构成）。
#   4. 粒度按视图分层：主干与 Decoder 层保持粗（整块模块一格），
#      算子级细节放在 MLA + DSA / DSA 索引器 视图。
#
# 行号基准：transformers modeling_hy_v4.py（正文里不展示，仅供核对）
# ============================================================


# ============================================================
# 1 主干：HYV4Model.forward（L812-872）
# ============================================================
MAIN = {
    "id": "main",
    "label": "HYV4Model",
    "title": "主干：HYV4Model.forward 的一次完整前向",
    "nodes": [
        n("ids", "input_ids", "int64 [B, T]", "tensor", H.format("814"),
          detail="position_ids 缺省时按 past_seen_tokens 递推"),
        n("cache_init", "DynamicCache(config)", "use_cache 且无 cache 时新建", "cache",
          H.format("828-829"),
          detail="只在 use_cache=True 且未传入 past_key_values 时新建；否则沿用传入的 cache"),
        n("embed", "embed_tokens  (nn.Embedding)", "120832 → 6144", "op", H.format("826"),
          detail="embed_tokens.weight [120832, 6144] · padding_idx=120002"),
        n("emb_out", "inputs_embeds", "bf16 [B, T, 6144]", "tensor", H.format("826")),

        n("rotary", "rotary_emb  (HYV4RotaryEmbedding)", "inv_freq 32 → cat 成 64",
          "op", H.format("849"),
          detail="dim = head_dim = qk_rope_head_dim 64；arange(0,64,2) 得 32 个频率\n"
                 "freqs = inv_freq @ position_ids；emb = cat(freqs, freqs) → 64 维，全程 fp32\n"
                 "产物给每层 MLA 与 DSA 索引器的 RoPE 共用"),
        n("pos_emb", "position_embeddings (cos, sin)", "fp32 [B, T, 64] × 2", "port",
          H.format("849"),
          detail="跨视图传出：每层 MLA 与 DSA 索引器共用这两张表"),
        n("causal_mask", "create_causal_mask(**mask_kwargs)",
          'dict["deepseek_sparse_attention"]', "op", H.format("836-846"),
          detail="allow_is_causal_skip=False —— 强制建掩码，因为索引器也要用到因果性\n"
                 "generate 已备好 dict 形态时会跳过这步"),

        n("hc_expand", "unsqueeze(2).expand(hc_mult)", "1 → 4 条流", "op", H.format("851"),
          detail="config.hc_mult=4；本处不读权重，只是把单流复制成 4 条残差流"),
        n("hc_streams", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor", H.format("851")),

        n("layers", "78 × HYV4DecoderLayer", "iHC carried 路径", "op", H.format("854-864"),
          drill="layer",
          detail="enable_ihc=True → 全层走 iHC；L0 dense FFN + L1-77 MoE\n"
                 "indexer_types：21 个 full + 57 个 shared；prev_topk_indices 逐层传递"),
        n("layer_out", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor", H.format("864")),

        n("hc_head", "hc_head  (HYV4HyperHead)", "24576 → 6144", "op", H.format("867"),
          drill="ihc",
          detail="hc_head_fn [4, 24576] fp32 · hc_head_base [4] · hc_head_scale [1]"),
        n("merged", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("867"),
          detail="4 条流收束回 1 条"),

        n("norm", "norm  (HYV4RMSNorm)", "[6144]", "op", H.format("867"),
          detail="model.norm.weight [6144] · rms_norm_eps=1e-5"),
        n("last_hidden", "last_hidden_state", "bf16 [B, T, 6144]", "tensor", H.format("870"),
          detail="BaseModelOutputWithPast.last_hidden_state"),

        n("lmhead", "lm_head  (nn.Linear)", "6144 → 120832", "op", H.format("950"),
          detail="lm_head.weight [120832, 6144]；enable_lm_head_fp32=True → fp32 GEMM\n"
                 "tie_word_embeddings=False（不共享 embedding 权重）"),
        n("slice", "hidden_states[:, slice_indices, :]", "只算需要的 logits", "op",
          H.format("949"),
          detail="slice(-logits_to_keep, None)；logits_to_keep=0 时取全部位置"),
        n("logits", "logits", "fp32 [B, T, 120832]", "tensor", H.format("950")),
    ],
    "edges": [
        e("ids", "embed"), e("cache_init", "layers", "past_key_values"),
        e("embed", "emb_out"), e("ids", "rotary"),
        e("rotary", "pos_emb"), e("pos_emb", "layers", "每层 MLA / 索引器共用"),
        e("causal_mask", "layers", "attention_mask"),
        e("emb_out", "hc_expand"),
        e("hc_expand", "hc_streams"), e("hc_streams", "layers"),
        e("layers", "layer_out"), e("layer_out", "hc_head"), e("hc_head", "merged"),
        e("merged", "norm"), e("norm", "last_hidden"), e("last_hidden", "slice"),
        e("slice", "lmhead"), e("lmhead", "logits"),
    ],
    "groups": [
        {"label": "HYV4Model（每次 forward 执行一遍）",
         "members": ["ids", "cache_init", "embed", "emb_out", "rotary", "pos_emb",
                     "causal_mask", "hc_expand", "hc_streams", "layers", "layer_out",
                     "hc_head", "merged", "norm", "last_hidden"],
         "fill": "#f4f7fb", "stroke": "#b6c2d6", "label_color": "#4a5b78"},
        {"label": "HYV4ForCausalLM 额外的 LM head",
         "members": ["slice", "lmhead", "logits"],
         "fill": "#f2f7f3", "stroke": "#93bfa3", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


# ============================================================
# 2 Decoder 层：HYV4DecoderLayer.forward（L691-728）
# ============================================================
LAYER = {
    "id": "layer",
    "label": "HYV4DecoderLayer",
    "title": "Decoder 层：HYV4DecoderLayer.forward（L691-728）",
    "nodes": [
        n("rope_in", "position_embeddings (cos, sin)", "fp32 [B, T, 64] × 2", "port",
          H.format("698"),
          detail="跨视图传入：由主干的 rotary_emb 算好，本层与 DSA 索引器共用"),
        n("in", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor", H.format("693"),
          detail="上一次迭代的输出；4 条残差流"),
        n("prepare", "unsqueeze(2) / reshape 成 3D", "2D 或 2D*hc → 3D（仅 L0 需要）", "op",
          H.format("706"), detail="其余层已是 3D，这一步是 no-op"),
        n("cur", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor"),

        n("pre1", "attn_hc()", "返回 (post, out)", "op", H.format("706"), drill="ihc",
          detail="attn_hc.fn [8,24576] fp32 · base [8] · scale [2]（检查点键 hc_attn_layer.hc_pre.*）\n"
                 "一次调用同时产出 post（4 个门）与压缩后的单流输出"),
        n("reduced1", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("706"),
          detail="post [B,T,4] | residual [B,T,4,6144]"),

        n("iln", "input_layernorm", "[6144]", "op", H.format("708"),
          detail="HYV4RMSNorm(6144) · eps=1e-5"),
        n("attn_in", "hidden_states", "bf16 [B, T, 6144]", "tensor"),

        n("attn", "self_attn  (HYV4Attention)", "64 头 Gated MLA + DSA", "op",
          H.format("709-718"), drill="attn",
          detail="num_attention_heads=64；返回 (attn_out, attn_weights, topk_indices)"),
        n("attn_out", "attn_out", "bf16 [B, T, 6144]", "tensor", H.format("709"),

          detail="self_attn 的第三个返回值是 topk_indices [B,T,2048] int32"),
        n("topk", "topk_indices", "int32 [B, T, 2048]", "tensor", H.format("709"),
          detail="传给下一层作 prev_topk_indices；shared 层直接复用"),

        n("post1", "post × attn_out + residual", "写回 4 条流", "op",
          H.format("719"),
          detail="post 是 [B,T,4] 逐流标量，与 [B,T,6144] 外积后写回 4 条流；fp32 计算"),
        n("mid", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor", H.format("719")),

        n("pre2", "ffn_hc()", "返回 (post, out)", "op", H.format("722"), drill="ihc",
          detail="ffn_hc.fn [8,24576] fp32 · base [8] · scale [2]（检查点键 hc_mlp_layer.hc_pre.*）"),
        n("reduced2", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("722"),
          detail="post [B,T,4] | residual [B,T,4,6144]"),

        n("paln", "post_attention_layernorm", "[6144]", "op", H.format("724"),
          detail="HYV4RMSNorm(6144) · eps=1e-5"),
        n("mlp_in", "hidden_states", "bf16 [B, T, 6144]", "tensor"),

        n("mlp", "self.mlp", "L0: HYV4MLP(18432)\nL1-77: HYV4MoE(2048)", "op",
          H.format("725"), drill="moe"),
        n("mlp_out", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("725")),

        n("post2", "post × hidden_states + residual", "写回 4 条流", "op", H.format("726"),
          detail="写回 4 条流后作为本层输出"),
        n("out", "hidden_states", "bf16 [B, T, 4, 6144]", "tensor", H.format("728"),
          detail="连同 topk_indices 一起返回给主干循环"),
    ],
    "edges": [
        e("rope_in", "attn", "cos/sin"),
        e("in", "prepare"), e("prepare", "cur"), e("cur", "pre1"), e("pre1", "reduced1"),
        e("reduced1", "iln"), e("iln", "attn_in"), e("attn_in", "attn"),
        e("attn", "attn_out"), e("attn", "topk"), e("attn_out", "post1"), e("cur", "post1"),
        e("post1", "mid"), e("mid", "pre2"), e("pre2", "reduced2"),
        e("reduced2", "paln"), e("paln", "mlp_in"), e("mlp_in", "mlp"),
        e("mlp", "mlp_out"), e("mlp_out", "post2"), e("mid", "post2"),
        e("post2", "out"), e("topk", "out", "跨层传递"),
    ],
    "groups": [
        {"label": "Attention 子层",
         "members": ["rope_in", "in", "prepare", "cur", "pre1", "reduced1", "iln",
                     "attn_in", "attn", "attn_out", "topk", "post1", "mid"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "MLP 子层",
         "members": ["pre2", "reduced2", "paln", "mlp_in", "mlp", "mlp_out",
                     "post2", "out"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


# ============================================================
# 3 Gated MLA + DSA：HYV4Attention.forward（L406-492）
# ============================================================
ATTN = {
    "id": "attn",
    "label": "HYV4Attention",
    "title": "Gated MLA + DSA：HYV4Attention.forward（L406-492）",
    "nodes": [
        n("rope_in", "position_embeddings (cos, sin)", "fp32 [B, T, 64] × 2", "port",
          H.format("409"),
          detail="跨视图传入：主干的 rotary_emb 算一次，本层与索引器共用"),
        n("x", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("408"),
          detail="iHC 压缩后的单流输入"),

        n("gate", "gate_proj", "6144 → 16384", "op", H.format("420"),
          detail="检查点键 linear_gate.weight [16384, 6144]（HF 属性名是 gate_proj）\n"
                 "gated_mla=True, gating_type=elementwise"),
        n("gate_states", "gate_states", "bf16 [B, T, 64, 256]", "tensor", H.format("420"),
          detail="每头 256 维的门控，最后与 attn_output 逐元素相乘"),

        n("qa", "q_a_proj", "6144 → 2048", "op", H.format("422"),
          detail="q_a_proj.weight [2048, 6144] · q_lora_rank=2048"),
        n("qa_out", "q_a_proj(x)", "bf16 [B, T, 2048]", "tensor", H.format("422")),
        n("qln", "q_a_layernorm", "[2048]", "op", H.format("422"),
          detail="HYV4RMSNorm(2048) · q_a_layernorm.weight [2048]"),
        n("qresid", "q_resid", "bf16 [B, T, 2048]", "tensor", H.format("422"),
          detail="同时喂给 DSA 索引器作 q 侧输入"),
        n("qb", "q_b_proj", "2048 → 16384", "op", H.format("423"),
          detail="q_b_proj.weight [16384, 2048] · 64 头 × 256"),
        n("q", "q_states", "bf16 [B, 64, T, 256]", "tensor", H.format("423")),
        n("qsplit", "split(qk_nope, qk_rope)", "[192, 64]", "op", H.format("424"),
          detail="qk_nope_head_dim=192 · qk_rope_head_dim=64"),
        n("q_nope", "q_pass", "bf16 [B, 64, T, 192]", "tensor", H.format("424")),
        n("q_rot", "q_rot", "bf16 [B, 64, T, 64]", "tensor", H.format("424")),

        n("kva", "kv_a_proj_with_mqa", "6144 → 576", "op", H.format("426"),
          detail="kv_a_proj_with_mqa.weight [576, 6144] = kv_lora_rank 512 + rope 64"),
        n("kvraw", "compressed_kv", "bf16 [B, T, 576]", "tensor", H.format("426")),
        n("kvsplit", "split(kv_lora, k_rot)", "[512, 64]", "op", H.format("427")),
        n("kv_pass", "kv_pass", "bf16 [B, T, 512]", "tensor", H.format("427")),
        n("k_rot_pre", "k_rot", "bf16 [B, T, 64]", "tensor", H.format("427")),

        n("kvln", "kv_a_layernorm", "[512]", "op", H.format("429"),
          detail="HYV4RMSNorm(512) · kv_a_layernorm.weight [512]"),
        n("k_pass_ln", "k_pass", "bf16 [B, 1, T, 512]", "tensor", H.format("429")),

        n("rope", "apply_rotary_pos_emb", "非交错 RoPE，只作用于 64 维", "op",
          H.format("431-434"),
          detail="q*cos + rotate_half(q)*sin；rotate_half 把后半维取负后与前半维互换\n"
                 "rope_theta=1e7；q_rot 与 k_rot 同一次调用"),
        n("q_rot_pe", "q_rot", "bf16 [B, 64, T, 64]", "tensor"),
        n("k_rot_pe", "k_rot", "bf16 [B, 1, T, 64]", "tensor"),

        n("cachew", "past_key_values.update", "写入压缩潜向量", "cache", H.format("437-438"),
          detail="缓存存的是压缩潜向量 512+64，展开发生在读回之后"),
        n("k_pass_c", "k_pass", "bf16 [B, 1, T, 512]", "tensor"),
        n("k_rot_c", "k_rot", "bf16 [B, 1, T, 64]", "tensor"),

        n("qcat", "cat(q_pass, q_rot)", "[192] + [64]", "op", H.format("440"),
          detail="query_states = torch.cat((q_pass, q_rot), dim=-1)"),
        n("query_states", "query_states", "bf16 [B, 64, T, 256]", "tensor", H.format("440")),

        n("expandkv", "expand_kv", "512 → 64 × 448", "op", H.format("442, 387-404"),
          detail="kv_b_proj.weight [28672, 512]；28672 = 64 × (192+256)"),
        n("key_states", "key_states", "bf16 [B, 64, T, 256]", "tensor", H.format("442")),
        n("value_states", "value_states", "bf16 [B, 64, T, 256]", "tensor", H.format("442")),

        n("indexer", "indexer / prev_topk_indices", "2048 选 top-k", "op",
          H.format("444-457"), drill="indexer",
          detail="indexer_types[layer]=full 时跑 HYV4Indexer；shared 时复用上一层"),
        n("topk_idx", "topk_indices", "int32 [B, T, 2048]", "tensor", H.format("444-457")),

        n("mask", "masked_fill", "-inf 掩码", "op", H.format("459-470"),
          detail="eager 路径把未选中位置置 dtype 最小值；flash 路径直接把索引交给 kernel"),
        n("masked", "attention_mask", "bf16 [B, 1, T, T]", "tensor", H.format("459-470")),
        n("sparse_idx", "sparse_indices = topk_indices", "flash 路径不走掩码", "tensor",
          H.format("471-472"),
          detail="仅当 _attn_implementation 不是 eager/sdpa 时：索引原样传入 kernel（indices=…）"),

        # eager_attention_forward 内部有 6 步，收成一格、点开看「注意力核心」视图
        n("attncore", "attention_interface", "64 头 · sink 参与归一化", "op",
          H.format("474-487"),
          drill="attncore",
          detail="注意力内核 6 步（QKᵀ → mask → 拼 sink → softmax → 去 sink → ·V）\n"
                 "s_aux=sinks [64] fp32（检查点键 learnable_sink_param）"),
        n("attn_out", "attn_output", "bf16 [B, 64, T, 256]", "tensor", H.format("477")),
        n("sink", "sinks", "fp32 [64]", "tensor", H.format("486"),
          detail="每头一个可学习标量，初始化 0.0"),

        n("gated", "attn_output * sigmoid(gate)", "逐元素门控", "op", H.format("490"),
          detail="gate_states 是 [B,T,64,256]，与 attn_output 逐元素相乘"),
        n("gated_out", "attn_output", "bf16 [B, 64, T, 256]", "tensor", H.format("490")),

        n("oproj", "o_proj", "16384 → 6144", "op", H.format("491-492"),
          detail="o_proj.weight [6144, 16384]"),
        n("out", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("492")),
    ],
    "edges": [
        e("x", "gate"), e("gate", "gate_states"), e("x", "qa"), e("qa", "qa_out"),
        e("qa_out", "qln"), e("qln", "qresid"), e("qresid", "qb"), e("qb", "q"),
        e("q", "qsplit"), e("qsplit", "q_nope"), e("qsplit", "q_rot"),
        e("x", "kva"), e("kva", "kvraw"), e("kvraw", "kvsplit"), e("kvsplit", "kv_pass"),
        e("kvsplit", "k_rot_pre"), e("kv_pass", "kvln"), e("kvln", "k_pass_ln"),
        e("rope_in", "rope", "cos/sin"),
        e("q_rot", "rope"), e("k_rot_pre", "rope"), e("rope", "q_rot_pe"),
        e("rope", "k_rot_pe"), e("k_pass_ln", "cachew"), e("k_rot_pe", "cachew"),
        e("cachew", "k_pass_c"), e("cachew", "k_rot_c"), e("q_nope", "qcat"),
        e("q_rot_pe", "qcat"), e("qcat", "query_states"),
        e("k_pass_c", "expandkv"), e("k_rot_c", "expandkv"),
        e("expandkv", "key_states"), e("expandkv", "value_states"),
        e("x", "indexer"), e("qresid", "indexer"), e("indexer", "topk_idx"),
        e("topk_idx", "mask"), e("mask", "masked"),
        e("topk_idx", "sparse_idx"), e("sparse_idx", "attncore", "flash 路径"),
        e("query_states", "attncore"), e("key_states", "attncore"),
        e("value_states", "attncore"), e("masked", "attncore"), e("sink", "attncore"),
        e("attncore", "attn_out"),
        e("attn_out", "gated"), e("gate_states", "gated"), e("gated", "gated_out"),
        e("gated_out", "oproj"), e("oproj", "out"),
    ],
    "groups": [
        {"label": "查询分支（q LoRA 压缩）",
         "members": ["rope_in", "qa", "qa_out", "qln", "qresid", "qb", "q", "qsplit",
                     "q_nope", "q_rot"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "KV 分支（潜向量压缩 + 缓存）",
         "members": ["kva", "kvraw", "kvsplit", "kv_pass", "k_rot_pre", "kvln",
                     "k_pass_ln", "rope", "q_rot_pe", "k_rot_pe", "cachew",
                     "k_pass_c", "k_rot_c", "qcat", "query_states", "expandkv",
                     "key_states", "value_states"],
         "fill": "#fdf6e8", "stroke": "#d8b264", "label_color": "#8a6a25"},
        {"label": "DSA 稀疏选择", "members": ["indexer", "topk_idx", "mask", "masked"],
         "fill": "#f3f2fb", "stroke": "#b3aede", "label_color": "#5b54a0"},
    ],
    "notes": [],
}


# ============================================================
# 4 DSA 索引器：HYV4Indexer.forward（L199-269）
# ============================================================
INDEXER = {
    "id": "indexer",
    "label": "HYV4Indexer",
    "title": "DSA 索引器：HYV4Indexer.forward（L199-269）",
    "nodes": [
        n("rope_in", "position_embeddings (cos, sin)", "fp32 [B, S, 64] × 2", "port",
          H.format("203"),
          detail="跨视图传入：与 MLA 主分支共用同一份 cos/sin"),
        n("x", "hidden_states", "bf16 [B, S, 6144]", "tensor", H.format("201")),
        n("qresid", "q_resid", "bf16 [B, S, 2048]", "tensor", H.format("202"),
          detail="复用 MLA 的 q_a_layernorm 输出，索引器不单独算 q"),

        n("wqb", "wq_b", "2048 → 4096", "op", H.format("232-233"),
          detail="wq_b.weight [4096, 2048] = index_n_heads 32 × head_dim 128"),
        n("q", "q", "bf16 [B, S, 32, 128]", "tensor", H.format("233")),
        n("qsplit", "q_pass, q_rot = torch.split(...)", "[64, 64]", "op", H.format("235"),
          detail="rope 取后 64 维（与 MLA 主分支的 q 布局相反）"),
        n("q_pe", "q_rot", "bf16 [B, S, 32, 64]", "tensor", H.format("235")),

        n("wk", "wk", "6144 → 128", "op", H.format("238-239"),
          detail="wk.weight [128, 6144] · index_head_dim=128"),
        n("k_pre", "wk(x)", "bf16 [B, S, 128]", "tensor", H.format("238")),
        n("knorm", "k_norm  (LayerNorm)", "[128]", "op", H.format("238-241"),
          detail="k_norm.weight / k_norm.bias [128]；权重在 fp32 常驻"),
        n("k", "k", "bf16 [B, S, 1, 128]", "tensor", H.format("238-241")),
        n("ksplit", "k_pass, k_rot = torch.split(...)", "[64, 64]", "op", H.format("241"),
          detail="单头 MQA：k 只有 1 个头"),
        n("k_pe", "k_rot", "bf16 [B, S, 1, 64]", "tensor", H.format("241")),

        n("rope", "apply_rotary_pos_emb", "unsqueeze_dim = 2", "op", H.format("243"),
          detail="q*cos + rotate_half(q)*sin；rotate_half(x)=cat(-x2, x1)\n"
                 "unsqueeze_dim=2（索引器的 q/k 是 [B,S,H,D]）；复用主分支算好的 cos/sin"),
        n("q_rot", "q_rot", "bf16 [B, S, 32, 64]", "tensor"),
        n("k_rot", "k_rot", "bf16 [B, S, 1, 64]", "tensor"),

        n("qcat", "cat(q_pass, q_rot)", "[64] + [64]", "op", H.format("244")),
        n("q_full", "q", "bf16 [B, S, 32, 128]", "tensor", H.format("244")),
        n("kcat", "cat(k_pass, k_rot)", "[64] + [64]", "op", H.format("245")),
        n("k_full", "k", "bf16 [B, S, 128]", "tensor", H.format("245"),
          detail="squeeze 掉单头维后是 [B, S, 128]"),

        n("cache", "update_indexer", "写入索引器自己的 K 缓存", "cache", H.format("247-248"),
          detail="与 MLA 的 KV 缓存相互独立"),
        n("k_cached", "k", "bf16 [B, S, 128]", "tensor"),

        n("dots", "torch.matmul", "q · kᵀ", "op", H.format("250"),
          detail="q.float() @ k.float()ᵀ —— 显式升 fp32；跳过 Hadamard 变换与 FP8 量化"),
        n("scores", "scores", "fp32 [B, S, 32, T]", "tensor", H.format("250")),
        n("relu", "relu", "负数截断", "op", H.format("251"),
          detail="负数截断为 0；打分不做 softmax"),
        n("scores_pos", "scores", "fp32 [B, S, 32, T]", "tensor", H.format("251")),

        n("wproj", "weights_proj", "6144 → 32", "op", H.format("255-259"),
          detail="weights_proj.weight [32, 6144]；乘 n_heads^-0.5 · softmax_scale"),
        n("weights", "weights", "fp32 [B, S, 32]", "tensor", H.format("255-259")),

        n("sum", "torch.matmul", "mhd, mkd -> mhk", "op", H.format("260"),
          detail="weights.unsqueeze(-2) @ scores：把 32 个头按权重压成一维"),
        n("index_scores", "index_scores", "fp32 [B, S, T]", "tensor", H.format("260")),

        n("causal", "masked_fill", "causal mask", "op", H.format("263-266"),
          detail="bool 掩码取与；浮点掩码直接相加，保证 padding 不参与打分"),
        n("masked", "index_scores", "fp32 [B, S, T]", "tensor", H.format("263-266")),

        n("topk", "topk", "min(index_topk, T)", "op", H.format("268-269"),
          detail="index_topk=2048；只有 21 个 full 层各带一份索引器权重"),
        n("indices", "topk_indices", "int32 [B, S, 2048]", "tensor", H.format("268-269")),
    ],
    "edges": [
        e("qresid", "wqb"), e("wqb", "q"), e("q", "qsplit"), e("qsplit", "q_pe"),
        e("x", "wk"), e("wk", "k_pre"), e("k_pre", "knorm"), e("knorm", "k"),
        e("k", "ksplit"), e("ksplit", "k_pe"), e("q_pe", "rope"), e("k_pe", "rope"),
        e("rope_in", "rope", "cos/sin"),
        e("rope", "q_rot"), e("rope", "k_rot"), e("qsplit", "qcat"),
        e("q_rot", "qcat"), e("qcat", "q_full"), e("ksplit", "kcat"),
        e("k_rot", "kcat"), e("kcat", "k_full"), e("k_full", "cache"),
        e("cache", "k_cached"), e("q_full", "dots"), e("k_cached", "dots"),
        e("dots", "scores"), e("scores", "relu"), e("relu", "scores_pos"),
        e("scores_pos", "sum"), e("x", "wproj"), e("wproj", "weights"),
        e("weights", "sum"), e("sum", "index_scores"), e("index_scores", "causal"),
        e("causal", "masked"), e("masked", "topk"), e("topk", "indices"),
    ],
    "groups": [
        {"label": "查询分支", "members": ["rope_in", "qresid", "wqb", "q", "qsplit", "q_pe"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "键分支与缓存",
         "members": ["x", "wk", "k_pre", "knorm", "k", "ksplit", "k_pe", "rope",
                     "q_rot", "k_rot", "qcat", "q_full", "kcat", "k_full", "cache",
                     "k_cached"],
         "fill": "#fdf6e8", "stroke": "#d8b264", "label_color": "#8a6a25"},
        {"label": "打分与选 top-k",
         "members": ["dots", "scores", "relu", "scores_pos", "wproj", "weights",
                     "sum", "index_scores", "causal", "masked", "topk", "indices"],
         "fill": "#f3f2fb", "stroke": "#b3aede", "label_color": "#5b54a0"},
    ],
    "notes": [],
}


# ============================================================
# 5 iHC：HYV4HyperConnection.forward（L632-651）+ HYV4HyperHead.forward（L666-675）
# ============================================================
IHC = {
    "id": "ihc",
    "label": "HYV4HyperConnection",
    "title": "iHC：HYV4HyperConnection / HYV4HyperHead",
    "nodes": [
        n("x", "hidden_streams", "bf16 [B, T, 4, 6144]", "tensor", H.format("632"),
          detail="iHC 的 4 条并行残差流；hc_mult=4"),

        n("flat", "flatten(2)", "4 × 6144 → 24576", "op", H.format("637"),
          detail="hidden_streams.flatten(2).float()：把 4 条流拼成一维并升 fp32"),
        n("flat_t", "flat", "fp32 [B, T, 24576]", "tensor", H.format("637")),

        n("norm", "input_norm", "无权重 RMSNorm", "op", H.format("639"),
          detail="HYV4UnweightedRMSNorm：只算 rsqrt 作为残差乘回去，无参数"),
        n("normed", "rsqrt", "fp32 [B, T, 1]", "tensor", H.format("639")),

        n("fn", "F.linear", "24576 → 8", "op", H.format("639"),
          detail="hc_fn [8, 24576] fp32；8 = 2 × hc_mult（pre 4 + post 4）"),
        n("fn_out", "fn(flat)", "fp32 [B, T, 8]", "tensor", H.format("639")),

        n("mixes", "F.linear(flat, fn) * input_norm(flat)", "逐元素乘", "op", H.format("639"),
          detail="mixes = F.linear(flat, fn) * input_norm(flat)"),
        n("mixes_t", "mixes", "fp32 [B, T, 8]", "tensor", H.format("639")),

        n("splitbase", "base.split(hc_mult)", "pre_b / post_b", "op", H.format("642"),
          detail="hc_base [8] fp32；pre 侧初值 -log(hc_mult-1)"),
        n("splitlogits", "mixes.split(hc_mult)", "pre_logits / post_logits", "op",
          H.format("644")),

        n("pre", "sigmoid(pre_logits · scale + pre_b)", "fp32 [B, T, 4]", "op",
          H.format("646"), detail="hc_eps=1e-6；pre 是压缩用的逐流门"),
        n("pre_t", "pre", "fp32 [B, T, 4]", "tensor", H.format("646")),
        n("post", "sigmoid(post_logits · scale + post_b)", "fp32 [B, T, 4]", "op",
          H.format("648"),
          detail="post = hc_post_magnitude × sigmoid(...) + hc_eps；magnitude=2.0"),
        n("post_t", "post", "fp32 [B, T, 4]", "tensor", H.format("648"),
          detail="HyperConnection 的第二个返回值；由调用方在 post 步骤里用来写回 4 条流"),

        n("reduce", "sum(pre × streams)", "4 条流 → 1 条", "op", H.format("649"),
          detail="torch.sum(pre.unsqueeze(-1) * hidden_streams, dim=2)"),
        n("out", "out.to(hidden_states.dtype)", "bf16 [B, T, 6144]", "tensor", H.format("651"),
          detail="转回 bf16，交给后面的 layernorm 与 self_attn/mlp"),

        n("head", "hc_head.forward  (HYV4HyperHead)", "4 条流 → 1 条", "op",
          H.format("666-675"),
          detail="hc_head_fn [4,24576] · hc_head_base [4] · hc_head_scale [1]；无 post 门"),
        n("headout", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("674-675"),
          detail="收束成单条流后进 model.norm"),
    ],
    "edges": [
        e("x", "flat"), e("flat", "flat_t"), e("flat_t", "norm"), e("norm", "normed"),
        e("flat_t", "fn"), e("fn", "fn_out"), e("fn_out", "mixes"), e("normed", "mixes"),
        e("mixes", "mixes_t"), e("mixes_t", "splitlogits"), e("splitbase", "pre"),
        e("splitbase", "post"), e("splitlogits", "pre"), e("splitlogits", "post"),
        e("pre", "pre_t"), e("post", "post_t"), e("pre_t", "reduce"),
        e("x", "reduce"), e("reduce", "out"), e("out", "head"), e("head", "headout"),
    ],
    "groups": [
        {"label": "HYV4HyperConnection（每层两次：attn_hc / ffn_hc）",
         "members": ["x", "flat", "flat_t", "norm", "normed", "fn", "fn_out", "mixes",
                     "mixes_t", "splitbase", "splitlogits", "pre", "pre_t", "post",
                     "post_t", "reduce", "out"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "HYV4HyperHead（全模型只有一处）",
         "members": ["head", "headout"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


# ============================================================
# 6 MoE：HYV4MoE.forward（L604-611）+ HYV4TopkRouter.forward（L524-549）
#        + HYV4Experts.forward（L565-580）与 _apply_gate（L582-587）
# ============================================================
MOE = {
    "id": "moe",
    "label": "HYV4MoE",
    "title": "MoE：HYV4MoE / HYV4TopkRouter / HYV4Experts",
    "nodes": [
        n("x", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("604"),
          detail="L1-77 走 HYV4MoE；L0 是 HYV4MLP(18432)，不经过这里"),

        n("gate_lin", "F.linear", "6144 → 256", "op", H.format("526"),
          detail="F.linear(x.float(), weight.float())；router.weight [256, 6144]"),
        n("router_logits", "router_logits", "fp32 [N, 256]", "tensor", H.format("526")),
        n("sigmoid", "sigmoid", "", "op", H.format("527"),
          detail="用 sigmoid 打分（不是 softmax）"),
        n("scores", "scores", "fp32 [N, 256]", "tensor", H.format("527")),

        n("bias", "+ e_score_correction_bias", "", "op", H.format("528"),
          detail="e_score_correction_bias [256] fp32（无辅助损失，初始 0）"),
        n("scores_for_choice", "scores_for_choice", "fp32 [N, 256]", "tensor", H.format("528")),

        n("grp", "topk(2).sum(-1)", "组内 top-2 求和", "op", H.format("529-533"),
          detail="n_group=1 → 分组退化，等价于全局 top-k"),
        n("group_scores", "group_scores", "fp32 [N, 1, 256]", "tensor", H.format("529-533")),

        n("sel", "topk(topk_group).scatter_", "生成组掩码", "op", H.format("534-542"),
          detail="topk_group=1；被选中组之外 masked_fill 成 -inf"),
        n("group_mask", "score_mask", "bool [N, 256]", "tensor", H.format("534-542")),

        n("topk", "topk(num_experts_per_tok)", "256 选 8", "op", H.format("543-544"),
          detail="num_experts_per_tok=8（共 256 个路由专家）"),
        n("topk_idx", "topk_indices", "int64 [N, 8]", "tensor", H.format("543")),
        n("topk_w", "topk_weights", "fp32 [N, 8]", "tensor", H.format("544")),

        n("normw", "topk_weights /= sum", "归一化 × routed_scaling_factor", "op",
          H.format("545-548"),
          detail="norm_topk_prob=True；× routed_scaling_factor 2.827"),
        n("weights_n", "topk_weights", "fp32 [N, 8]", "tensor", H.format("548")),

        n("exp", "HYV4Experts", "256 个专家 · 命中者逐个算", "op", H.format("565-580"),
          drill="experts",
          detail="gate_up_proj [256,4096,6144] · down_proj [256,6144,2048]\n"
                 "点开看逐专家循环与 index_add_ 累加"),
        n("exp_out", "final", "fp32 [N, 6144]", "tensor", H.format("580")),

        n("shr", "shared_experts  (HYV4MLP)", "6144 → 2048 → 6144", "op",
          H.format("600-602, 610"),
          detail="intermediate = moe_intermediate_size × n_shared_experts = 2048×1\n"
                 "与路由专家并联；swiglu_limit 只作用于路由专家"),
        n("shr_out", "shared_experts(residuals)", "bf16 [B, T, 6144]", "tensor", H.format("610")),

        n("add", "hidden_states + shared_experts(residuals)", "routed + shared", "op",
          H.format("610"),
          detail="每个 token 激活 top-8 路由专家 + 1 个共享专家"),
        n("out", "hidden_states", "bf16 [B, T, 6144]", "tensor", H.format("611"),
          detail="回到 Decoder 层，由 post 门写回 4 条流"),
    ],
    "edges": [
        e("x", "gate_lin"), e("gate_lin", "router_logits"), e("router_logits", "sigmoid"),
        e("sigmoid", "scores"), e("scores", "bias"), e("bias", "scores_for_choice"),
        e("scores_for_choice", "grp"), e("grp", "group_scores"), e("group_scores", "sel"),
        e("sel", "group_mask"), e("group_mask", "topk"),
        e("scores_for_choice", "topk"), e("topk", "topk_idx"), e("topk", "topk_w"),
        e("topk_w", "normw"), e("normw", "weights_n"),
        # HYV4Experts 内部较细，收成一格、点开看 experts 视图
        e("topk_idx", "exp"), e("weights_n", "exp"), e("exp", "exp_out"),
        # 共享专家与路由专家并联
        e("x", "shr"), e("shr", "shr_out"),
        e("exp_out", "add"), e("shr_out", "add"), e("add", "out"),
    ],
    "groups": [
        {"label": "路由（HYV4TopkRouter）",
         "members": ["gate_lin", "router_logits", "sigmoid", "scores", "bias",
                     "scores_for_choice", "grp", "group_scores", "sel", "group_mask",
                     "topk", "topk_idx", "topk_w", "normw", "weights_n"],
         "fill": "#f3f2fb", "stroke": "#b3aede", "label_color": "#5b54a0"},
        {"label": "256 选 8 的路由专家（HYV4Experts）",
         "members": ["exp", "exp_out"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "共享专家（HYV4MLP，不 clamp）", "members": ["shr", "shr_out"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


# ============================================================
# 7 MTP：检查点里有，transformers 显式忽略（modeling_hy_v4.py:761）
# ============================================================
MTP = {
    "id": "mtp",
    "label": "HYV4MTP",
    "title": "MTP 草稿层：检查点存在，transformers 实现不加载",
    "nodes": [
        n("ids", "input_ids", "int64 [B, T]", "tensor", "vLLM mtp.py:496",
          detail="投机解码的草稿输入；与目标模型同一批 token"),
        n("embed", "embed_tokens", "120832 → 6144", "op", "vLLM mtp.py:501-502",
          detail="与主干共用 embed_tokens.weight [120832, 6144]"),
        n("emb", "inputs_embeds", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:502"),
        n("pos0", "torch.where(positions == 0, 0, x)", "首位置屏蔽", "op", "vLLM mtp.py:503",
          detail="torch.where((positions == 0).unsqueeze(-1), 0, inputs_embeds)"),
        n("emb_masked", "inputs_embeds", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:503"),

        n("prev", "previous_hidden_states", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:498",
          detail="上一层（目标模型）的 hidden_states"),

        n("enorm", "enorm  (RMSNorm)", "[6144]", "op", "mtp_layers.0.enorm.weight",
          detail="enorm.weight [6144]"),
        n("emb_n", "inputs_embeds", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:381"),
        n("hnorm", "hnorm  (RMSNorm)", "[6144]", "op", "mtp_layers.0.hnorm.weight",
          detail="hnorm.weight [6144]"),
        n("prev_n", "previous_hidden_states", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:382"),

        n("cat", "cat([embeds, hidden])", "6144 + 6144 = 12288", "op", "vLLM mtp.py:383-385",
          detail="torch.cat([inputs_embeds, previous_hidden_states], dim=-1)"),
        n("cat_t", "cat(...)", "bf16 [B, T, 12288]", "tensor", "vLLM mtp.py:383-385"),
        n("proj", "eh_proj", "12288 → 6144", "op", "mtp_layers.0.eh_proj.weight",
          detail="eh_proj.weight [6144, 12288]"),
        n("proj_t", "hidden_states", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:383-385"),

        # 草稿块就是一个 Decoder 层，粒度与 Decoder 层视图一致：收成一格
        n("block", "mtp_block  (HYV4DecoderLayer)", "MLA + DSA + MoE", "op",
          "vLLM mtp.py:386-390", drill="layer",
          detail="enable_ihc=False → 走普通单流残差，草稿块没有 hc_* 权重\n"
                 "点开看 Decoder 层视图"),
        n("block_t", "hidden_states", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:386-390"),
        n("final", "final_layernorm", "[6144]", "op",
          "mtp_layers.0.final_layernorm.weight",
          detail="final_layernorm.weight [6144]"),
        n("final_t", "hidden_states", "bf16 [B, T, 6144]", "tensor", "vLLM mtp.py:391"),

        n("head", "shared lm_head", "6144 → 120832", "op", "vLLM mtp.py:519-521",
          detail="与主干共用 lm_head，不额外占参数"),
        n("logits", "logits", "fp32 [B, T, 120832]", "tensor", "vLLM mtp.py:513-521",
          detail="投机解码的草稿分布；num_nextn_predict_layers=1"),
    ],
    "edges": [
        e("ids", "embed"), e("embed", "emb"), e("emb", "pos0"), e("pos0", "emb_masked"),
        e("emb_masked", "enorm"), e("enorm", "emb_n"), e("prev", "hnorm"),
        e("hnorm", "prev_n"), e("emb_n", "cat"), e("prev_n", "cat"), e("cat", "cat_t"),
        e("cat_t", "proj"), e("proj", "proj_t"), e("proj_t", "block"),
        e("block", "block_t"), e("block_t", "final"), e("final", "final_t"),
        e("final_t", "head"), e("head", "logits"),
    ],
    "groups": [
        {"label": "MTP 草稿层（检查点里 27 个张量）",
         "members": ["ids", "embed", "emb", "pos0", "emb_masked", "prev", "enorm",
                     "emb_n", "hnorm", "prev_n", "cat", "cat_t", "proj", "proj_t",
                     "block", "block_t", "final", "final_t"],
         "fill": "#fdf6e8", "stroke": "#d8b264", "label_color": "#8a6a25"},
        {"label": "投机解码时使用", "members": ["head", "logits"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [
        {"at": "block", "text": "transformers 显式忽略这部分权重（modeling_hy_v4.py:761）",
         "dx": 16, "dy": 16},
    ],
}


# ============================================================
# 8 HYV4Experts（L565-580）：逐专家循环，属于「该细就细」的部分
#    单独一个视图：MoE 视图里只留一格入口。
# ============================================================
EXPERTS = {
    "id": "experts",
    "label": "HYV4Experts",
    "title": "路由专家：HYV4Experts.forward（L565-580）",
    "nodes": [
        n("x", "hidden_states", "fp32 [N, 6144]", "tensor", H.format("565"),
          detail="展平后的 token；L1-77 每层都要跑一遍这里"),
        n("idx", "top_k_index", "int64 [N, 8]", "tensor", H.format("566"),
          detail="路由给出的 8 个专家下标"),
        n("w", "top_k_weights", "fp32 [N, 8]", "tensor", H.format("566"),
          detail="路由归一化并乘以 2.827 之后的权重"),

        n("zeros", "torch.zeros_like(hidden_states)", "先占位再累加", "op", H.format("568"),
          detail="final 是累加器；后面每个命中的专家都 index_add_ 到它上面"),
        n("final0", "final", "fp32 [N, 6144]", "tensor", H.format("568")),

        n("one_hot", "F.one_hot(..., num_classes=256+1)", "标记命中的 (token, 专家)", "op",
          H.format("570"),
          detail="多出来的第 256 类用来吞掉 padding 位的下标"),
        n("mask", "mask", "bool [257, 8, N]", "tensor", H.format("570")),

        n("hit", "mask.sum((-1,-2)) > 0 → nonzero", "本层实际用到的专家", "op",
          H.format("571"),
          detail="只遍历命中的专家，没被用到的专家一次都不算"),
        n("hit_idx", "hit", "命中的专家下标集合", "tensor", H.format("571")),

        # —— 循环体：对每个命中的专家执行下面这一串 ——
        n("where", "torch.where(mask[expert_idx])", "该专家负责的 token 位置", "op",
          H.format("576"),
          detail="取出该专家负责的 token 下标与它在 top-8 里的位次"),
        n("tok", "hidden_states[token_idx]", "fp32 [tokens, 6144]", "tensor", H.format("577")),

        n("eup", "F.linear(·, gate_up_proj[expert_idx])", "6144 → 4096", "op",
          H.format("577"),
          detail="gate_up_proj [256,4096,6144]；4096 = 2 × 专家中间维 2048"),
        n("gate_up", "gate_up", "fp32 [tokens, 4096]", "tensor", H.format("577")),

        n("chunk", "chunk(gate_up, 2)", "gate / up 各 2048", "op", H.format("583"),
          detail="gate, up = gate_up.chunk(2, dim=-1)"),
        n("g", "gate", "fp32 [tokens, 2048]", "tensor", H.format("583")),
        n("u", "up", "fp32 [tokens, 2048]", "tensor", H.format("583")),

        n("clamp", "clamp(gate) / clamp(up)", "gate 上限 10，up 双向 [-10, 10]", "op",
          H.format("584-585"),
          detail="swiglu_limit=10.0；gate 只封顶、up 双向夹。共享专家不做这个 clamp"),
        n("gc", "gate", "fp32 [tokens, 2048]", "tensor", H.format("584")),
        n("uc", "up", "fp32 [tokens, 2048]", "tensor", H.format("585")),

        n("swiglu", "silu(gate) × up", "SwiGLU", "op", H.format("587")),
        n("act", "silu(gate) × up", "fp32 [tokens, 2048]", "tensor", H.format("587")),

        n("edown", "F.linear(·, down_proj[expert_idx])", "2048 → 6144", "op",
          H.format("578"),
          detail="down_proj [256, 6144, 2048]"),
        n("down", "down_proj", "fp32 [tokens, 6144]", "tensor", H.format("578")),

        n("scale", "× top_k_weights[token_idx, top_k_pos]", "按该专家的路由权重缩放", "op",
          H.format("578"),
          detail="同一个 token 命中多个专家时，这里的权重是它在该专家上的路由权重"),
        n("cur", "current", "fp32 [tokens, 6144]", "tensor", H.format("578")),

        n("add", "final.index_add_(0, token_idx, current)", "累加回该 token", "op",
          H.format("579"),
          detail="同一个 token 命中多个专家时，这里做的是累加"),
        n("out", "final", "fp32 [N, 6144]", "tensor", H.format("580"),
          detail="256 个专家共用同一个 3D 权重张量，按命中集合循环"),
    ],
    "edges": [
        e("idx", "one_hot"), e("one_hot", "mask"), e("mask", "hit"), e("hit", "hit_idx"),
        e("hit_idx", "where"), e("where", "tok"), e("x", "tok"),
        e("tok", "eup"), e("eup", "gate_up"), e("gate_up", "chunk"),
        e("chunk", "g"), e("chunk", "u"), e("g", "clamp"), e("u", "clamp"),
        e("clamp", "gc"), e("clamp", "uc"), e("gc", "swiglu"), e("uc", "swiglu"),
        e("swiglu", "act"), e("act", "edown"), e("edown", "down"),
        e("down", "scale"), e("w", "scale"), e("scale", "cur"),
        e("cur", "add"), e("zeros", "final0"), e("final0", "add"), e("add", "out"),
    ],
    "groups": [
        {"label": "准备命中集合", "members": ["idx", "one_hot", "mask", "hit", "hit_idx"],
         "fill": "#f3f2fb", "stroke": "#b3aede", "label_color": "#5b54a0"},
        {"label": "循环体：每个命中的专家算一次（源码 L572-579 的 for）",
         "members": ["where", "tok", "eup", "gate_up", "chunk", "g", "u", "clamp",
                     "gc", "uc", "swiglu", "act", "edown", "down", "scale", "cur"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "累加器", "members": ["zeros", "final0", "add", "out"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


# ============================================================
# 9 注意力核心：eager_attention_forward（L286-312）
#    MLA 视图里只留一格入口（attention_interface）。
# ============================================================
ATTNCORE = {
    "id": "attncore",
    "label": "eager_attention",
    "title": "注意力核心：eager_attention_forward（L286-312）",
    "nodes": [
        n("q", "query", "bf16 [B, 64, T, 256]", "tensor", H.format("286")),
        n("k", "key", "bf16 [B, 64, T, 256]", "tensor", H.format("287")),
        n("v", "value", "bf16 [B, 64, T, 256]", "tensor", H.format("288")),

        n("repeat_kv", "repeat_kv", "每头一份，本模型为 no-op", "op", H.format("294-295"),
          detail="MLA 已把 KV 展开成每头一份，num_key_value_groups=1，所以这里不复制"),
        n("vr", "value_states", "bf16 [B, 64, T, 256]", "tensor", H.format("295")),
        n("kr", "key_states", "bf16 [B, 64, T, 256]", "tensor", H.format("294")),

        n("qk", "torch.matmul", "mhd, nhd -> mhn", "op", H.format("296"),
          detail="等价于 query @ keyᵀ：matmul(query, key.transpose(2,3)) * scaling，scaling=0.0625"),
        n("scores", "attn_weights", "fp32 [B, 64, T, T]", "tensor", H.format("296")),

        n("addmask", "+ attention_mask", "因果 / DSA 稀疏掩码", "op", H.format("297-298"),
          detail="掩码由 DSA 的 topk 与 causal mask 合并而来；未选中位置是 dtype 最小值"),
        n("scores_m", "attn_weights", "fp32 [B, 64, T, T]", "tensor", H.format("298")),

        n("sink", "sinks", "fp32 [64]", "tensor", H.format("300"),
          detail="每头一个可学习标量（检查点键 learnable_sink_param）"),
        n("catsink", "cat([attn_weights, sinks])", "T + 1 列", "op", H.format("300-301"),
          detail="把 sink 拼成一列一起参与归一化——它不是加一个偏置，而是多一个可分配的槽位"),
        n("combined", "combined_logits", "fp32 [B, 64, T, T+1]", "tensor", H.format("301")),

        n("submax", "- max(dim=-1)", "防 bf16 溢出", "op", H.format("306"),
          detail="源码注释：这一减法不在原始实现里，为避免 bf16/fp16 溢出而加"),
        n("stable", "combined_logits", "fp32 [B, 64, T, T+1]", "tensor", H.format("306")),

        n("softmax", "F.softmax", "dtype 与输入一致", "op", H.format("307")),
        n("probs", "probs", "fp32 [B, 64, T, T+1]", "tensor", H.format("307")),

        n("drop_sink", "scores = probs[..., :-1]", "丢掉 sink 那一列", "op", H.format("308"),
          detail="sink 的作用已经体现在归一化里；丢掉后各位置概率之和小于 1"),
        n("attn_w", "attn_weights", "bf16 [B, 64, T, T]", "tensor", H.format("309"),
          detail="乘 V 用，同时作为 attention 的第二个返回值"),

        n("av", "torch.matmul", "mhk, mkd -> mhd", "op", H.format("309-311"),
          detail="等价于 scores @ value：先 dropout（训练时 p=attention_dropout）再 matmul"),
        n("out", "attn_output", "bf16 [B, 64, T, 256]", "tensor", H.format("310-311"),
          detail="transpose(1,2).contiguous() 后返回"),
    ],
    "edges": [
        e("q", "qk"), e("k", "repeat_kv"), e("repeat_kv", "kr"), e("kr", "qk"),
        e("qk", "scores"), e("scores", "addmask"), e("addmask", "scores_m"),
        e("scores_m", "catsink"), e("sink", "catsink"), e("catsink", "combined"),
        e("combined", "submax"), e("submax", "stable"), e("stable", "softmax"),
        e("softmax", "probs"), e("probs", "drop_sink"), e("drop_sink", "attn_w"),
        e("attn_w", "av"), e("v", "repeat_kv"), e("repeat_kv", "vr"), e("vr", "av"),
        e("av", "out"),
    ],
    "groups": [
        {"label": "打分与掩码", "members": ["q", "k", "repeat_kv", "kr", "qk", "scores",
                                          "addmask", "scores_m"],
         "fill": "#eef3fb", "stroke": "#9db6d8", "label_color": "#4a6b96"},
        {"label": "sink 与归一化",
         "members": ["sink", "catsink", "combined", "submax", "stable", "softmax",
                     "probs", "drop_sink", "attn_w"],
         "fill": "#f3f2fb", "stroke": "#b3aede", "label_color": "#5b54a0"},
        {"label": "加权求和", "members": ["v", "vr", "av", "out"],
         "fill": "#f3f7f4", "stroke": "#9dc0ab", "label_color": "#4b7a5c"},
    ],
    "notes": [],
}


VIEWS = [MAIN, LAYER, ATTN, ATTNCORE, INDEXER, IHC, MOE, EXPERTS, MTP]


# ============================================================
# config.json 参数面板
#
# 数值逐条抄自官方 config.json，未做任何加工；面板按用途分组，便于对照。
# 需要更新时重新对照 https://huggingface.co/tencent/Hy4-preview/raw/main/config.json
# ============================================================
CONFIG_VERSION = "transformers_version 5.16.2 · dtype bfloat16"

CONFIG_GROUPS = [
    ("模型规模", [
        ("architectures", "[\"HYV4ForCausalLM\"]"),
        ("model_type", "hy_v4"),
        ("vocab_size", "120832"),
        ("hidden_size", "6144"),
        ("num_hidden_layers", "78"),
        ("max_position_embeddings", "1048576（1M）"),
        ("tie_word_embeddings", "False"),
        ("dtype / torch_dtype", "bfloat16"),
    ]),
    ("注意力（Gated MLA + DSA）", [
        ("num_attention_heads", "64"),
        ("num_key_value_heads", "8（config 原值；__post_init__ 里被覆盖为 64）"),
        ("head_dim", "64（只指 RoPE 那一段）"),
        ("q_lora_rank", "2048"),
        ("kv_lora_rank", "512"),
        ("qk_nope_head_dim", "192"),
        ("qk_rope_head_dim", "64"),
        ("qk_head_dim", "256（= 192 + 64）"),
        ("v_head_dim", "256"),
        ("gated_mla", "True"),
        ("gating_type", "elementwise"),
        ("learnable_sink", "True"),
        ("learnable_sink_init", "0.0"),
        ("attention_bias", "False"),
        ("attention_dropout", "0.0"),
        ("use_mla / use_dsa", "True / True"),
    ]),
    ("DSA 索引器", [
        ("index_n_heads", "32"),
        ("index_head_dim", "128"),
        ("index_topk", "2048"),
        ("indexer_types", "78 项：21 个 full + 57 个 shared"),
        ("full 层下标", "0, 1, 5, 9, 13, … , 73, 77（步长 4，前两层连续）"),
    ]),
    ("iHC 残差流", [
        ("enable_ihc", "True"),
        ("hc_mult", "4"),
        ("hc_magnitude", "2.0"),
        ("hc_eps", "1e-06"),
        ("rms_norm_eps", "1e-05"),
    ]),
    ("MoE / FFN", [
        ("mlp_layer_types", "78 项：1 个 dense（第 0 层）+ 77 个 sparse"),
        ("intermediate_size", "18432（dense 层）"),
        ("moe_intermediate_size", "2048（专家）"),
        ("n_routed_experts", "256"),
        ("n_shared_experts", "1"),
        ("num_experts_per_tok", "8"),
        ("n_group / topk_group", "1 / 1（分组退化，等价全局 top-k）"),
        ("norm_topk_prob", "True"),
        ("routed_scaling_factor", "2.827"),
        ("swiglu_limit", "10.0"),
        ("hidden_act", "silu"),
    ]),
    ("层类型与位置编码", [
        ("layer_types", "78 项全为 deepseek_sparse_attention"),
        ("rope_parameters", "rope_theta 10000000 / rope_type default"),
        ("bos_token_id", "120000"),
        ("eos_token_id", "120025"),
        ("pad_token_id", "120002"),
        ("use_cache", "True"),
    ]),
    ("MTP", [
        ("num_nextn_predict_layers", "1"),
        ("mtp_loss_factor", "0.1"),
    ]),
    ("其它", [
        ("initializer_range", "0.006"),
        ("bitwise_backward_align", "False"),
        ("enable_lm_head_fp32", "True"),
        ("transformers_version", "5.16.2"),
    ]),
]


def build_config_panel():
    """把 config.json 渲染成可折叠面板（构建期生成，无脚本也能读）。"""
    blocks = []
    for title, rows in CONFIG_GROUPS:
        body = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
        )
        blocks.append(
            f'<details class="cfg-group" open><summary>{title}</summary>'
            f"<table>{body}</table></details>"
        )
    return (
        f'<div class="cfg-note">config.json 原值，共 59 个键；'
        f"这里按用途分组全部列出（{CONFIG_VERSION}）。</div>"
        + "".join(blocks)
    )


def build_fallback():
    rows = []
    for view in VIEWS:
        rows.append(f"<h2>{view['label']}：{view['title']}</h2>")
        rows.append("<table><tr><th>节点</th><th>形状 / 参数</th></tr>")
        for node in view["nodes"]:
            shape = node.get("shape") or node.get("sub") or ""
            rows.append(
                "<tr><td>{}</td><td>{}</td></tr>".format(node["name"], shape)
            )
        rows.append("</table>")
    return "".join(rows)


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="腾讯混元 Hy4 preview 的算子级前向数据流：主干、Decoder 层、Gated MLA 与 DSA、DSA 索引器、iHC、MoE 与 MTP，每个节点标注源码行号或权重键。">
  <meta name="dojo:summary" content="Hy4 preview 的算子级前向数据流：78 层、hidden 6144、4 条并行残差流、64 头 Gated MLA、32 头 DSA 索引器选 top-2048、256 选 8 的 MoE。形状与超参逐条对准源码行号、config.json 与 2006 个权重张量。">
  <meta name="dojo:type" content="dataflow">
  <meta name="dojo:topics" content="模型结构,注意力机制">
  <meta name="dojo:tag" content="数据流">
  <title>Hy4 Preview 前向数据流 · Dojo</title>
  <link rel="stylesheet" href="../../libs/katex.min.css">
  <script defer src="../../libs/katex.min.js"></script>
  <script defer src="../../libs/auto-render.min.js"
    onload="renderMathInElement(document.body, { delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
    ], throwOnError: false });"></script>
  <link rel="stylesheet" href="../../libs/dojo-dataflow.css">
  <link rel="stylesheet" href="../../libs/dojo-flow.css">
  <noscript><style>
    /* 无脚本时画布不可交互，把节点表与 config 面板都直接显示出来 */
    .flow-config[hidden] { display: block !important; }
    .flow-cfg-btn, .flow-zoom, .flow-bar { display: none; }
  </style></noscript>
</head>
<body>
<div class="flow-app" id="flow-app">
  <noscript>
    <div class="flow-fallback">__FALLBACK__</div>
  </noscript>
  <nav class="flow-bar">__TABS__</nav>
  <button type="button" class="flow-cfg-btn" id="flow-cfg-btn"
          aria-expanded="false" aria-controls="flow-config">config.json</button>
  <div class="flow-zoom">
    <button type="button" data-act="out" title="缩小">−</button>
    <span class="flow-zoom-label" style="padding:0 6px;line-height:24px;font-size:12px;color:var(--flow-muted)">100%</span>
    <button type="button" data-act="in" title="放大">+</button>
    <button type="button" data-act="reset" title="复位">⟲</button>
  </div>
  <section class="flow-config" id="flow-config" hidden>
    <h2 class="cfg-title">config.json 参数</h2>
    __CONFIG__
  </section>
</div>
<script>window.DOJO_FLOW_DATA = __DATA__;</script>
<script src="../../libs/elk.bundled.js"></script>
<script src="../../libs/dojo-flow.js"></script>
<script>
(function () {
  var app = document.getElementById('flow-app');
  if (window.DojoFlow && window.DOJO_FLOW_DATA) {
    window.__flow = window.DojoFlow.mount(app, window.DOJO_FLOW_DATA);
  }
  // config.json 浮层：独立于视图切换，点按钮或 Esc 关闭
  var btn = document.getElementById('flow-cfg-btn');
  var panel = document.getElementById('flow-config');
  function setCfg(open) {
    if (!btn || !panel) return;
    panel.hidden = !open;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.classList.toggle('on', open);
  }
  if (btn && panel) {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      setCfg(panel.hidden);
    });
    panel.addEventListener('click', function (e) { e.stopPropagation(); });
    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') setCfg(false);
    });
    var hash = (location.hash || '').replace('#', '');
    if (hash === 'config') setCfg(true);
  }
})();
</script>
</body>
</html>
"""


def main():
    tabs = "".join(
        '<button type="button" class="flow-tab{on}" data-view="{vid}">{label}</button>'.format(
            on=" on" if i == 0 else "", vid=v["id"], label=v["label"]
        )
        for i, v in enumerate(VIEWS)
    )
    data = json.dumps({"views": VIEWS}, ensure_ascii=False, separators=(",", ":"))
    html = (
        PAGE.replace("__FALLBACK__", build_fallback())
        .replace("__TABS__", tabs)
        .replace("__CONFIG__", build_config_panel())
        .replace("__DATA__", data)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html)} bytes, {len(VIEWS)} views)")


if __name__ == "__main__":
    main()
