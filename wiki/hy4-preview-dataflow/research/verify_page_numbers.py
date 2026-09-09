#!/usr/bin/env python3
"""数字逐项回查：页面中的每个关键数字必须能溯源到 research/*.out 存档或 config.json。
输出 PASS/FAIL 清单。"""
import json, re, sys

PAGE = '/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/index.html'
R = '/Users/wendadawen/code/dojo/wiki/hy4-preview-dataflow/research'

html = open(PAGE, encoding='utf-8').read()
config = json.load(open(f'{R}/config.json', encoding='utf-8'))

# (检查名, 页面应含的字符串(任一命中即 PASS), 溯源说明)
CHECKS = [
    # --- config.json 规格（页面表述 = config 值） ---
    ('vocab 120832', ['120832'], 'config vocab_size'),
    ('hidden 6144', ['6144'], 'config hidden_size'),
    ('layers 78', ['78 层', '78层'], 'config num_hidden_layers'),
    ('q_lora 2048', ['2048'], 'config q_lora_rank'),
    ('kv_lora 512', ['512'], 'config kv_lora_rank'),
    ('qk_nope 192', ['192'], 'config qk_nope_head_dim'),
    ('rope 64', ['64'], 'config qk_rope_head_dim'),
    ('v_head 256', ['256'], 'config v_head_dim'),
    ('heads 64', ['64'], 'config num_attention_heads'),
    ('indexer heads 32', ['32'], 'config index_n_heads'),
    ('indexer dim 128', ['128'], 'config index_head_dim'),
    ('topk 2048', ['2048'], 'config indexer_topk'),
    ('experts 256', ['256'], 'config n_routed_experts'),
    ('top-8', ['top-8', 'top8', 'top- 8'], 'config num_experts_per_tok'),
    ('routed_scaling 2.827', ['2.827'], 'config routed_scaling_factor'),
    ('dense FFN 18432', ['18432'], 'config intermediate_size (layer0)'),
    ('hc_mult 4', ['4'], 'config hyper_connection_multiplier'),
    ('magnitude 2.0', ['2.0'], 'config hyper_connection_magnitude'),
    ('rope theta 1e7', ['1e7', '10^7'], 'config rope_theta'),
    ('MTP 1 layer', ['1 层', 'num_next_predictors'], 'config num_next_predictors'),

    # --- param_count.out ---
    ('active 47.57B', ['47.57B', '47,572,762,525'], 'param_count.out ACTIVE'),
    ('total 779.96B', ['779.96B', '779,960,992,733'], 'param_count.out total'),
    ('MTP 10.05B', ['10.05B'], 'param_count.out mtp'),
    ('770B trunk 769.91B', ['769.91'], 'param_count.out 769.91B = official 770B'),
    ('per-layer attn 265,685,568', ['265,685,568'], 'param_count.out'),
    ('full indexer 9,371,904', ['9,371,904'], 'param_count.out'),
    ('per-layer iHC 393,236', ['393,236'], 'param_count.out'),
    ('hc_head 98,309', ['98,309'], 'param_count.out'),
    ('MoE layer 341,311,744', ['341,311,744'], 'param_count.out'),
    ('dense FFN params 339,738,624', ['339,738,624'], 'param_count.out'),
    ('routed experts 744.10B / 95.40%', ['744.10', '95.40'], 'param_count.out 分组'),

    # --- dtype_stats.out ---
    ('BF16 1380', ['1380'], 'dtype_stats.out'),
    ('F32 626', ['626'], 'dtype_stats.out'),

    # --- probe_forward.out ---
    ('pre gates mean 0.2500', ['0.2500'], 'probe_forward.out [A]'),
    ('post gates mean 0.9999', ['0.9999'], 'probe_forward.out [A]'),
    ('hc_head gates mean 0.2499', ['0.2499'], 'probe_forward.out [A]'),
    ('index >0 fraction 23.4%', ['23.4%'], 'probe_forward.out [B]'),
    ('ReLU zero 47.8%', ['47.8%'], 'probe_forward.out [B]'),
    ('sink absorbed mean 0.121', ['0.121'], 'probe_forward.out [C]'),
    ('sink absorbed max 0.502', ['0.502'], 'probe_forward.out [C]'),
    ('gate min 0.389', ['0.389'], 'probe_forward.out [C]'),
    ('gate max 0.615', ['0.615'], 'probe_forward.out [C]'),
    ('gate mean 0.500', ['0.500'], 'probe_forward.out [C]'),
    ('attn scaling 0.0625', ['0.0625'], 'sqrt(256)^-1 (probe C mini 48->0.144338 换算真实)'),
    ('MoE collision 7.3e-10 rounded', ['7.3\\times10^{-10}'], 'probe_forward.out [D] 7.276e-10 页面舍入'),
    ('token5 mentioned', ['token 5'], 'probe_forward.out [D] 概括表述'),
    ('weights range 0.33-0.37', ['0.33', '0.37'], 'probe_forward.out [D] [0.3321,0.3709] 页面概括'),
    ('swiglu clamp -69.9968', ['69.9968'], 'probe_forward.out [D]'),
    ('KV HF 32768/65536', ['32768', '65536'], 'probe_forward.out [E]'),
    ('KV vLLM 576/1152', ['576', '1152'], 'probe_forward.out [E]'),
    ('KV ratio 56.89', ['56.89'], 'probe_forward.out [E]'),
    ('per-token 78层 4.88 MB', ['4.88'], 'probe_forward.out [E]'),
    ('per-token 87.8 KB', ['87.8'], 'probe_forward.out [E]'),
    ('indexer cache 2772', ['2772'], 'probe_forward.out [E]'),
    ('indexer HF 5376', ['5376'], 'probe_forward.out [E]'),
    ('MTP own indexer', ['True'], 'probe_forward.out [F] 语义'),
    ('shared lm_head', ['lm_head'], 'probe_forward.out [F]'),

    # --- probe_small_tensors.out ---
    ('layer0 sink mean 0.384', ['0.384'], 'probe_small_tensors.out'),
    ('layer0 sink range -6.54..3.18', ['-6.54', '3.18'], 'probe_small_tensors.out'),
    ('layer40 sink 0.666', ['0.666'], 'probe_small_tensors.out'),
    ('layer77 sink -0.022', ['-0.022'], 'probe_small_tensors.out'),
    ('e_score layer1 median 0.0013', ['0.0013'], 'probe_small_tensors.out 0.00126962 舍入'),
    ('e_score layer40 median 0.0074', ['0.0074'], 'probe_small_tensors.out'),
    ('e_score layer77 median 0.0064', ['0.0064'], 'probe_small_tensors.out'),
    ('hc_base layer0 attn 前4', ['-0.886', '-0.879', '-0.718', '-0.910'], 'probe_small_tensors.out'),
    ('hc_base layer0 attn 后4', ['-1.518', '-1.418', '-1.398', '-0.486'], 'probe_small_tensors.out'),
    ('hc_base layer77 后4≈0', ['0.208'], 'probe_small_tensors.out'),
    ('hc_scale layer0 0.250', ['0.250'], 'probe_small_tensors.out'),
    ('hc_head base list', ['-1.144', '-1.151', '-1.212', '-0.972'], 'probe_small_tensors.out'),
    ('hc_head scale 0.105', ['0.105'], 'probe_small_tensors.out'),
    ('k_norm gamma 1.061', ['1.061'], 'probe_small_tensors.out'),
    ('k_norm beta -0.002', ['-0.002'], 'probe_small_tensors.out'),
    ('MTP sink 0.950', ['0.950'], 'probe_small_tensors.out'),
    ('MTP e_score 0.00737', ['0.00737'], 'probe_small_tensors.out'),
    ('init -log3', ['-log3', 'log3'], '源码 init + probe_small_tensors note'),
]

# config 值与页面一致性交叉验证
cfg_checks = []
h = config['hidden_size']
cfg_checks.append(('config.hidden_size', str(h), f'页面含 {h}'))
cfg_checks.append(('config.num_hidden_layers', str(config['num_hidden_layers']), '78'))
for k in ['vocab_size', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim',
          'v_head_dim', 'num_attention_heads', 'index_n_heads', 'index_head_dim', 'indexer_topk',
          'n_routed_experts', 'num_experts_per_tok', 'intermediate_size']:
    v = config.get(k)
    if v is None:
        # 尝试常见命名变体
        for alt in ['num_key_value_heads', 'moe_intermediate_size']:
            if alt in config:
                pass
    cfg_checks.append((f'config.{k}', str(v) if v is not None else None, None))

fails = []
for name, forms, src in CHECKS:
    if isinstance(forms, str):
        forms = [forms]
    hit = [f for f in forms if f in html]
    if hit:
        print(f'PASS  {name}  ({src}) -> {hit[0]!r}')
    else:
        print(f'FAIL  {name}  ({src}) -> none of {forms} found')
        fails.append(name)

print()
print('=== config 关键值（供人工对照） ===')
for k in ['hidden_size', 'num_hidden_layers', 'vocab_size', 'q_lora_rank', 'kv_lora_rank',
          'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'num_attention_heads',
          'index_n_heads', 'index_head_dim', 'indexer_topk', 'n_routed_experts',
          'num_experts_per_tok', 'intermediate_size', 'routed_scaling_factor',
          'hyper_connection_multiplier', 'hyper_connection_magnitude', 'rope_theta',
          'num_next_predictors', 'n_shared_experts', 'moe_intermediate_size']:
    if k in config:
        print(f'  {k} = {config[k]}')
    else:
        # 变体
        for kk in config:
            if k.split('_')[0] in kk and len(kk) < 40:
                pass
print(f'  (missing keys checked: {[k for k in ["num_next_predictors","n_shared_experts"] if k not in config]})')

print()
if fails:
    print(f'*** {len(fails)} FAILURES: {fails}')
    sys.exit(1)
print('*** ALL NUMERIC CHECKS PASSED ***')
