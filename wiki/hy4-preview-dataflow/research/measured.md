# 实测产物清单

本页的实测产物原先存放在本目录下，现已从仓库移除（内容不发布，且体积可观）。
下表是它们被移除时的登记，用于说明本页「含实测」的判定依据来自何处。

## 2026-09-21 重做：三方交叉核对

页面按源码重画（7 个视图：主干 / Decoder 层 / MLA+DSA / DSA 索引器 / iHC / MoE /
MTP）。核对方式与结论：

- **源码行号**：页面每个节点的 `src` 都是对着
  `transformers/models/hy_v4/modeling_hy_v4.py`（md5 `e0e494e9b017c483140f902957506273`）
  实际行号抓的，脚本逐行回查过一次，没有越界或错位。
- **配置**：`tencent/Hy4-preview/config.json`（md5 `e3f1035fe87e90539703eaedb6c821ef`）
  的 `num_hidden_layers`、`hidden_size`、`hc_mult`、`indexer_types`（21 full + 57 shared）、
  `mlp_layer_types`（第 0 层 dense、其余 77 层 sparse）等与页面一致。
- **权重**：`model.safetensors.index.json` 共 2006 个张量；页面上的形状（如
  `q_b_proj [16384, 2048]`、`kv_b_proj [28672, 512]`、`experts.gate_up_proj [256, 4096, 6144]`、
  `hc_head_fn [4, 24576]`）逐个回读 safetensors 文件头核对过。
- **命名分歧**：检查点键名（`linear_gate`、`learnable_sink_param`、
  `hc_attn_layer.hc_pre.hc_fn`）与 transformers 属性名（`gate_proj`、`sinks`、
  `attn_hc.fn`）不同名；vLLM/SGLang 的加载器显式做了映射，页面已标注这一点。
- **MTP**：检查点里 `mtp_layers.0` 有 27 个张量，但 transformers 实现用
  `_keys_to_ignore_on_load_unexpected`（L761）显式忽略；vLLM `mtp.py` 有完整实现，
  页面据此单独画了一张并标注该分歧。
- **渲染自检**：七个视图在 1512×900 下逐一截图查看；连线穿框 0、节点与标签无越界或
  被顶栏遮挡；拖拽节点后连线跟随、滚轮缩放缓动、切视图复位均实测通过，无 JS 报错。
- **生成脚本**：`.dojo/scripts/build_hy4_dataflow.py`（只输出数据 + 挂载引擎）。

| 文件 | 体积 | 说明 |
|---|---|---|
| `apply_review_fixes.py` | 0.0 KB | 实测脚本 |
| `chat_template.jinja` | 0.0 KB | 模型模板快照 |
| `config.json` | 0.0 KB | 配置或中间数据 |
| `diag_final.out` | 688 B | 运行输出存档 |
| `diag_final.py` | 0.0 KB | 实测脚本 |
| `diag_layer0.py` | 0.0 KB | 实测脚本 |
| `diag_layers.py` | 0.0 KB | 实测脚本 |
| `diag_ties.py` | 0.0 KB | 实测脚本 |
| `diag_topk.py` | 0.0 KB | 实测脚本 |
| `dsa_paper.pdf` | 0.9 KB | 原始材料 |
| `dsa_paper.txt` | 0.1 KB | 提取的文本材料 |
| `dtype_stats.out` | 0.0 KB | 运行输出存档 |
| `finetune_hy_v4_patches.py` | 0.0 KB | 实测脚本 |
| `generation_config.json` | 189 B | 配置或中间数据 |
| `headers.json` | 0.3 KB | 配置或中间数据 |
| `indexcache_paper.pdf` | 0.5 KB | 原始材料 |
| `indexcache_paper.txt` | 0.1 KB | 提取的文本材料 |
| `mini_model.py` | 0.0 KB | 实测脚本 |
| `model.safetensors.index.json` | 0.2 KB | 配置或中间数据 |
| `param_count.out` | 605 B | 运行输出存档 |
| `probe_forward.out` | 0.0 KB | 运行输出存档 |
| `probe_forward.py` | 0.0 KB | 实测脚本 |
| `probe_small_tensors.out` | 0.0 KB | 运行输出存档 |
| `probe_small_tensors.py` | 0.0 KB | 实测脚本 |
| `read_headers.py` | 0.0 KB | 实测脚本 |
| `render_test.py` | 0.0 KB | 实测脚本 |
| `src/__init__.py` | 991 B | 实测脚本 |
| `src/configuration_hy_v4.py` | 0.0 KB | 实测脚本 |
| `src/modeling_hy_v4.py` | 0.0 KB | 实测脚本 |
| `src/modular_hy_v4.py` | 0.0 KB | 实测脚本 |
| `verify_page_numbers.py` | 0.0 KB | 实测脚本 |
| `verify_structure.py` | 0.0 KB | 实测脚本 |
| `vllm/__init__.py` | 107 B | 实测脚本 |
| `vllm/attention.py` | 0.0 KB | 实测脚本 |
| `vllm/flashmla_sparse.py` | 0.0 KB | 实测脚本 |
| `vllm/hc.py` | 0.0 KB | 实测脚本 |
| `vllm/model.py` | 0.0 KB | 实测脚本 |
| `vllm/moe.py` | 0.0 KB | 实测脚本 |
| `vllm/mtp.py` | 0.0 KB | 实测脚本 |
| `vllm/triton_ihc.py` | 0.0 KB | 实测脚本 |
| `vllm_hy_v4.py` | 14 B | 实测脚本 |
