"""拉取 Qwen3.5 家族六个型号的 config 关键字段，生成家族对比表（需网络）。

对照口径：MoE 型号给出专家数/top-k/中间维，稠密型号给出 FFN 中间维。
"""
import json, subprocess

MODELS = ["Qwen3.5-397B-A17B", "Qwen3.5-122B-A10B", "Qwen3.5-35B-A3B", "Qwen3.5-27B", "Qwen3.5-9B", "Qwen3.5-4B"]
print(f"{'型号':<18s} {'层数':>4s} {'hidden':>6s} {'Q/KV':>7s} {'GDN v头':>7s} {'FFN':>20s} {'上下文':>7s}")
for m in MODELS:
    r = subprocess.run(["curl", "-sL", f"https://huggingface.co/Qwen/{m}/resolve/main/config.json"],
                       capture_output=True, timeout=120)
    c = json.loads(r.stdout)
    tc = c.get("text_config", c)
    q, kv = tc["num_attention_heads"], tc["num_key_value_heads"]
    if "num_experts" in tc:
        ffn = f"{tc['num_experts']}专家 top-{tc['num_experts_per_tok']} I{tc['moe_intermediate_size']}"
    else:
        ffn = f"稠密 FFN {tc['intermediate_size']}"
    print(f"{m:<18s} {tc['num_hidden_layers']:>4d} {tc['hidden_size']:>6d} {q}/{kv:>4d} {tc['linear_num_value_heads']:>7d} {ffn:>20s} {tc['max_position_embeddings']:>7d} tie={tc.get('tie_word_embeddings', c.get('tie_word_embeddings', False))}")
