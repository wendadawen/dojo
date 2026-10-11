# 冲突记录

- 位置：「dcp 的取值与约束」里 GQA 的整除约束。两处说法：官方博客 5.4 GQA Backend 写「(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0」，除数是重复因子 tp//KV 头数；官方仓库 vllm/config/model.py L1580-L1587 写「num_q_per_kv = total_num_attention_heads // total_num_kv_heads」后判 `num_q_per_kv % decode_context_parallel_size`，除数是每 KV 头的查询头数。两者一般不等。页面采用：表述收口到「GQA 的上界是重复因子 t/H、且 dcp 要能整除」，不在页面并列这一条最细的整除对象。
