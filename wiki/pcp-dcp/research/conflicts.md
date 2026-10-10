# 冲突记录

- 位置：正文第 2 章「dcp 的取值与约束」里 GQA 的整除约束。两处说法：官方博客 5.4 GQA Backend 写「(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0」，除数是重复因子 tp//KV 头数；官方仓库 vllm/config/model.py L1580-L1581 写「num_q_per_kv = total_num_attention_heads // total_num_kv_heads」后判 `num_q_per_kv % decode_context_parallel_size`，除数是查询头数/KV 头数。两者一般不等。页面采用：两条都作为源码/文档各自的约束，页面不并列这一条最细的整除条件（表述在上界与 MLA 的整除条件上收口），不在页面上讨论。
