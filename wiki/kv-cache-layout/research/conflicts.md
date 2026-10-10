# 冲突记录

- 「默认布局」：blog/vllm-rfc-42082.md L39 把 LBHNC 标为 default（`**LBHNC** (**default**, hetero-TP-friendly)`）；repo/vllm/attention_backends_utils.py L193-L196 与 blog/flashinfer-kv-layout.rst L18-L20 则都默认优先 NHD（LBNHC）。按正文和表格取一处：页面写「默认取 NHD」，指 FlashInfer 文档给出的默认与 vLLM 源码的默认偏好，RFC 的提案默认不作为页面依据。
