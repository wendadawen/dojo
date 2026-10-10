# DeepSeekMoE

- 描述：DeepSeekMoE 是一种混合专家（MoE）架构，把每个专家切成更小的多个、每个 token 激活更多个，再隔离出始终激活的共享专家，让路由专家更专精。
- 摘要：把专家切细，扩大激活专家可组合的方式；再隔离出共享专家承载公共知识，让路由专家更少冗余、更专精。
- 主题：模型结构
- 标签：MoE

## 长页

- 开头｜交代层、隐藏状态、前馈层（每 token 各算、含中间维）、MoE 层路由（token 是切分单位）；专家不专精两问题；DeepSeekMoE 是什么、两条措施｜400｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Preliminaries: Mixture-of-Experts for Transformers；paper/src-v1/main.tex Hyper-Parameters
- ### 把专家切细｜这一章讲把每个专家切成 m 份、每个 token 激活相应更多专家，两条内容在小节｜0｜paper/src-v1/main.tex Fine-Grained Expert Segmentation
- #### 切分与激活公式｜中间维降到 1/m、专家数变 mN，激活数增到 mK 以维持计算量；切分后 MoE 层的输出与门值公式、各符号｜350｜paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Preliminaries: Mixture-of-Experts for Transformers
- #### 组合灵活性与粒度收益｜切细后可组合的激活专家方式数放大，专家越细整体越好；N=16 的例子与消融数字折叠｜200｜paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex L596-L599
- ### 隔离共享专家｜隔离 K_s 个专家每个 token 都分给它们，路由激活数减 K_s；共享专家承载公共知识、降冗余；完整公式与符号｜350｜paper/src-v1/main.tex Shared Expert Isolation
- ### 负载均衡｜先一句说明什么是损失；自动路由可能负载不均（坍缩风险、专家跨设备时瓶颈），用专家级与设备级两个损失约束；两个公式折叠｜200｜paper/src-v1/main.tex Load Balance Consideration；paper/src-v1/main.tex Expert-Level Balance Loss.；paper/src-v1/main.tex Device-Level Balance Loss.
- ### 专精到什么程度｜冗余怎么衡量与结果（路由专家越不可替代）、共享专家去掉后变差、少激活几个路由专家仍够用；数字折叠｜300｜paper/src-v1/main.tex L623-L634；paper/src-v1/main.tex L656-L665
- ### 适用条件与边界｜专家粒度受效率限制、共享与激活路由专家的比例；代价是注意力参数少、多选类任务偏弱；一句与常规 MoE 的区别｜300｜paper/src-v1/main.tex Model Settings.；paper/src-v1/main.tex Training Settings.；paper/src-v1/main.tex Ratios Between Shared and Routed Experts.；paper/src-v1/main.tex Internal Comparison with DeepSeek 7B；paper/src-v1/main.tex Related Work

## 短页

- ### 每个 token 激活更多的小专家｜是什么：MoE 层把前馈层（FFN，每 token 各算、内有中间维）换成多专家、每 token 只选几个算；DeepSeekMoE 把专家切小并加共享专家｜280｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Shared Expert Isolation
- ### 专家该更专精｜为什么需要：常规 MoE 专家少，一个专家可能装下多类知识（知识混杂），不同专家又可能重复学公共知识（知识冗余）｜300｜paper/src-v1/main.tex Introduction
- ### 切细与共享专家｜怎么做：把前馈层的中间维降到 1/m 并相应多激活几个（路由指给每个 token 挑专家那步）；隔离一部分专家作始终激活的共享专家，其余路由激活数减 K_s 个｜280｜paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Shared Expert Isolation
