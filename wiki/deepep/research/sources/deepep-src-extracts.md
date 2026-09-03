# DeepEP 源码摘录（commit 01dc3aa，2026-08-04，main 分支）

## 1. V1 low_latency_dispatch docstring（初始 commit ebfe47e:deep_ep/buffer.py）

    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        A low-latency implementation for dispatching with IBGDA **with implicit FP8 casting**.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Even for ranks in the same node, NVLink are fully disabled for simplicity.
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
            low-latency kernels' result tensor at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tuple with received tokens for each expert. The first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receive. As mentioned before, all not tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """

## 2. V1 low_latency_combine docstring（初始 commit ebfe47e:deep_ep/buffer.py）

    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Even for ranks in the same node, NVLink are fully disabled for simplicity.
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
            low-latency kernels' result tensor at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you not set this flag, the kernel will ensure the data's arrival.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, num_topk]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """

## 3. 初始 README 开头（ebfe47e:README.md，2025-02-24 发布）

# DeepEP

DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also as known as MoE dispatch and combine. The library also supports low-precision operations, including FP8.

To align with the group-limited gating algorithm proposed in the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper, DeepEP offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from NVLink domain to RDMA domain. These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks. Additionally, they support SM (Streaming Multiprocessors) number control.

For latency-sensitive inference decoding, DeepEP includes a set of low-latency kernels with pure RDMA to minimize delays. The library also introduces a hook-based communication-computation overlapping method that does not occupy any SM resource.

Notice: the implementation in this library may have some slight differences from the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper.

## Performance


## 4. V1 示例代码关键注释（legacy.md 已含，此处为 buffer.py get_low_latency_rdma_size_hint 相关注释）

    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return deep_ep_cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)

    def get_local_buffer_tensor(self, dtype: torch.dtype, size: Optional[torch.Size] = None,

## 5. V1 buffer.py 初始化代码中 IBGDA 的释义（初始 commit ebfe47e:deep_ep/buffer.py）

    # Enable IBGDA for the low latency mode, which refers to "no package forwarding between NVLink and RDMA"
    if low_latency_mode:
        assert num_qps_per_rank > 0
        os.environ['NVSHMEM_DISABLE_P2P'] = '1'
        os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
        os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
        os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'

## 6. V2 elastic.py get_theoretical_num_sms 内部注释（commit 01dc3aa，行 750-758）

        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0

## 7. V1 internode_ll.cu 低延迟内核的槽位地址计算（初始 commit ebfe47e:csrc/kernels/internode_ll.cu，dispatch 侧约行 133-141）

                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_int2);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;

（combine 侧约行 228-232 为同构偏移：local_expert_idx * num_ranks * num_max + src_rank * num_max；
atomic_counter_per_expert 位于每次 kernel 启动的 workspace（行 310），即计数器是发送 rank 本地的——
段内 slot_idx 是"该来源 rank 发往此专家的第几个 token"。）
