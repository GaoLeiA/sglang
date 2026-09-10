# Triton fused ops

需要包含：

*   名称
    
*   功能说明
    
*   参考实现
    
*   涉及框架及模型
    

# vllm

## 1. Flash Linear Attention 算子 (`vllm/third_party/flash_linear_attention/ops/`)

这些算子服务于**线性注意力 / 门控线性 RNN / 混合线性注意力模型**（如 GDN 系列、MiniMax 类、各类 delta-rule / gated-delta-net 模型）。

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| solve\_tril.py | `solve_tril`(37) | 解下三角 (lower-triangular) 方程组，用于递归状态前向 | GDN / gated-delta-net 线性注意力模型 |
| solve\_tril.py | `solve_tril_bwd_dq`(112) | 反向传播计算 dQ | 同上 |
| solve\_tril.py | `solve_tril_bwd_dKh`(237) | 反向传播计算 dK/dh | 同上 |
| wy\_fast.py | `_wy_fast_fwd`(28) | WY 表示快速前向（chunk-wise 计算 WY 矩阵） | 线性注意力模型（MiniMax 类） |
| chunk\_delta\_h.py | `chunk_delta_rule_bwd_dh`(44) | chunk 级 delta-rule 反向 dH | GDN / delta-rule 线性模型 |
| cumsum.py | `cumsum_fwd`(26) | 沿序列做前缀和 | 线性注意力归一化 |
| cumsum.py | `cumsum_bwd`(83) | 前缀和反向 | 同上 |
| layernorm\_guard.py | `layernorm_guard_fwd`(66) | 带 gate 的 LayerNorm 前向 | GDN / gated 线性模型 |
| l2norm.py | `l2norm_fwd`(27) | L2 归一化前向 | 线性注意力状态归一化 |
| l2norm.py | `l2norm_bwd_dx`(58) | L2 归一化反向 dx | 同上 |
| l2norm.py | `l2norm_bwd_dg`(78) | L2 归一化反向 dg | 同上 |
| fused\_sigmoid\_gating.py | `fused_sigmoid_gating_fwd`(23) | 融合 sigmoid 门控前向 | GDN / 门控线性模型 |
| chunk\_scaled\_dot\_kkt.py | `chunk_scaled_dot_kkt_fwd`(46) | chunk 级 K·Kᵀ 缩放点积（key 相关性） | 线性注意力 |
| kda.py | 多个 (156,251,543,649,839,1041,1204,1574) | KDA (Key-Delta-Attention) 前向/反向的分块实现（chunk 与非 chunk 的 fwd/bwd dh/dq/dk/dv） | GDN / 线性注意力 |
| fused\_recurrent.py | `fused_recurrent_fwd`(26) | 融合递归状态前向 | 线性注意力 RNN |
| fused\_recurrent.py | `fused_recurrent_bwd`(255) | 融合递归状态反向 | 同上 |
| op.py | `fwd`(30),`bwd`(54) | 通用算子前向/反向包装 | 线性注意力 |
| fused\_norm\_gate.py | `fused_norm_gate_fwd`(26) | 融合 normalize+gate 前向 | GDN |
| fused\_norm\_gate.py | `fused_norm_gate_bwd`(126) | 融合 normalize+gate 反向 | 同上 |
| chunk\_o.py | `chunk_gated_delta_rule_fwd`(41) | chunk 级 gated-delta-rule 输出前向 | GDN / 门控线性模型 |
| fused\_gdn\_prefill\_post\_conv.py | `fused_gdn_prefill_post_conv_fwd`(19) | GDN prefill 阶段卷积后处理融合前向 | Qwen GDN / OLMo GDN |

## 2. LoRA 算子 (`vllm/lora/ops/triton_ops/`)

这些算子服务于**任何加载了 LoRA adapter 的模型**（通用，与基座模型无关）。

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| lora\_expand\_fp8\_op.py | `lora_expand_fp8`(50) | FP8 LoRA 扩展（B 矩阵乘加） | 任意 FP8 + LoRA 模型 |
| lora\_expand\_op.py | `lora_expand`(23) | BF16 LoRA 扩展 | 任意 LoRA 模型 |
| fused\_moe\_lora\_op.py | `_token_compute`(19) | MoE+LoRA token 级计算 | 任意 MoE + LoRA 模型 |
| fused\_moe\_lora\_op.py | `_moe_lora_b_kernel`(36) | MoE LoRA B 矩阵乘 | 同上 |
| fused\_moe\_lora\_op.py | `_moe_lora_a_kernel`(53) | MoE LoRA A 矩阵乘 | 同上 |
| fused\_moe\_lora\_op.py | `_moe_lora_b_mode`(76) | MoE LoRA B 模式分发 | 同上 |
| fused\_moe\_lora\_op.py | `_moe_lora_bwd_dA`(121) | MoE LoRA 反向 dA | 同上 |
| fused\_moe\_lora\_op.py | `_fused_moe_lora_b_kernel`(515) | 融合 MoE LoRA B（变体） | 同上 |
| fused\_moe\_lora\_op.py | `_fused_moe_lora_a_kernel`(894) | 融合 MoE LoRA A（变体） | 同上 |
| kernel\_utils.py | `sgmv_expand`(10) | SGMV 扩展（分组 gather-MV） | LoRA |
| kernel\_utils.py | `sgmv_shrink`(110) | SGMV 收缩（分组 scatter-MV） | LoRA |
| kernel\_utils.py | `sgmv_expand_slice`(237) | SGMV 扩展切片 | LoRA |
| lora\_shrink\_op.py | `lora_shrink`(23) | BF16 LoRA 收缩 | 任意 LoRA 模型 |
| fp8\_kernel\_utils.py | `fp8_sgmv_expand`(10) | FP8 SGMV 扩展 | FP8 + LoRA |
| fp8\_kernel\_utils.py | `fp8_sgmv_shrink`(62) | FP8 SGMV 收缩 | 同上 |
| fp8\_kernel\_utils.py | `fp8_sgmv_expand_slice`(194) | FP8 SGMV 扩展切片 | 同上 |
| fp8\_kernel\_utils.py | `fp8_group_gemm`(387) | FP8 分组 GEMM | 同上 |
| fused\_moe\_lora\_fp8\_op.py | `triton_fused_moe_lora_b`(18) | FP8 MoE LoRA B 融合 | FP8 MoE + LoRA |
| fused\_moe\_lora\_fp8\_op.py | `triton_fused_moe_lora_a`(35) | FP8 MoE LoRA A 融合 | 同上 |
| fused\_moe\_lora\_fp8\_op.py | `triton_fused_moe_lora_b_mode`(52) | FP8 MoE LoRA B 模式 | 同上 |
| fused\_moe\_lora\_fp8\_op.py | `triton_fused_moe_lora_bwd_dA`(118) | FP8 MoE LoRA 反向 dA | 同上 |
| lora\_shrink\_fp8\_op.py | `lora_shrink_fp8`(87) | FP8 LoRA 收缩 | FP8 + LoRA |

## 3. Sampling / Spec-Decode / 通用 worker 算子 (`vllm/v1/sample/`, `vllm/v1/worker/`)

这些算子大部分是**通用生成/采样路径**（任何模型都会走），Mamba 相关仅用于 SSM 模型。

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| sample/ops/topk\_topp\_triton.py | `top_k_renorm`(70) | top-k 重新归一化 | 任意模型采样 |
| sample/ops/topk\_topp\_triton.py | `top_p_renorm`(93) | top-p 重新归一化 | 任意模型采样 |
| sample/rejection\_sampler.py | `get_num_draft_tokens`(714) | 计算被接受的 draft token 数 | 投机解码（任意 draft 模型） |
| sample/rejection\_sampler.py | `rejection_sample`(773) | 拒绝采样主核 | 同上 |
| sample/rejection\_sampler.py | `sample_recovered_tokens`(849) | 恢复 token 采样 | 同上 |
| sample/rejection\_sampler.py | `expand_batch_to_tokens`(872) | batch→token 展开 | 同上 |
| v1/kv\_offload/cpu/swap\_blocks\_triton.py | `swap_blocks`(24) | CPU↔GPU KV block 交换 | 任意（KV offload） |
| v1/worker/block\_table.py | `prepend_block_table`(379) | 前缀拼接 block table | 任意模型 |
| v1/worker/mamba\_utils.py | `copy_to_contiguous`(26) | 状态拷贝到连续内存 | Mamba / SSM |
| v1/worker/mamba\_utils.py | `copy_streaming`(153) | 流式状态拷贝 | 同上 |
| v1/worker/mamba\_utils.py | `copy_to_contiguous_bwd`(283) | 反向拷贝 | 同上 |
| v1/worker/mamba\_utils.py | `zero_initial_states`(329) | 初始化状态清零 | 同上 |
| v1/worker/mamba\_utils.py | `make_contiguous_states`(408) | 状态连续化 | 同上 |
| v1/worker/utils.py | `arange`(44) | 生成连续索引 (arange) | 通用 |
| v1/worker/gpu/sample/logit\_bias.py | `logit_bias_kernel`(147) | 给 logits 加偏置 | 任意模型 |
| v1/worker/gpu/sample/min\_p.py | `min_p_sampling`(8) | min-p 采样截断 | 任意模型 |
| v1/worker/gpu/sample/bad\_words.py | `bad_words`(100) | 屏蔽 bad words | 任意模型 |
| v1/worker/gpu/sample/gumbel.py | `gumbel_noise`(17) | Gumbel 噪声生成 | 任意（Gumbel 采样） |
| v1/worker/gpu/sample/gumbel.py | `gumbel_softmax`(61) | Gumbel-softmax | 同上 |
| v1/worker/gpu/sample/gumbel.py | `gumbel_sampling`(76) | Gumbel 采样 | 同上 |
| v1/worker/gpu/sample/gumbel.py | `gumbel_edm`(84) | Gumbel EDM 变体 | 同上 |
| v1/worker/gpu/sample/gumbel.py | `gumbel_sampling_v2`(161) | Gumbel 采样 v2 | 同上 |
| v1/worker/gpu/sample/prompt\_logprob.py | `prompt_logprob`(145) | 计算 prompt logprob | 任意模型 |
| v1/worker/gpu/sample/penalties.py | `frequency_penalties`(106) | 频率惩罚 | 任意模型 |
| v1/worker/gpu/sample/penalties.py | `presence_penalties`(218) | 存在惩罚 | 任意模型 |
| v1/worker/gpu/sample/logprob.py | `gather_logprobs`(16) | 收集 logprob | 任意模型 |
| v1/worker/gpu/sample/logprob.py | `compute_logprobs`(60) | 计算 logprob | 同上 |
| v1/worker/gpu/sample/logprob.py | `sample_logprobs`(188) | 采样 logprob | 同上 |
| v1/worker/gpu/model\_states/mamba\_hybrid.py | `copy_to_contiguous`(331) | Mamba-hybrid 状态连续化 | Mamba-hybrid (Jamba 等) |
| v1/worker/gpu/block\_table.py | `expand_chunked_seq_block_table`(215) | chunk 序列 block 表展开 | 任意 |
| v1/worker/gpu/block\_table.py | `expand_block_table`(255) | block 表展开 | 任意 |
| v1/worker/gpu/cp\_utils.py | `gather_input_ids`(35) | 上下文并行 gather input ids | 上下文并行模型 |
| v1/worker/gpu/input\_batch.py | 6个 (190,267,325,431,480,582,614) | 构造/拼接 input 张量（token ids、positions、slot mapping 等） | 任意模型 |
| v1/worker/gpu/metrics/logits.py | `gather_logits`(9) | 按 token 位置 gather logits | 任意模型 |

## 4. Attention 算子 (`vllm/v1/attention/`)

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| ops/dcp\_alltoall.py | `dcp_prefill_alltoall`(132) | 上下文并行 prefill all-to-all 分发 | 上下文并行 / 分离 prefill |
| ops/dcp\_alltoall.py | `dcp_decode_alltoall`(195) | 上下文并行 decode all-to-all | 同上 |
| ops/triton\_attention\_helpers.py | `_fwd_kernel_varlen` 等(21,27,44,78,109,141,241,271,378,400,417) | 各类注意力辅助/变体核（varlen、带 mask、带 logits、reduce 等） | 通用注意力 |
| ops/triton\_turboquant\_decode.py | `turboquant_decode`(43) | TurboQuant 解码阶段量化 KV | DeepSeek-V3/R1、Qwen3-Next、MiMo、GLM（TurboQuant KV） |
| ops/triton\_turboquant\_decode.py | `turboquant_recv`(321) | TurboQuant 接收反量化 | 同上 |
| ops/triton\_fp8\_mqa\_logits.py | `_fwd_kernel_fp8_mqa_logits`(48) | FP8 MQA logits 计算 | DeepSeek MLA FP8 (ROCm) |
| ops/triton\_merge\_attn\_states.py | `merge_attn_states_fwd`(58) | 合并分段注意力输出 | 上下文并行 / chunked |
| ops/triton\_merge\_attn\_states.py | `merge_attn_states_bwd`(184) | 合并注意力反向 | 同上 |
| ops/int4\_per\_token\_head.py | `int4_quant`(46),`int4_dequant`(52),`int4_dequant2`(63),`per_token_head_quant`(318),`per_token_head_dequant`(1000) | INT4 KV 量化/反量化（per-token / per-head） | INT4 KV cache 模型 |
| ops/prefix\_prefill.py | 3个 (38,106,496) | 前缀共享 prefill 注意力（chunk+fwd+bwd） | 通用 / 前缀缓存 |
| ops/common.py | `_fwd_kernel`(9),`rescale_kv`(260),`rotary_embedding`(386) | 通用注意力核、KV rescale、RoPE | 通用注意力 |
| ops/triton\_turboquant\_store.py | `turboquant_store`(25),`turboquant_store_rescale`(144),`turboquant_store_residual`(220) | TurboQuant KV 写回/重缩放/残差 | DeepSeek-V3/R1、Qwen3-Next 等 |
| ops/triton\_prefill\_attention.py | `_fwd_kernel`(36) | prefill 注意力前向 | 通用注意力 |
| ops/triton\_decode\_attention.py | `tanh`(53),`_fwd_kernel_stage1`(68),`_fwd_grouped_kernel_stage1`(278),`_fwd_kernel_stage2`(575) | 解码注意力（双阶段/分组）前向 | 通用注意力 |
| ops/triton\_reshape\_and\_cache\_flash.py | `reshape_and_cache_flash`(33),`_fwd_kernel`(154),`gather_cache`(463) | KV 缓存 reshape/写回/gather | 通用注意力 |
| ops/triton\_unified\_attention\_diffkv.py | `kernel_unified_attention_diffkv`(50),`kernel_reduce_segments_diffkv`(306) | 差分 KV (DiffKV) 统一注意力 | Qwen3-Next (DiffKV) 等 |
| ops/chunked\_prefill\_paged\_decode.py | `_fwd_kernel`(40),`_fwd_kernel_v2`(45) | chunked prefill 与 paged decode 融合 | 通用注意力 |
| ops/triton\_unified\_attention.py | 6个 (38,67,110,145,178,684) | 统一注意力主核（prefill/decode/varlen/fwd/bwd） | 通用注意力（默认 Triton 后端） |
| backends/rocm\_aiter\_fa.py | `cp_mha_gather_cache_kernel`(47),`reshape_and_cache_shuffle_kernel`(216) | ROCm AITER 缓存 gather/reshape-shuffle | ROCm 上的 MLA/注意力 |
| backends/mla/indexer.py | `_prepare_uniform_decode_kernel`(43),`kernel`(222) | MLA 解码索引准备 | DeepSeek-V2/V3/R1、Qwen3-Next (MLA) |
| backends/mla/compressor\_utils.py | `_compressed_slot_mapping_kernel`(8) | MLA 压缩 slot 映射 | 同上 |
| backends/mla/sparse\_swa.py | `sparse_mla_fwd`(295),`sparse_mla_bwd_dq`(734),`sparse_mla_bwd_dkv`(794) | 稀疏滑动窗口 MLA 前向/反向 | DeepSeek 稀疏 MLA |
| backends/mla/sparse\_utils.py | `sparse_mla_topk`(11) | 稀疏 MLA topk 选择 | 同上 |
| backends/flashinfer.py | `_trtllm_prefill_attn_kvfp8_dequant`(105),`_copy_page_indices_kernel`(2386) | TRT-LLM KV-FP8 prefill 反量化 / 页索引拷贝 | DeepSeek FP8 (TRT-LLM 路径) |
| worker/gpu/mm/rope.py | `mrope_rotary_embedding`(166) | 多模态 RoPE (mRoPE) | Qwen2-VL/2.5-VL/3-VL |
| worker/gpu/buffer\_utils.py | `copy_to_buffer`(274),`copy_from_buffer`(312) | KV buffer 拷贝 | 通用 |
| worker/gpu/structured\_outputs.py | `apply_bias`(85) | 结构化输出 bias 应用 | 任意（结构化解码） |

## 5. Fused MoE 算子 (`vllm/model_executor/layers/fused_moe/`)

服务于**所有 Mixture-of-Experts 模型**（Mixtral、Qwen-MoE、DeepSeek-MoE/V3/R1、DBRX、OLMoE、gpt-oss、MiniMax、Kimi-K2、Qwen3-Next-MoE、GLM-4.5 等）。

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| experts/nvfp4\_emulation\_moe.py | `moe_forward_kernel`(50) | NVFP4（Blackwell）MoE 模拟前向 | DeepSeek-V3/R1 FP4 路径 |
| experts/trtllm\_lora\_moe.py | `trtllm_lora_moe_a`(49),`trtllm_lora_moe_b`(73) | TRT-LLM LoRA MoE A/B | TensorRT-LLM MoE+LoRA |
| experts/fused\_batched\_moe.py | 3个 (46,191,294) | 批处理融合 MoE（grouped GEMM 变体 fwd/bwd） | 各类 MoE |
| experts/mxfp8\_native\_moe.py | `moe_forward_kernel`(56) | MXFP8 原生 MoE 前向 | MXFP8 MoE 模型 |
| experts/batched\_deep\_gemm\_moe.py | `moe_forward_kernel`(57) | DeepGEMM 批处理 MoE 前向 | DeepSeek-V3/R1 (DeepGEMM) |
| fused\_moe.py | `fused_moe_kernel`(44),`fused_moe_kernel_persistent`(64),`fused_experts`(298),`fused_moe_kernel_noaux`(1010) | 融合 MoE 专家前向（persistent/无辅助损耗变体） | 通用 MoE |
| router/base\_router.py | `moe_align_block_size`(18) | 专家 token 对齐分块 | 任意 MoE |
| router/bf16x3\_router\_gemm\_cutedsl.py | `bf16x3_router_gemm`(352) | bf16x3 路由 GEMM (CuteDSL) | DeepSeek-V3 路由 |
| router/dsv4\_topk.py | `dsv4_topk`(40) | DeepSeek-V4 top-k 路由选择 | DeepSeek-V4 |
| prepare\_finalize/deepep\_v2.py | `moe_deepep_dispatch`(427) | DeepEP-v2 专家并行分发/聚合 | DeepSeek 专家并行 |
| utils.py | `_fwd_kernel`(47),`act_and_mul`(391),`silu_and_mul`(483) | MoE 前处理 / 激活 (SiLU·mul) | 通用 MoE |
| deep\_gemm\_utils.py | 4个 (107,114,160,355) | DeepGEMM 辅助/包装核 | DeepSeek MoE |
| moe\_fused\_mul\_sum.py | `fused_mul_sum`(10) | 融合乘加求和 | MoE 路由 |

## 6. Mamba / SSM 算子 (`vllm/model_executor/layers/mamba/`)

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| ops/mamba\_ssm.py | `selective_scan_fwd`(197),`selective_scan_bwd`(203),`selective_scan_bwd_scan`(209),`selective_scan_bwd_dx`(240) | Mamba selective-scan 前向/反向 | Mamba / Mamba2 (Codestral Mamba, Jamba) |
| ops/causal\_conv1d.py | `causal_conv1d_fwd`(16),`causal_conv1d_bwd`(762) | 因果 1D 卷积 fwd/bwd | Mamba / 混合模型 |
| ops/ssd\_state\_passing.py | `ssd_state_passing`(26) | SSD 状态传递 | Mamba2 / 混合 |
| ops/triton\_helpers.py | `segsum`(7) | 段求和指数 | SSD / 线性注意力 |
| ops/ssd\_chunk\_scan.py | `ssd_chunk_scan_combined`(147) | SSD chunk scan 融合 | Mamba2 |
| ops/selective\_state\_update\_replayssm\_output\_only.py | 2个 (24,134) | 选择性状态更新（仅重放输出） | Mamba |
| ops/ssd\_bmm.py | `ssd_bmm`(64) | SSD 块矩阵乘 | Mamba2 |
| ops/layernorm\_gated.py | `layernorm_gated`(13) | 门控 LayerNorm | Mamba |
| ops/ssd\_chunk\_state.py | 2个 (28,195) | SSD chunk 状态前向/反向 | Mamba2 |
| ops/gather\_initial\_states.py | `gather_initial_states`(10) | 收集初始状态 | Mamba2 |
| gdn/qwen\_gdn\_linear\_attn.py | `fused_qk_gdn_fwd`(1679) | Qwen GDN 线性注意力融合前向 | Qwen GDN |
| gdn/olmo\_gdn\_linear\_attn.py | `fused_qk_gdn_fwd`(563) | OLMo GDN 线性注意力融合前向 | OLMo GDN |
| linear/bailing\_linear\_attn.py | `bailing_linear_attn_fwd`(144) | Bailing 线性注意力前向 | Bailing 线性注意力模型 |

## 7. 量化算子 (`vllm/model_executor/layers/quantization/`, `.../kernels/linear/`)

服务于**使用该量化格式的任意模型**。

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| awq\_triton.py | `awq_dequantize`(11),`awq_gemm`(108) | AWQ 反量化 / W4A16 GEMM | AWQ 量化模型 |
| compressed\_tensors/triton\_scaled\_mm.py | `triton_scaled_mm`(18) | 缩放矩阵乘 (W8A8/FP8) | compressed-tensors 模型 |
| compressed\_tensors/compressed\_tensors\_embedding.py | `embedding`(25) | 压缩 embedding 查表 | compressed-tensors 模型 |
| utils/nvfp4\_emulation\_utils.py | 4个 (25,46,119,137) | NVFP4 模拟量化/反量化/转换 | Blackwell FP4 模型 |
| utils/int8\_utils.py | 6个 (51,58,64,69,123,233) | INT8 量化/反量化/矩阵乘 | W8A8 INT8 模型 |
| utils/fp8\_utils.py | 5个 (61,119,305,469,735) | FP8 量化/反量化/per-tensor/per-channel GEMM | FP8 模型 (DeepSeek 等) |
| utils/mxfp8\_utils.py | `mxfp8_quantize`(96) | MXFP8 量化 | MXFP8 模型 |
| qutlass\_utils.py | `qutlass_gemm`(23) | QUTLASS GEMM 包装 | 量化 GEMM |
| kernels/linear/mxfp8/rocm\_native.py | `mxfp8_quantize`(27) | ROCm 原生 MXFP8 量化 | ROCm MXFP8 模型 |
| kernels/linear/mixed\_precision/triton\_w4a16.py | `awq_dequantize`(39) | W4A16 反量化 (GPTQ/quark) | GPTQ / W4A16 模型 |
| kernels/linear/mixed\_precision/rdna\_hybrid\_w4a16.py | `awq_dequantize`(74) | RDNA 混合 W4A16 反量化 | AMD RDNA W4A16 模型 |

## 8. 模型专用 / 杂项算子

| 文件 | 算子 (行) | 功能作用 | 可能调用的模型 |
| --- | --- | --- | --- |
| layers/lightning\_attn.py | 5个 (12,141,250,315,594) | Lightning 注意力（分块递归/门控/retention）前向与反向 | MiniMax-01 (Lightning Attention) |
| layers/fused\_qk\_rmsnorm.py | `fused_qk_rmsnorm`(9) | 融合 QK RMSNorm | Gemma3 / Gemma 系 |
| layers/fused\_qk\_norm\_rope.py | `fused_qk_norm_rope`(16) | 融合 QK-Norm + RoPE | Gemma3 |
| layers/activation.py | `silu_and_mul`(26) | SiLU 门控激活 | 通用 (SwiGLU) |
| layers/sparse\_attn\_indexer.py | `sparse_attn_indexer`(127) | 稀疏注意力索引 | DeepSeek 稀疏 MLA |
| layers/attention/sparse\_mla\_attention.py | 2个 (310,351) | 稀疏 MLA 注意力前/反向 | DeepSeek-V3.1/3.2 稀疏 |
| layers/batch\_invariant.py | 6个 (31,41,208,341,448,774) | 批不变 (batch-invariant) 矩阵乘 | GDN 系列 |
| layers/rotary\_embedding/mrope.py | `mrope_rotary_embedding`(15) | 多模态 RoPE | Qwen2-VL/2.5-VL/3-VL |
| models/gemma4.py | `gemma4_qk_norm_rope`(98) | Gemma4 融合 QK-Norm+RoPE | Gemma4 |
| models/qwen3\_vl.py | `qwen3_vl_mrope`(172) | Qwen3-VL mRoPE | Qwen3-VL |
| kernels/mhc/triton.py | 2个 (11,65) | MHC (multi-head cache / 跨层 KV 共享) 读写 | 共享 KV 模型 (MiniMax 等) |
| kernels/attention/dsa/dcp\_indexer\_cutedsl.py | `dcp_indexer`(69) | DSA 上下文并行索引 (CuteDSL) | DeepSeek 稀疏注意力 |
| kernels/triton/qkv\_padded\_fp8\_quant.py | `qkv_padded_fp8_quant`(22) | QKV 填充+FP8 量化 | FP8 KV cache (DeepSeek 等) |
| distributed/kv\_transfer/.../hf3fs/utils/gather\_scatter\_helper.py | 2个 (9,68) | HF3FS KV 传输 gather/scatter | 使用 HF3FS KV 连接器的任意模型（分离 prefill） |

# omni（v0.28.0）

| **算子名称（kernel 函数名）** | **代码出处** | **算子功能** | **被调用模型** |
| --- | --- | --- | --- |
| `mot_unified_gemm_kernel` | `vllm_omni/diffusion/layers/mot/ops/mot_gemm.py` | MoT 统一 GEMM 入口：按 Text/VAE 路由融合多组权重做矩阵乘（含 W8A8/W8A16 分派） | Bagel、MoT 扩散模型 |
| `_get_mot_pointers` | 同上 | MoT GEMM 的指针路由辅助：间接索引选择、grid 坐标与指针分派 | 内部供 `mot_unified_gemm_kernel` 调用 |
| `_core_standard_gemm` | 同上 | 标准 GEMM 计算核心（BF16/FP16、W8A8），循环后反量化 | 内部供 `mot_unified_gemm_kernel` 调用 |
| `_core_weight_only_gemm` | 同上 | 权重量化 GEMM 核心（W8A16，循环内 on-the-fly 反量化） | 内部供 `mot_unified_gemm_kernel` 调用 |
| `_mot_rms_norm_kernel` | `vllm_omni/diffusion/layers/mot/ops/mot_rms_norm.py` | MoT 按 token 路由的 RMSNorm（隐藏维度） | Bagel、MoT 扩散模型 |
| `_mot_rms_norm_qk_kernel` | 同上 | MoT 按 head 维度的 Q/K RMSNorm（支持权重共享） | Bagel、MoT 扩散模型 |
| `qk_norm_rope_kernel` | `vllm_omni/diffusion/models/sensenova_u1/fused_rmsnorm_rope.py` | Q/K 的 RMSNorm 与 RoPE 旋转位置编码融合 | SenseNova-U1、Magi-Human |
| `_small_decode_kernel` | `vllm_omni/attention/fish_kvcache_triton.py` | 短序列 KV-cache 解码注意力（单 kernel online softmax） | Fish-Speech（自回归解码） |
| `_split_partial_decode_kernel` | 同上 | 长序列分块解码的「部分计算」kernel | Fish-Speech（自回归解码） |
| `_split_combine_decode_kernel` | 同上 | 长序列分块解码的「归约合并」kernel | Fish-Speech（自回归解码） |
| `_kernel`（SnakeBeta） | `vllm_omni/model_executor/models/common/snake_activation.py` | 融合 SnakeBeta 周期激活（预计算 exp(alpha)/inv(beta) 缓冲） | Higgs-Audio-V2、CosyVoice3、Qwen2.5-Omni、Qwen3-Omni、CoVo-Audio、IndexTTS2、Qwen3-TTS |
| `_rms_norm_fwd_kernel` | `vllm_omni/model_executor/models/omnivoice/omnivoice_generator.py` | OmniVoice 专用 RMSNorm 前向 | OmniVoice |
| `_swiglu_fwd_kernel` | 同上 | OmniVoice 专用 SwiGLU 前向（`silu(gate)*up`） | OmniVoice |
| `_fused_add_rms_norm_fwd_kernel` | 同上 | OmniVoice 专用「残差相加 + RMSNorm」融合前向 | OmniVoice |
| \_rms\_norm\_rope\_kernel | vllm\_omni/diffusion/layers/fused\_qk\_norm\_rope.py | 融合 RMSNorm + packed non-interleaved RoPE：对 Q 或 K 先做 RMSNorm（FP32），再应用旋转位置编码。一个 program 处理一组 (token, head\_group) | MiniMax-H3 |
| \_adaptive\_group\_norm\_silu\_kernel | vllm\_omni/model\_executor/models/common/ops/fused\_adaptive\_group\_norm\_silu.py | 融合 AdaGN + SiLU：`SiLU(GroupNorm(x) * (1+scale) + shift)`。两遍 Welford 归约计算 mean/var，归一化后应用自适应调制和 SiLU 激活 | 通用 DiT/UNet ResBlock (AdaGN + SiLU 模式) |
| \_group\_norm\_silu\_kernel | vllm\_omni/model\_executor/models/common/ops/fused\_group\_norm\_silu.py | 融合 GroupNorm + SiLU：`SiLU(GroupNorm(x, weight, bias))`。与 AdaGN kernel 结构相同的两遍 Welford，但不包含自适应 scale/shift 调制 | HunyuanImage3 |
| batch\_matmul\_kernel | vllm\_omni/model\_executor/models/nemotron\_voicechat/nemo\_vendored/ear\_tts\_model.py | 索引式 batched matmul：对每个 batch item `b`，gather 权重矩阵 `w[y[b]]`（按 `y` 索引），计算 `w[y[b]] @ x[b]`。一个 program 处理 (batch, dout\_block)。用于 MoGHead mixture-of-Gaussians 预测头 | Nemotron VoiceChat |
| \_indexed\_scale\_shift\_kernel | vllm\_omni/diffusion/attention/ops/minimax\_h3\_modulation.py | 索引式 AdaLN 仿射变换：`x * (1 + scale[indices]) + shift[indices]`，原地写入。用于 FinalLayer 的归一化后调制 | MiniMax-H3 DIT中，FP32计算 |
| \_indexed\_gate\_kernel | 同上 | 门控残差：`x + gate[indices] * other`。用于 DiT Block 中 attention/ffn 输出的门控融合 | MiniMax-H3  DIT中，FP32计算 |
| \_rms\_norm\_indexed\_scale\_shift\_kernel | 同上 | 融合 RMSNorm + AdaLN 仿射：先做 RMSNorm（FP32 累加），再做索引式 scale/shift。用于 DiT Block 的 `norm1`/`norm2` 前置归一化 | MiniMax-H3  DIT中，FP32计算 |
| \_indexed\_gate\_rms\_norm\_scale\_shift\_kernel | 同上 | 融合门控残差 + RMSNorm + AdaLN：一步完成 `residual + gate*branch → RMSNorm → scale/shift`，输出两个张量（残差更新 + 调制输出）。用于 DiT Block 中 attention 后的 `norm2` 路径 | MiniMax-H3 DIT中，FP32计算 |
| \_qk\_rms\_norm\_rope\_exact\_kernel | vllm\_omni/diffusion/models/minimax\_h3/ops/vae/qk\_norm\_rope.py | 融合 Q/K RMSNorm + RoPE（bit-exact）：对 VAE attention 的 Q 和 K 先做 RMSNorm（FP32 归一化），再应用 RoPE 旋转位置编码。使用 `tl.inline_asm_elementwise` 内嵌 PTX 指令 `mul.rn.f16x2` 确保乘法不收缩为 FMA，保证与 eager 模式 bit-exact 一致 | MiniMax-H3 VAE中，涉及PTX指令，非cuda平台不会进入 |
| \_scaled\_residual\_exact\_kernel | vllm\_omni/diffusion/models/minimax\_h3/ops/vae/scaled\_residual.py | bit-exact 缩放残差更新：`output = residual + branch * scale`。用内嵌 PTX `mul.rn.f32` 防止 FMA 收缩，保留独立的乘-乘-加舍入边界 | MiniMax-H3 VAE中，涉及PTX指令，非cuda平台不会进入 |
| \_mul\_rn\_f32 | vllm\_omni/diffusion/models/minimax\_h3/ops/vae/scaled\_residual.py | Triton JIT helper，用 PTX `mul.rn.f32` 做严格 round-to-nearest 乘法，防止编译器将 `branch * scale` 优化为 FMA | MiniMax-H3 VAE中，涉及PTX指令，非cuda平台不会进入 |

# SGLang

## 1. Flash Linear Attention / GDN 融合算子 (`python/sglang/kernels/ops/attention/fla/`, `.../mamba/`)

服务于 **GDN (Gated Delta Net)、线性注意力与混合状态空间模型**（如 Qwen 3.5、Mamba / Mamba2、Bailing 等）。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| attention/triton_gdn_fused_proj.py | `fused_qkvzba_split_reshape_cat_contiguous_kernel`(167) | 投影结果拆分 + 形状重排 + 拼接连续化融合 | Qwen 3.5 / GDN 模型 |
| attention/triton_gdn_fused_proj.py | `_fused_qkvzba_causal_conv1d_update_contiguous_kernel`(366) | Decode 投影拆分 + Conv1D 状态更新融合 | Qwen 3.5 / GDN (CUDA Decode 路径) |
| attention/triton_gdn_fused_proj.py | `fused_qkv_split_gdn_prefill_kernel`(668) | Prefill 阶段从 packed conv 输出中解包并提取 Q/K/V 融合 | Qwen 3.5 / GDN Prefill |
| attention/fla/fused_gdn_gating.py | `fused_gdn_gating_kernel`(11) | 融合门控 $g$ 与 $\beta$ 计算 | GDN 线性注意力模型 |
| attention/fla/fused_recurrent.py | `fused_recurrent_gated_delta_rule_packed_decode_kernel`(187) | Packed Decode 核心：QKV 提取 + 门控 + recurrent 状态更新融合 | Qwen 3.5 / GDN Triton Decode |
| attention/fla/fused_sigmoid_gating_recurrent.py | `fused_sigmoid_gating_delta_rule_update_kernel`(11) | 非 packed Decode / Verify 路径中融合 Sigmoid 门控与 Delta-rule 状态更新 | GDN / 门控线性模型 |
| attention/fla/chunk_fwd.py | `chunk_gated_delta_rule_fwd_kkt_solve_kernel`(40) | 分块前向计算：融合 KKT 求解、三角求解与 W/U 矩阵重构 | GDN Chunk Prefill |
| attention/fla/chunk_delta_h.py | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`(36) | Chunk 级 Delta-rule 跨块递归前向状态传递融合 | GDN 线性模型 |
| attention/fla/fused_norm_gate.py | `layer_norm_gated_fwd_kernel`(26) | 融合 LayerNorm 归一化与门控输出 | GDN / 门控线性模型 |
| attention/fla/cumsum.py | `chunk_local_cumsum_scalar_kernel`(22), `chunk_local_cumsum_vector_kernel` | 分块局部前缀和累计（支持标量与向量分支） | 线性注意力衰减系数计算 |
| attention/fla/l2norm.py | `l2norm_fwd_kernel`(55), `gdn_prefill_qkv_prepare_kernel`(98) | L2 归一化前向 / GDN Prefill QKV 准备融合 | 开启 Q/K 归一化的 GDN 模型 |
| mamba/causal_conv1d_triton.py | `_causal_conv1d_fwd_kernel`(19), `_causal_conv1d_update_kernel`(586) | 因果 1D 卷积前向 / 解码状态更新 | Mamba / SSM / 混合模型 |
| mamba/mamba_state_scatter_triton.py | `_fused_mamba_state_scatter_with_mask_kernel`(38), `_fused_conv_window_scatter_with_mask_kernel`(96) | 带掩码的状态 Scatter 融合与卷积滑动窗口写回 | Mamba 状态追踪 |

## 2. Attention 与 KV-Cache 融合算子 (`python/sglang/kernels/ops/attention/`, `.../kvcache/`)

服务于**标准注意力、MLA (Multi-head Latent Attention)、稀疏注意力与 KV 缓存写入**。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| kvcache/triton_store_cache.py | `_triton_fused_store_flashmla_kernel`(19) | 写入 FlashMLA KV 缓存，融合 FP8/BF16 精度转换与槽位映射 | DeepSeek-V2 / V3 / R1 |
| kvcache/triton_store_cache.py | `_triton_fused_store_indexer_kernel`(159) | 索引表与 KV 缓存联合存储融合 | 稀疏检索 / DSA 模型 |
| kvcache/rope_cache.py | `_fused_qk_rope_reshape_and_cache_kernel`(24) | 融合 RoPE 旋转编码、Reshape 与 KV-Cache 写入 | 通用 Transformer |
| attention/decode_attention.py | `_fwd_kernel_stage1`(28), `_fwd_grouped_kernel_stage1`(252), `_fwd_kernel_stage2`(508) | 两阶段 Decode 注意力：分片局部点积计算 + 在线 Softmax 归约融合 | 通用 LLM Triton Decode |
| attention/extend_attention.py | `_fwd_kernel`(39), `_fwd_kernel_dense_prefill`(142) | Prefill / Extend 阶段注意力前向融合核 | 通用 LLM Prefill |
| attention/fused_qk_norm_rope_store.py | `_fused_qk_norm_rope_store_kernel`(16) | 融合 Q/K RMSNorm 归一化 + RoPE + KV 缓存写入 | Gemma4、SenseNova 等 |
| attention/fused_qk_rmsnorm_rope_gate.py | `_fused_qk_rmsnorm_rope_gate_kernel`(18) | 融合 Q/K RMSNorm + RoPE + 门控输出 | 带 QK-Norm 门控模型 |
| attention/mrope.py | `apply_interleaved_rope_kernel`(15) | 多模态交织 3D 旋转位置编码 (mRoPE) | Qwen2-VL / Qwen2.5-VL / Qwen3-VL |
| attention/dsa/triton_sparse_mla.py | `_sparse_mla_fwd_kernel`(32) | 稀疏 MLA 核心计算前向融合 | DeepSeek 稀疏注意力 (DSA) |
| attention/dsv4/fused_compress_triton.py | `_fused_ape_pool_norm_rope_kernel`(24), `_c4_decode_kernel`(78) | APE 池化 + 归一化 + RoPE 融合与压缩解码 | DeepSeek-V4 |

## 3. Fused MoE 与 Router 算子 (`python/sglang/kernels/ops/moe/`)

服务于**所有 Mixture-of-Experts 稀疏模型**（DeepSeek-V3/R1、Qwen-MoE、Mixtral、Kimi-K2/K3、MiniMax 等）。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| moe/fused_moe_triton_kernels.py | `fused_moe_kernel`(124), `fused_moe_kernel_gptq_awq`(58) | 融合 MoE 专家 GEMM + 激活函数 (SiLU/GELU)，支持 FP8/AWQ/GPTQ | 通用 MoE 模型 |
| moe/fused_moe_triton_kernels.py | `_moe_sum_reduce_kernel`(218), `_fused_append_shared_experts_kernel`(312) | 专家权重加权聚合 + 共享专家输出相加融合 | DeepSeek / Qwen-MoE 等带共享专家的模型 |
| moe/fused_moe_lora_kernel.py | `_fused_moe_lora_kernel`(40) | 融合 MoE 专家计算与 LoRA A/B 矩阵乘 | 任意加载了 LoRA 的 MoE 模型 |
| moe/moe_align_block_size.py | `moe_align_block_size_kernel`(15) | 专家 Token 分块边界对齐重组 | 通用 MoE 调度 |
| moe/moe_fused_mul_sum.py | `moe_fused_mul_sum_kernel`(10) | 专家权重相乘与跨专家求和归约融合 | MoE 路由后处理 |
| moe/router.py | `fused_moe_router_cudacore_kernel`(26), `fused_moe_router_tensorcore_kernel`(84) | 门控 Logits 投影 + Top-K 路由选择融合 | MoE Router |
| moe/sigmoid_gate_topk_renorm.py | `_sigmoid_gate_topk_renorm_kernel`(18) | Sigmoid 门控 + Top-K 选通 + 权重重归一化融合 | DeepSeek-V3 / R1 门控路由 |
| moe/minimax_m3_swiglu.py | `_swiglu_oai_kernel`(16), `_swiglu_oai_mxfp8_quant_kernel`(48) | SwiGLU 激活 + MXFP8 量化融合 | MiniMax-M3 |

## 4. Fused LoRA 算子 (`python/sglang/kernels/ops/gemm/`, `.../moe/`)

服务于**加载了多租户 / 动态 LoRA 适配器的模型**。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| gemm/qkv_lora_b.py | `_qkv_lora_b_kernel`(9) | QKV 投影与 LoRA B 矩阵乘相加融合 | 任意加载 QKV LoRA 模型 |
| gemm/gate_up_lora_b.py | `_gate_up_lora_b_kernel`(9) | MLP Gate-Up 投影与 LoRA B 矩阵乘融合 | 任意加载 MLP LoRA 模型 |
| gemm/sgemm_lora_a.py, sgemm_lora_b.py | `_sgemm_lora_a_kernel`(9), `_sgemm_lora_b_kernel`(9) | 分组 SGMV / GEMM LoRA A/B 扩展与收缩 | 通用 LoRA 批量推理 |
| gemm/kv_b_lora_absorbed.py | `_step_a_q_kernel`(111), `_step_b_q_kernel`(301) | MLA KV-LoRA 矩阵吸收（Absorbed GEMM）融合 | DeepSeek MLA + LoRA |
| gemm/chunked_embedding_lora_a.py | `_chunked_embedding_lora_a_kernel`(8) | 分块词表 Embedding 与 LoRA A 计算融合 | 词表 LoRA 适配器 |

## 5. LayerNorm / Activation / Sampling 融合算子 (`python/sglang/kernels/ops/elementwise/`, `.../layernorm/`, `.../sampling/`)

服务于**通用网络层残差连接、归一化、激活及生成采样**。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| elementwise/elementwise.py | `fused_dual_residual_rmsnorm_kernel`(25), `fused_rmsnorm_kernel`(56) | 双残差相加 + RMSNorm 归一化融合 | DeepSeek / LLaMA 系列 |
| elementwise/elementwise.py | `silu_and_mul_kernel`(88), `gelu_and_mul_kernel`(112) | 门控激活 SwiGLU / GeGLU 元素乘法融合 | 通用 MLP 层 |
| layernorm/gemma4_fused_ops.py | `_gemma_rmsnorm_residual_kernel`(14), `_gemma_qkv_rmsnorm_store`(82) | Gemma 风格残差 RMSNorm 与 QKV 归一化存储融合 | Gemma4 |
| sampling/topk_topp_sampling.py | `fused_topk_topp_sampling_kernel`(18) | Top-K 截断与 Top-P 过滤重归一化采样融合 | 任意自回归采样 |
| grammar/bitmask_ops.py | `apply_token_bitmask_inplace_kernel`(14) | 词表 Logits 原地应用语法约束 Bitmask 掩码融合 | 结构化输出 / 语法约束采样 |

## 6. 量化算子 (`python/sglang/kernels/ops/quantization/`)

服务于**低比特权重量化与 KV 缓存量化模型**。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| quantization/awq_triton.py | `awq_dequantize_kernel`(11), `awq_gemm_kernel`(108) | AWQ W4A16 反量化与矩阵乘融合 | AWQ 量化模型 |
| quantization/fp8_kernel.py | `_per_token_group_quant_8bit`(14), `scaled_mm_kernel`(120) | Per-token 分组 FP8 量化与带 Scale 矩阵乘融合 | FP8 (W8A8 / W8A16) 模型 |
| quantization/fp8_kernel.py | `_per_tensor_quant_mla_fp8_stage1`(220) | MLA FP8 Per-tensor 两阶段快速量化 | DeepSeek MLA FP8 |

## 7. 多模态与扩散模型融合算子 (`python/sglang/kernels/ops/diffusion/`, `.../mm/`)

服务于**图像生成、视频生成 Diffusion DiT 与多模态模型**。

| 文件 | 算子 (行) | 功能作用与融合点 | 可能调用的模型 |
| --- | --- | --- | --- |
| diffusion/modulate/scale_shift_triton.py | `_fused_layernorm_scale_shift_gate_select01_kernel`(72) | LayerNorm + AdaLN 仿射调制 (Scale/Shift) + 门控残差融合 | DiT (FLUX / Sana 等) |
| diffusion/norm/group_norm_silu_triton.py | `_group_norm_silu_contiguous_kernel`(20) | GroupNorm 归一化 + SiLU 激活融合 (两遍 Welford 归约) | DiT / UNet 图像生成 |
| diffusion/activation/silu_mul_bitexact.py | `_silu_mul_kernel`(15) | Bit-exact 严格精度的 SiLU·mul 门控激活融合 | 严格对齐基线的 Diffusion |
| diffusion/norm/wan_rmsnorm_silu_triton.py | `_wan_rmsnorm_silu_kernel`(18) | RMSNorm + SiLU 激活融合 | Wan2.1 视频生成 |
| mm/process/image.py | `_normalize_and_patchify_kernel`(16) | 图像归一化 + Patch 切片打包融合 | Vision Encoder (ViT) |