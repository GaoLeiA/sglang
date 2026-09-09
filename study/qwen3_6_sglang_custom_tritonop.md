# Qwen 系列在 SGLang 中的 Triton / Custom OP 清单（CUDA 重点）

参照用户提供的 `qwen3_6_custom_tritonop.md`，保留 **Env、Triton ops、Custom ops** 结构，补充与原 vLLM 清单的对应关系。本文关注用户指定的 **CUDA／Triton** 路径。

源码基线：SGLang 本地提交 `03d06a764e4a83268eefd1bafc676418f7269c89`，整理日期 2026-09-09。链接指向本仓库；行号对应该快照。

**模型范围：** 当前仓库中没有独立的 `qwen3_6.py`。本清单按参考文档涉及的 GDN 混合 Attention、MoE、MTP、视觉功能，核对 `qwen3_5.py`、`qwen3_5_text.py`、`qwen3_next.py`、`qwen3_5_mtp.py` 和 `qwen3_vl.py` 的相关调用。尚未提供目标 checkpoint 的 `config.json`，因此不能将以下所有分支断言为某个 Qwen3.6 型号的实际执行清单。最终应以 `architectures`、`model_type`、量化配置、模型规模和启动参数确定。

参考文件末尾的“走 CPU 实现”和 `torch_vacc` 条件属于原文背景；本文不将它们作为 CUDA 清单的执行要求。

## Env

未检查或启动实际 SGLang 容器。下表记录 **当前源码声明**，不是已安装镜像的 `pip freeze` 结果，也不沿用参考文件中的 vLLM 镜像版本。

| 组件 | 本仓库声明 / 状态 | 依据 |
|---|---|---|
| SGLang | 以本次 commit 为准；项目版本动态生成 | `python/pyproject.toml` 的 `dynamic = ["version"]` |
| Python | `>=3.10` | [pyproject.toml](../python/pyproject.toml) |
| PyTorch | `torch==2.13.0`，构建和项目依赖中均指定 | 同上 |
| Triton | 主项目依赖列表未直接指定版本；不能由参考镜像的 3.7.1 推导当前版本 | 同上；需在目标容器内核验 |
| Transformers | `transformers==5.12.1` | 同上 |
| NumPy | `numpy`，未固定版本 | 同上 |
| Tokenizers | `tokenizers==0.22.2` | 同上 |
| Safetensors | 主项目依赖列表未直接声明；实际解析版本待目标环境核验 | 同上 |
| OpenAI SDK | `openai==2.6.1` | 同上 |
| Requests | `requests`，未固定版本 | 同上 |
| PyZMQ | `pyzmq>=25.1.2` | 同上 |
| SGLang Kernel | `sglang-kernel==0.4.6.post1`，Python 导入名为 `sgl_kernel` | 同上 |
| FlashInfer | `flashinfer_python[cu13]==0.6.18` | 同上 |
| FlashAttention 4 | `flash-attn-4>=4.0.0b18` | 同上；不等于所有 Attention 默认走 FA4 |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]==4.6.2` | 同上 |
| DeepGEMM / DeepEP | `sgl-deep-gemm==0.1.7`、`sgl-deep-ep==0.1.2` | 同上 |
| 默认 Docker 构建基底 | `CUDA_VERSION=13.0.3`，Ubuntu 24.04 CUDA devel 镜像 | [docker/Dockerfile](../docker/Dockerfile)；构建参数可覆盖 |

### 优先级和阅读约定

- **P0-基础**：普通文本推理涉及的功能；具体优化实现仍取决于后端。
- **P0-GDN / MoE / MTP / VL / FP8 / WNA16 / 多卡**：启用该功能后优先核对，不代表所有模型无条件需要。
- **P1-条件**：替代实现、融合优化或特定路径。其功能可能必需，但该 Kernel 不是唯一实现。

“名称”区分设备 Kernel 与 Python wrapper；一个 wrapper 可能启动多个 Kernel。`@triton.jit`、CUDA JIT、AOT 扩展和外部库分别标注。调用处给出 wrapper 及关键上层位置，避免只列定义但不知道如何接入。

## Triton ops

### 1. GDN / Linear Attention

主调用链：`Qwen3_5GatedDeltaNet` 相关模型实现 → `RadixLinearAttention` → `GDNAttnBackend` → `GDNKernelDispatcher` → `TritonGDNKernel` 或其他选定后端。Prefill、Decode、Verify 可以选择不同实现。

| 名称 | 优先级 / 适用条件 | 源码文件 | 调用处 |
|---|---|---|---|
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | P0-GDN：投影结果拆分、重排、拼接；满足对应 head ratio / 路径时融合 | [triton_gdn_fused_proj.py:167](../python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py#L167) | 同文件 `fused_qkvzba_split_reshape_cat_contiguous()`；[qwen3_5.py](../python/sglang/srt/models/qwen3_5.py) 输入投影处理，以及 [gdn_backend.py](../python/sglang/srt/layers/attention/linear/gdn_backend.py) |
| `_fused_qkvzba_causal_conv1d_update_contiguous_kernel` | P1-条件：CUDA Decode 投影拆分 + Conv1D 融合；要求开关、形状和状态布局支持 | [triton_gdn_fused_proj.py:366](../python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py#L366) | 同文件 `fused_qkvzba_causal_conv1d_update_contiguous()`；[gdn_backend.py:439](../python/sglang/srt/layers/attention/linear/gdn_backend.py#L439) |
| `_causal_conv1d_fwd_kernel` | P1-条件：Conv1D Prefill 的 Triton 实现；CUDA GDN Prefill 会选择 CUDA conv wrapper，不能直接认定此 Kernel 必经 | [causal_conv1d_triton.py:19](../python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L19) | 同文件 `causal_conv1d_fn()`；CUDA 适配入口见 [mamba/causal_conv1d.py](../python/sglang/srt/layers/attention/mamba/causal_conv1d.py) |
| `_causal_conv1d_update_kernel` | P0-GDN：普通非融合 Decode 的卷积状态更新；融合路径可替换 | [causal_conv1d_triton.py:586](../python/sglang/kernels/ops/mamba/causal_conv1d_triton.py#L586) | 同文件 `causal_conv1d_update()`；[gdn_backend.py:439](../python/sglang/srt/layers/attention/linear/gdn_backend.py#L439) 的普通 conv 分支 |
| `fused_qkv_split_gdn_prefill_kernel` | P0-GDN：Prefill 中从 packed conv 输出提取 Q/K/V 的融合路径 | [triton_gdn_fused_proj.py:668](../python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py#L668) | 同文件 `fused_qkv_split_gdn_prefill()`；[gdn_backend.py:686](../python/sglang/srt/layers/attention/linear/gdn_backend.py#L686) |
| `fused_gdn_gating_kernel` | P0-GDN：计算门控 `g` / `beta`；packed Decode 等融合路径可能吸收此工作 | [fused_gdn_gating.py:11](../python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py#L11) | 同文件 `fused_gdn_gating()`；[gdn_backend.py:887](../python/sglang/srt/layers/attention/linear/gdn_backend.py#L887) |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | P0-GDN：Triton packed Decode 核心，融合 QKV 提取、门控和 recurrent 更新；其他 GDN backend / ReplaySSM 可替换 | [fused_recurrent.py:187](../python/sglang/kernels/ops/attention/fla/fused_recurrent.py#L187) | 同文件 `fused_recurrent_gated_delta_rule_packed_decode()`；[gdn_triton.py:46](../python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py#L46) |
| `fused_sigmoid_gating_delta_rule_update_kernel` | P0-GDN：非 packed Decode / 相应 Verify 更新路径，含门控与状态更新 | [fused_sigmoid_gating_recurrent.py:11](../python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py#L11) | 同文件 `fused_sigmoid_gating_delta_rule_update()`；[gdn_triton.py:138](../python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py#L138)；Verify 还需看 GDN backend |
| `l2norm_fwd_kernel` / `l2norm_fwd_kernel1` | P0-GDN：Triton chunk Prefill 开启 Q/K L2 norm 时；wrapper 按形状选择 | [l2norm.py:55](../python/sglang/kernels/ops/attention/fla/l2norm.py#L55) | 同文件 `l2norm_fwd()`；[chunk.py](../python/sglang/kernels/ops/attention/fla/chunk.py) 的 `ChunkGatedDeltaRuleFunction.forward()` |
| `chunk_local_cumsum_scalar_kernel` / `chunk_local_cumsum_vector_kernel` | P0-GDN：Triton chunk Prefill 的分块累计计算；按输入分支选择 | [cumsum.py:22](../python/sglang/kernels/ops/attention/fla/cumsum.py#L22) | `chunk_local_cumsum()`；[chunk.py:36](../python/sglang/kernels/ops/attention/fla/chunk.py#L36) |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | P0-GDN：Triton chunk Prefill 块内计算，融合 KKT、三角求解及 W/U 重建工作 | [chunk_fwd.py:40](../python/sglang/kernels/ops/attention/fla/chunk_fwd.py#L40) | 同文件 `chunk_gated_delta_rule_fwd_intra()`；`chunk_gated_delta_rule_fwd()` |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | P0-GDN：Triton chunk Prefill 状态递推 | [chunk_delta_h.py:53](../python/sglang/kernels/ops/attention/fla/chunk_delta_h.py#L53) | 同文件 `chunk_gated_delta_rule_fwd_h()`；`chunk_gated_delta_rule_fwd()` |
| `chunk_fwd_kernel_o` | P0-GDN：Triton chunk Prefill 输出计算 | [chunk_o.py:30](../python/sglang/kernels/ops/attention/fla/chunk_o.py#L30) | 同文件 `chunk_fwd_o()`；`chunk_gated_delta_rule_fwd()` |
| `_layer_norm_fwd_1pass_kernel` | P0-GDN：输出 gated RMSNorm；不能误当成无 gate 的普通 LayerNorm | [layernorm_gated.py:75](../python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L75) | `_layer_norm_fwd()` → `rms_norm_gated()` / `RMSNorm`；[qwen3_5.py](../python/sglang/srt/models/qwen3_5.py) 的 `self.norm(core_attn_out, z)` |

**与参考文件的重要差别：** CUDA 下并不存在一个通用、同名的 `qwen_gdn_attention_core` 可直接替代整条链。应划清 Conv1D、QKV 准备、gating、recurrent/chunk GDN 和 gated norm 的边界；替换其中一个融合核心，不代表外围工作全部消失。

`chunk_gated_delta_rule()` 是上层 Python 组合入口，不是单个 `@triton.jit` Kernel。当前调用链在 [chunk.py](../python/sglang/kernels/ops/attention/fla/chunk.py) 中清楚列出，不能根据旧版 FLA 文件列表机械列出已经不在主链上的独立 `solve_tril` 等 Kernel。

### 2. Full Attention / RoPE / KV 元数据

| 名称 | 优先级 / 适用条件 | 源码文件 | 调用处 |
|---|---|---|---|
| `_triton_mrope_forward_fused` | P0-基础/VL：M-RoPE，CUDA 路径要求二维 positions 且存在 mrope section | [rotary_triton.py:13](../python/sglang/kernels/ops/attention/rotary_triton.py#L13) | `triton_mrope_fused()`；[mrope.py:258](../python/sglang/srt/layers/rotary_embedding/mrope.py#L258) |
| `_fused_qk_gemma_rmsnorm_kernel` / `_fused_qk_gemma_rmsnorm_gate_kernel` | P1-条件：Qwen full-attn 的 Q/K norm 及 gate 拆分融合，普通 norm 分支可替代 | [utils.py:527](../python/sglang/srt/models/utils.py#L527) | `fused_qk_gemma_rmsnorm()` / `fused_qk_gemma_rmsnorm_with_gate()`；[qwen3_5.py](../python/sglang/srt/models/qwen3_5.py) |
| `_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1` / `_fwd_kernel_stage2` | P0-基础，条件为 full Attention 选 Triton backend；FA/FlashInfer 不要求同时执行 | [decode_attention.py:226](../python/sglang/kernels/ops/attention/decode_attention.py#L226) | `decode_attention_fwd()` → 普通/grouped wrapper；[triton_backend.py:2119](../python/sglang/srt/layers/attention/triton_backend.py#L2119) |
| `_fwd_kernel` / `_fwd_kernel_unified` / `_fwd_kernel_dense_prefill` | P0-基础，条件为 full Attention Prefill 选 Triton backend | [extend_attention.py](../python/sglang/kernels/ops/attention/extend_attention.py) | [extend_attention.py:848](../python/sglang/kernels/ops/attention/extend_attention.py#L848)；[triton_backend.py:1519](../python/sglang/srt/layers/attention/triton_backend.py#L1519)；分别服务普通、unified、dense-prefill 分支，非每次全部运行 |
| `_fused_sigmoid_mul_kernel` | P0-基础，限 Qwen full-attn 输出 gate 的对应融合分支 | [elementwise.py:380](../python/sglang/kernels/ops/elementwise/elementwise.py#L380) | `fused_sigmoid_mul()`；[qwen3_5.py:1443](../python/sglang/srt/models/qwen3_5.py#L1443)，对 Attention 输出施加 sigmoid gate |
| `alloc_extend_kernel` / `alloc_decode_kernel` | P0-基础：分页 allocator 路径的槽位分配；其他 allocator 可不同 | [allocator.py:16](../python/sglang/kernels/ops/memory/allocator.py#L16) | [paged.py:183](../python/sglang/srt/mem_cache/allocator/paged.py#L183) / `alloc_decode()` |
| `write_req_to_token_pool_triton` | P0-基础：维护请求到物理 token 槽位映射 | [common.py:9](../python/sglang/kernels/ops/memory/common.py#L9) | [allocation.py](../python/sglang/srt/mem_cache/allocation.py) 写入请求映射路径；功能对应原文 slot mapping 的一部分 |

`MambaPool.clear_slots()` 负责 GDN/SSM 等状态槽位清理，源码为 [memory_pool.py](../python/sglang/srt/mem_cache/memory_pool.py)。它与 vLLM 的 `_zero_kv_blocks_kernel` 不是同一接口：前者不能直接当成 full-attention KV block 清零的等价替代。需要核对实际内存池分配、可见长度、初始化和复用语义。

### 3. MTP / EAGLE 请求准备与验证

以下为 EAGLE 类 speculative 基础设施中的相关工作，不表示启用某个单层 MTP 后全部必经。多层、tree/chain、greedy/random、Spec V2 和状态缓存选项都会改变路径。

| 名称 | 优先级 / 适用条件 | 源码文件 | 调用处 |
|---|---|---|---|
| `assign_extend_cache_locs` / `assign_extend_cache_locs_uniform` | P0-MTP：draft/verify 扩展槽位准备；uniform 是对应等长分支 | [cache_locs.py:330](../python/sglang/kernels/ops/speculative/cache_locs.py#L330) | 同文件 `assign_extend_cache_locs_func()` / `assign_extend_cache_locs_uniform_func()`；[eagle_utils.py](../python/sglang/srt/speculative/eagle_utils.py)、[spec_utils.py](../python/sglang/srt/speculative/spec_utils.py) |
| `generate_draft_decode_kv_indices` | P0-MTP：对应 draft Decode Attention 的 KV 索引构建 | [cache_locs.py:56](../python/sglang/kernels/ops/speculative/cache_locs.py#L56) | [triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py) 及 [spec_utils.py](../python/sglang/srt/speculative/spec_utils.py) 的关联路径 |
| `fill_bonus_tokens` | P0-MTP：相关 EAGLE 路径的 bonus token 填入 | [eagle.py:14](../python/sglang/kernels/ops/speculative/eagle.py#L14) | 同文件 `fill_bonus_tokens_func()`；[eagle_worker_common.py:627](../python/sglang/srt/speculative/eagle_worker_common.py#L627) |
| `fill_accept_out_cache_loc` | P0-MTP：接受 token 对应的输出缓存位置整理 | [eagle.py:56](../python/sglang/kernels/ops/speculative/eagle.py#L56) | 同文件 `fill_accept_out_cache_loc_func()`；[spec_utils.py:757](../python/sglang/srt/speculative/spec_utils.py#L757) |
| `fill_draft_extend_prepare_buffers_kernel` | P0-MTP，限相关多层/准备路径：准备 draft extend buffers | [multi_layer_eagle.py:504](../python/sglang/kernels/ops/speculative/multi_layer_eagle.py#L504) | 同文件 `fill_draft_extend_prepare_buffers_triton()`；[multi_layer_eagle_draft_extend_cuda_graph_runner.py:729](../python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py#L729) 通过别名 `fill_draft_extend_prepare_buffers` 调用 |
| `rotate_input_ids_kernel` | P0-MTP，限多层链：推进下一层 draft 输入 token | [multi_layer_eagle.py:29](../python/sglang/kernels/ops/speculative/multi_layer_eagle.py#L29) | 同文件 `rotate_input_ids()`；[multi_layer_eagle_worker_v2.py:646](../python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py#L646) |
| `speculative_sampling_classic_kernel` | P0-MTP，限启用 rejection sampling 的随机 chain 路径 | [reject_sampling.py:6](../python/sglang/kernels/ops/speculative/reject_sampling.py#L6) | `chain_speculative_sampling_triton()`；[eagle_utils.py](../python/sglang/srt/speculative/eagle_utils.py) 根据 `speculative_use_rejection_sampling` 选择 |

**CUDA 贪心验证不放进上表：** 当前 `verify_tree_greedy_func()` 在 CUDA 分支调用 `sgl_kernel.verify_tree_greedy`，而该函数中的 Triton 分支用于 XPU。虽然仓库存在 `verify_tree_greedy_kernel_triton`，不能因此把它列作 CUDA MTP 必需 Kernel。实际 CUDA 实现列在下方 Custom ops。

### 4. MoE / 量化补充

| 名称 | 优先级 / 适用条件 | 源码文件 | 调用处 |
|---|---|---|---|
| `_router_triton_kernel` | P0-MoE：常规 CUDA softmax routing 分支；特殊 pack / 其他优化路由可能改道 | [moe_fused_gate.py:90](../python/sglang/kernels/ops/moe/moe_fused_gate.py#L90) | 同文件 `moe_fused_gate()`；[topk.py:956](../python/sglang/srt/layers/moe/topk.py#L956) 的 `_is_cuda` 分支；上层 `Qwen2MoeSparseMoeBlock` |
| `fused_moe_kernel` | P0-MoE，Triton MoE runner：专家矩阵乘法 | [fused_moe_triton_kernels.py:325](../python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L325) | `invoke_fused_moe_kernel()`；[fused_moe.py:482](../python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py#L482)；[triton.py](../python/sglang/srt/layers/moe/moe_runner/triton.py) |
| `fused_moe_kernel_gptq_awq` | P0-WNA16/MoE，仅 GPTQ/AWQ 且选择对应 Triton 专家计算 | [fused_moe_triton_kernels.py:93](../python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L93) | `invoke_fused_moe_kernel()` 的量化分支；不是所有低比特模型的通用 GEMM |
| `act_and_mul_kernel` / `_moe_sum_reduce_kernel` | P0-MoE，Triton runner 的未被其他融合覆盖的激活/归并步骤 | [fused_moe_triton_kernels.py:1083](../python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1083) | `act_and_mul_triton()` / `moe_sum_reduce_triton()`；[fused_moe.py](../python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py) |
| `_per_token_group_quant_8bit` / `_per_token_group_quant_8bit_colmajor` | P0-FP8，使用对应 Triton per-token/group 量化路径；CUDA JIT 等可能替代 | [fp8_kernel.py:156](../python/sglang/kernels/ops/quantization/fp8_kernel.py#L156) | 同文件量化 wrapper；[fp8.py](../python/sglang/srt/layers/quantization/fp8.py) / [fp8_utils.py](../python/sglang/srt/layers/quantization/fp8_utils.py) 分派链 |
| `_w8a8_block_fp8_matmul` | P0-FP8，限 Triton block FP8 matmul 路径；DeepGEMM/CUTLASS 等可替代 | [fp8_kernel.py:941](../python/sglang/kernels/ops/quantization/fp8_kernel.py#L941) | 同文件 block FP8 wrapper；[fp8_utils.py](../python/sglang/srt/layers/quantization/fp8_utils.py) 的 GEMM 后端分派 |

## Custom ops（CUDA JIT / AOT / 外部库入口）

本节沿用参考文档的广义 “Custom ops” 分类。表中部分名称是 Python 函数或外部库入口，**不是**可以直接通过 `torch.ops.<namespace>.<name>` 调用的注册符号。真正的注册 schema 可从 [common_extension.cc](../python/sglang/kernels/aot/csrc/common_extension.cc) 和相应 Python wrapper 交叉核对。

| 名称 | 优先级 / 适用条件 | 源码文件 | 调用处 |
|---|---|---|---|
| `silu_and_mul` | P0-基础：MLP 激活；CUDA JIT，未对齐形状回退 AOT；不是 Triton | [activation.py:182](../python/sglang/kernels/ops/activation/activation.py#L182)；[AOT activation.cu](../python/sglang/kernels/aot/csrc/elementwise/activation.cu) | [activation.py:92](../python/sglang/srt/layers/activation.py#L92) → `SiluAndMul.forward_cuda()`；[qwen2_moe.py](../python/sglang/srt/models/qwen2_moe.py) |
| `rmsnorm` / `gemma_rmsnorm` / residual-norm 变体 | P0-基础：decoder norm，按具体类、dtype、形状选择 JIT/AOT/其他实现 | [ops/layernorm/](../python/sglang/kernels/ops/layernorm/)、[AOT elementwise/](../python/sglang/kernels/aot/csrc/elementwise/) | [layernorm.py](../python/sglang/srt/layers/layernorm.py) 的 `GemmaRMSNorm` 等；Qwen3.5 模型创建相应层。`1 + weight` 等语义不能当作普通 RMSNorm 忽略 |
| `causal_conv1d_fwd` | P0-GDN：CUDA Prefill 的卷积入口；当前 registry 选择 CUDA JIT | [ops/mamba/__init__.py](../python/sglang/kernels/ops/mamba/__init__.py)、[causal_conv1d.py](../python/sglang/kernels/ops/mamba/causal_conv1d.py)、[JIT causal_conv1d.cuh](../python/sglang/kernels/jit/csrc/mamba/causal_conv1d.cuh) | [causal_conv1d.py:43](../python/sglang/srt/layers/attention/mamba/causal_conv1d.py#L43)；`gdn_backend.py` 在 CUDA 下替换 Prefill conv wrapper |
| `FlashInferGDNKernel` 的 chunk / decode / MTP 入口 | P0-GDN，选择 FlashInfer linear-attention 后端时 | [gdn_flashinfer.py](../python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py)，实际设备实现来自 FlashInfer | [gdn_backend.py](../python/sglang/srt/layers/attention/linear/gdn_backend.py) 的 `GDNKernelDispatcher`；不能只按 `chunk_gated_delta_rule_fi` 一个名称覆盖全部模式 |
| `CuteDSLGDNKernel` | P1-条件：选择 CuTe DSL 的 GDN 路径；不是 Triton | [gdn_cutedsl.py](../python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py)、[cutedsl_gdn.py](../python/sglang/kernels/ops/attention/cutedsl_gdn.py) | `GDNKernelDispatcher`；当前 dispatcher 对不支持 CuTe Prefill 的 GPU 可回退 Triton |
| `store_cache` | P0-基础：MHA KV buffer 写入的 CUDA JIT 路径，受 row bytes 等条件约束 | [kvcache.py:56](../python/sglang/kernels/ops/kvcache/kvcache.py#L56)；[kvcache.cuh](../python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh) | [memory_pool.py:143](../python/sglang/srt/mem_cache/memory_pool.py#L143)；功能对应原文的 KV cache 写入，但不是 vLLM block layout 的同一 ABI |
| `flash_attn_varlen_func` | P0-基础/VL，选择相应 FlashAttention 路径 | [flash_attention.py](../python/sglang/kernels/ops/attention/flash_attention.py) 及所选择的 FA3/FA4 包 | [flashattention_backend.py](../python/sglang/srt/layers/attention/flashattention_backend.py)、[vision.py](../python/sglang/srt/layers/attention/vision.py)；不使用 `_vllm_fa3_C` 命名空间 |
| `get_scheduler_metadata` | P0-基础，限需要且提供此 API 的 FA 分支 | [flashattention_backend.py](../python/sglang/srt/layers/attention/flashattention_backend.py) 的导入及 `_get_scheduler_metadata` | 同文件调度元数据准备；某些 backend 将该字段设为 `None`，不能要求所有 FA 实现提供同一接口 |
| `merge_state_v2` | P0-基础，限 Attention 分段结果需合并的路径 | [merge_attn_states.cu](../python/sglang/kernels/aot/csrc/attention/merge_attn_states.cu)、[common_extension.cc](../python/sglang/kernels/aot/csrc/common_extension.cc) | [flashattention_backend.py:3665](../python/sglang/srt/layers/attention/flashattention_backend.py#L3665) 及同文件直接调用；对应原文 `merge_attn_states` 功能 |
| `scaled_dot_product_attention` | P0-VL，限 SDPA vision backend；PyTorch 库算子 | [vision.py](../python/sglang/srt/layers/attention/vision.py) 的 `VisionSdpaAttention` | [vision.py:431](../python/sglang/srt/layers/attention/vision.py#L431)；FA/FlashInfer vision 路径可替代 |
| `verify_tree_greedy` | P0-MTP，CUDA 贪心验证；AOT 扩展 | [eagle_utils.cu:323](../python/sglang/kernels/aot/csrc/speculative/eagle_utils.cu#L323)、[common_extension.cc](../python/sglang/kernels/aot/csrc/common_extension.cc) | [eagle_utils.py:377](../python/sglang/srt/speculative/eagle_utils.py#L377) → CUDA 分支导入 `sgl_kernel.verify_tree_greedy` |
| `tree_speculative_sampling_target_only` | P0-MTP，随机 target-only tree 验证分支；与 classical chain rejection 分支区分 | [speculative_sampling.cu](../python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cu) 及 `sgl_kernel` 导出 | [eagle_utils.py](../python/sglang/srt/speculative/eagle_utils.py) 的 `sampling_fn` 选择；使用 rejection sampling 时改走上表 Triton chain 实现 |
| `top_k_top_p_sampling_from_probs` | P0-基础：普通 CUDA top-k/top-p 随机采样；本链路从 FlashInfer 导入 | [sampler.py](../python/sglang/srt/layers/sampler.py)，设备实现在 FlashInfer | [sampler.py:390](../python/sglang/srt/layers/sampler.py#L390)；不是原文 `_topk_topp_kernel` 的同名迁移 |
| `top_k_renorm_prob` / `top_p_renorm_prob` | P0-MTP/采样：相关概率过滤重归一化 | [sgl_kernel/sampling.py](../python/sglang/kernels/aot/python/sgl_kernel/sampling.py)；wrapper 可转到 FlashInfer，不能仅因包名判定总在 AOT 中执行 | [sampler.py](../python/sglang/srt/layers/sampler.py)、[eagle_utils.py](../python/sglang/srt/speculative/eagle_utils.py) |
| `topk_softmax` | P1-条件：兼容/替代入口；**当前常规 CUDA `fused_topk()` 路径已走前述 Triton router** | [moe_topk_softmax.py](../python/sglang/kernels/ops/moe/moe_topk_softmax.py)、[AOT CUDA](../python/sglang/kernels/aot/csrc/moe/moe_topk_softmax_kernels.cu) | [topk.py](../python/sglang/srt/layers/moe/topk.py)；需看分支，不能照搬参考文件把 AOT topk 列作 CUDA 必经 |
| FP8 GEMM 后端入口 | P0-FP8：取决于 block/channel/tensor scaling、硬件和配置 | [fp8.py](../python/sglang/srt/layers/quantization/fp8.py)、[fp8_utils.py](../python/sglang/srt/layers/quantization/fp8_utils.py)、[ops/gemm/](../python/sglang/kernels/ops/gemm/) | `Fp8LinearMethod.apply()` 及 `dispatch_w8a8_block_fp8_linear()` 等；CUDA 侧可能为 Triton、DeepGEMM、CUTLASS/FlashInfer，不是 `fp8_scaled_mm_cpu` |
| WNA16 GEMM / Marlin 等 | P0-WNA16：仅相应 AWQ/GPTQ/权重格式与后端组合 | [quantization/awq/](../python/sglang/srt/layers/quantization/awq/)、[quantization/gptq/](../python/sglang/srt/layers/quantization/gptq/)、[marlin_utils.py](../python/sglang/srt/layers/quantization/marlin_utils.py) | 相应 quantization method 的 `apply()`；MoE Triton 路径见上表。不能将 CPU `wna16_gemm` 当作通用 CUDA 接口 |
| `all_gatherv` / `reduce_scatterv` | P0-多卡，限相应 DP/TP token dispatcher 和组合条件 | [parallel_state.py:1355](../python/sglang/srt/distributed/parallel_state.py#L1355)、同文件 `reduce_scatterv()`；通信后端非 Triton 算子 | [token_dispatcher/standard.py](../python/sglang/srt/layers/moe/token_dispatcher/standard.py) 的 dispatch/combine；[flashinfer.py](../python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py) 也有相关路径 |
| DeepEP dispatch / combine | P0-多卡，选择 DeepEP EP 路径时 | [token_dispatcher/deepep.py](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py)、[deepep_v2.py](../python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py)，底层来自 DeepEP | MoE token dispatcher；与 all-gatherv/reduce-scatterv 路径按配置区分，不要求同时走所有通信方案 |

## 与原 vLLM 清单的对应关系

“功能对应”只表示完成类似工作，不表示张量布局、参数顺序、随机采样语义或 ABI 一致。没有同名实现时给出实际功能位置，避免为凑表臆造 SGLang 算子。

| 原文名称 | SGLang 对应 / 处理结论 |
|---|---|
| `eagle_step_slot_mapping_metadata_kernel` | 功能拆分到 `cache_locs.py` 的槽位分配、draft KV 索引以及 Attention 元数据准备；无同名一对一接口 |
| `eagle_prepare_inputs_padded_kernel` | `multi_layer_eagle.py` 的 prepare-buffer / widened-input 函数族，加 worker 准备逻辑；具体取决于 speculative 模式 |
| `eagle_prepare_next_token_padded_kernel` | `fill_bonus_tokens`、`rotate_input_ids_kernel` 等相关 token 准备职责，非同一 ABI |
| `expand_kernel` | 概率、温度等扩展在 `eagle_utils.py` 中可用 `torch.repeat_interleave` 等完成；不需要假设同名 Triton Kernel |
| `sample_recovered_tokens_kernel` | classical chain rejection 中看 `speculative_sampling_classic_kernel` 内部采样；target-only tree 是另一分支 |
| `rejection_random_sample_kernel` | `chain_speculative_sampling_triton` / `speculative_sampling_classic_kernel`，前提为选择对应 rejection 模式 |
| `rejection_greedy_sample_kernel` | CUDA：`sgl_kernel.verify_tree_greedy`；不是该 wrapper 的 XPU Triton fallback |
| `_triton_mrope_forward` | `_triton_mrope_forward_fused` → `triton_mrope_fused` |
| `_causal_conv1d_fwd_kernel` | 同名 Triton 定义存在，但 CUDA GDN Prefill 的实际入口优先核对 CUDA conv wrapper |
| `_fused_post_conv_kernel` | 相关工作分布于 `fused_qkv_split_gdn_prefill`、`fused_gdn_gating`、chunk Q/K norm 等；不能宣称某一个就是完整等价替代 |
| `_causal_conv1d_update_kernel` | 同名 Triton Kernel；融合投影+conv 分支可以覆盖相关步骤 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | SGLang 存在同名 Triton Kernel，由 `TritonGDNKernel.packed_decode()` 调用；仍需核对参数和状态 layout |
| `_compute_slot_mapping_kernel` | 分页分配器 `alloc_extend_kernel` / `alloc_decode_kernel`，加 `write_req_to_token_pool_triton` 等共同承担关联工作 |
| `_zero_kv_blocks_kernel` | 没有按该接口机械对应；检查 KV pool 生命周期。GDN `clear_slots()` 仅是相关状态清理，不是 full KV 清零的等价项 |
| `layer_norm_fwd_kernel` | `_layer_norm_fwd_1pass_kernel` 与 `rms_norm_gated()`；保留 gate、norm 顺序等语义 |
| `_bilinear_pos_embed_kernel` | 当前 `Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_*()` 以 PyTorch embedding 索引、权重乘法和求和实现，没有在该调用链发现同名 Triton Kernel |
| `rotary_kernel`（vision） | `vision.py` 中 `apply_rotary_pos_emb` / native eager 路径及相应 backend；不能套用外部 vLLM FlashAttention 安装路径 |
| `_update_min_larger_stats` | 原 top-k/top-p 实现的内部辅助函数；SGLang CUDA 采样走 FlashInfer 相关入口，无需创建同名函数 |
| `_topk_topp_kernel` | 普通 CUDA 采样：`flashinfer.sampling.top_k_top_p_sampling_from_probs`；概率过滤另看 renorm wrapper |
| `_C::silu_and_mul` | SGLang `silu_and_mul` CUDA JIT/AOT 分派；命名空间与 schema 不同 |
| `_moe_C::topk_softmax` | 常规 CUDA routing 对应 `moe_fused_gate` 的 Triton router；AOT/JIT `topk_softmax` 是其他相关入口 |
| `fused_experts`（原 CPU/XPU） | CUDA Triton runner：`fused_experts` → `_fused_moe_kernel_sequence` → `invoke_fused_moe_kernel`；也可选择其他 MoE backend |
| `qwen_gdn_attention_core`（原 CPU/XPU） | CUDA 中按 `GDNKernelDispatcher` 及外围 conv/projection/norm 组件组织，没有同名总接口 |
| `chunk_gated_delta_rule_fi` | `FlashInferGDNKernel` 的对应 Prefill 入口；Triton 替代入口为 `chunk_gated_delta_rule` 组合管线 |
| `sdpa` | PyTorch `F.scaled_dot_product_attention`，条件为选择 `VisionSdpaAttention` |
| `reshape_and_cache` | KV pool 的 `set_kv_buffer` / `_set_kv_buffer_impl` / `store_cache` 等；布局非一对一 |
| `_vllm_fa3_C::fwd` | SGLang 选择的 `flash_attn_varlen_func` 及对应库；不复用 vLLM namespace |
| `_vllm_fa3_C::get_scheduler_metadata` | FlashAttention backend 中可选 `_get_scheduler_metadata`，依所选 FA 实现决定 |
| `_C_cache_ops::reshape_and_cache_flash` | 由 SGLang KV pool 写入路径核对；不能仅因为公共 namespace 存在同名 wrapper 就认定 Qwen 必经 |
| `_C::merge_attn_states` | SGLang `merge_state_v2` 及 wrapper，需核对输出/LSE 的语义和 layout |
| `fp8_gemm`（原 CPU） | CUDA FP8 quant method 分派到 Triton / DeepGEMM / CUTLASS 等 |
| `wna16_gemm`（原 CPU） | CUDA AWQ/GPTQ/Marlin 等分支；MoE 可有 `fused_moe_kernel_gptq_awq` |
| `all_gatherv` | `GroupCoordinator.all_gatherv()`，标准/相关 dispatcher 条件调用 |
| `reduce_scatterv` | `GroupCoordinator.reduce_scatterv()`，标准/相关 dispatcher 条件调用；DeepEP 为不同分发路径 |

## 接口适配时优先固定的内容

如果后续要按这张表实现或替换算子，应以 **Python wrapper 的签名和行为** 作为第一层接口边界，再向下拆 Kernel。优先明确以下内容：

| 功能 | 接口必须约定的内容 |
|---|---|
| GDN Prefill / Decode | Q/K/V 是 packed 还是分离；每 rank head 数；`a/b` 与 `g/beta` 是否在内核内转换；状态池 shape/dtype/stride；是否原地更新 |
| GDN MTP / Verify | 候选序列或树结构、验证窗口、状态保存/回滚、接受后提交位置；普通 Decode 内核不能自动覆盖这些语义 |
| Full Attention | paged KV 布局、`page_size`、slot/token 索引、Q/K/V head 关系、mask、scale、输出 LSE |
| M-RoPE | 一维/二维 positions、mrope section、interleaved、partial rotary、Q/K 原地更新 |
| Gated Norm | RMS/LayerNorm、epsilon、`norm_before_gate`、激活函数、Gemma 的权重语义 |
| MoE | TopK IDs/weights dtype、softmax/renorm、专家权重布局、量化 scale、shared expert、dispatch/combine 边界 |
| Speculative sampling | greedy / target-only / classical rejection 区别、draft/target 概率、接受索引、bonus token、随机数与可复现要求 |
| FP8 / WNA16 | 权重打包格式、scale 粒度与排列、zero-point、激活量化、accumulator 和输出 dtype |

### 建议核对顺序

1. 先固定 checkpoint、GPU 型号、BF16/FP16/量化格式、Full Attention backend、Linear Attention 的 Prefill/Decode/Verify backend，以及是否开启 MTP。
2. 无量化、无 MTP 的文本主链先核对 norm、RoPE、Full Attention、GDN、MLP；MoE checkpoint 再核对 router 和专家计算。
3. 再加入 MTP 的输入准备、验证和 GDN 状态管理；有视觉输入时再加入视觉插值、RoPE 与 Attention。
4. 最后核对量化和多卡通信。使用运行时 trace/profiler 确认实际调用，再裁剪最终 P0 集合。

本次已进行源码定义、调用分支和文件链接的静态核对；未运行模型或 GPU profiler。因此该文档是 **CUDA 相关算子的源码适配清单**，不是某个 checkpoint 的完整运行时算子 trace，也不是对原 vLLM ABI 的兼容承诺。
