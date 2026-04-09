# NVFP4 K Cache / KQ Native CUDA 技术细节

## 1. 背景与目标

当前实现的目标不是把整条注意力都改成 NVFP4，而是先收敛一个可工作的子集：

- `K cache` 以 `NVFP4` 形式存储
- `V cache` 仍保持常规类型，当前显式禁用 `V = nvfp4`
- `KQ = K_cache(NVFP4) * Q(runtime quantized to NVFP4)` 走 CUDA native FP4/cuBLASLt 路径
- 非 `flash_attn` 路径生效
- 仅支持 CUDA 后端

这条路径的目的有两个：

1. 降低 K cache 占用
2. 验证 `KQ` 是否可以在不回退到 F16/F32 的情况下直接走 `NVFP4 * NVFP4`

当前代码状态下，质量问题已经从“明显错误输出”修到“可以正常生成可读句子”，但性能还没有超过 `f16/f16` 基线。

## 2. 当前支持范围

当前限制是代码里硬编码出来的，不是文档层面的约定：

- `K cache = nvfp4` 允许
- `V cache = nvfp4` 禁止
- `flash_attn + K cache = nvfp4` 禁止
- `V cache` 只要是量化类型，仍要求 `flash_attn`

对应代码：

- `src/llama-context.cpp`
- `common/arg.cpp`
- `tools/llama-bench/llama-bench.cpp`
- `tools/server/README.md`

这意味着当前可用组合是：

- `-ctk nvfp4 -ctv f16 -fa 0`

## 3. 整体数据流

当前实现可以分成 4 段：

1. 模型权重侧的 NVFP4 matmul 元信息绑定
2. K 写入 KV cache 时实时量化成 NVFP4，并记录 sidecar scale
3. 从 KV cache 读 K 时，把 sidecar scale 重新挂到 K view 上
4. `ggml_mul_mat(k, q)` 命中 CUDA native NVFP4 path，Q 在 matmul 前实时量化成 NVFP4

对应主路径：

- 权重侧 scale 绑定：`src/llama-model.cpp`
- K cache 分配 / 视图 / 写入：`src/llama-kv-cache-unified.cpp`
- 注意力图拼接：`src/llama-graph.cpp`
- CUDA native NVFP4 matmul：`ggml/src/ggml-cuda/nvfp4-matmul.cu`
- CUDA `set_rows` NVFP4 写入：`ggml/src/ggml-cuda/set-rows.cu`

## 4. NVFP4 的 scale 语义

### 4.1 两级 scale

当前实现里，NVFP4 的恢复不是靠一个 scale 完成，而是两级：

1. block 内的 `E4M3` scale，存放在 `block_nvfp4::e`
2. block 外的 `global_scale`

量化时采用：

`x ~= q_fp4 * (scale_block / global_scale)`

其中：

- `q_fp4` 是 4-bit FP4 codebook 值
- `scale_block` 是每个 16-value block 的 `E4M3` scale
- `global_scale` 是当前 row 的外部放大系数

当前实现里真正写入 sidecar 的不是 `global_scale`，而是：

`input_scale = 1 / global_scale`

原因是 matmul 输出端最终需要乘回这个补偿量。

### 4.2 global scale 公式

当前代码使用：

`global_scale = (FP4_MAX * E4M3_HALF_MAX) / amax = 1344 / amax`

常量定义在：

- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/nvfp4-matmul.cu`

其中：

- `FP4_MAX = 6`
- `E4M3_HALF_MAX = 224`

这里必须用 `224`，不是 `448`。原因是当前 NVFP4 路径在解码 `block.e` 时采用的是 `e4m3_to_fp32_half()` 语义，也就是 E4M3 解码值再乘 `0.5`。

## 5. K cache 的内存布局

### 5.1 主缓存

每层 K cache 仍是一个 3D tensor：

- 形状：`[n_embd_k_gqa, kv_size, n_stream]`
- 数据类型：`GGML_TYPE_NVFP4`

对应代码：

- `src/llama-kv-cache-unified.cpp`

### 5.2 sidecar scale

当 `type_k == GGML_TYPE_NVFP4` 时，额外分配一份 `k_scale`：

- 类型：`F32`
- 逻辑上按 `slot` 存一标量
- 物理大小：`kv_size * n_stream`

对应代码：

- `src/llama-kv-cache-unified.cpp`
- `src/llama-kv-cache-unified.h`

`k_scale` 的语义是每个 slot 的 `input_scale = 1 / global_scale`。

## 6. K 写入 KV cache 的实现

### 6.1 图层入口

`cpy_k()` 在 `supports_set_rows` 分支中调用：

- `ggml_set_rows(ctx, k, k_cur, k_idxs)`

如果该层有 `k_scale`，则把 `k_scale` 绑定到这个 `set_rows` 结果 tensor 上：

- `ggml_tensor_set_nvfp4_scale(res, layers[ikv].k_scale)`

对应代码：

- `src/llama-kv-cache-unified.cpp`

### 6.2 CUDA set_rows NVFP4 分支

真正的量化发生在：

- `ggml/src/ggml-cuda/set-rows.cu`

当前逻辑是：

1. 先按输入 row 计算 `amax_rows[i]`
2. 对每个 row 求：
   - `global_scale_i = 1344 / amax_rows[i]`
   - `input_scale_i = 1 / global_scale_i`
3. 对 row 内每个 16-value block：
   - 计算 `vmax`
   - 编码 `scale_block = global_scale_i * (vmax / 6)`
   - 量化每个值到 FP4
4. 把 `input_scale_i` 写到目标 slot 对应的 `k_scale[dst_row]`

关键点：

- 现在是“按 row/token 单独算 scale”，不是“整批 token 共用一个 scale”
- 这是这轮修复最关键的点

### 6.3 为什么要改成 per-row

之前的实现把一整个 `set_rows` batch 的 token 共用一个 `amax/global_scale`。这在 prompt 阶段尤其有问题，因为一次 prompt 可能同时写入多个 token：

- 某个 token 的幅值较大时，会拉低整批 token 的 `global_scale`
- 其余 token 被迫共享这个较差的量化尺度
- K cache 进入注意力时误差显著放大

修复后，K cache 写入精度不再依赖 `-ub 1`。

## 7. 从 KV cache 读 K 的实现

`get_k()` 会返回一个 4D K view：

- 形状：`[n_embd_head_k, n_head_kv, n_kv, n_stream]`

如果该层存在 `k_scale`，则同时构造一个 `scale view`：

- 形状：`[n_kv, 1, 1, n_stream]`

并把它绑定到返回的 K view 上：

- `ggml_tensor_set_nvfp4_scale(res, scale)`

对应代码：

- `src/llama-kv-cache-unified.cpp`

这样后续注意力图只拿到 `k` 本身，也能通过 tensor 元信息取回这个 sidecar scale。

## 8. 注意力图中的 KQ 路径

在非 `flash_attn` 路径里，注意力图会做：

1. `kq = ggml_mul_mat(ctx0, k, q)`
2. 如果 `k` 挂了 NVFP4 sidecar scale，则再做：
   - `kq = ggml_mul(ctx0, kq, k_scale)`

对应代码：

- `src/llama-graph.cpp`

这里的 `k_scale` 就是前面说的 `input_scale_k = 1 / global_scale_k`。

这一步的意义是把 K 侧在量化时引入的外部放大量补回来。也就是说：

- `K` 侧补偿在图里做
- `Q` 侧补偿在 CUDA native matmul 里做

## 9. KQ 的 CUDA native NVFP4 路径

### 9.1 触发条件

`ggml_cuda_mul_mat_nvfp4_native()` 当前要求：

- `src0->type == GGML_TYPE_NVFP4`
- `src1->type == GGML_TYPE_F32`
- `dst->type == GGML_TYPE_F32`
- 非转置、连续布局
- `M/K` 对齐到 16
- `K % QK_NVFP4 == 0`
- 仅 CUDA + cuBLASLt + FP4 toolchain

对应代码：

- `ggml/src/ggml-cuda/nvfp4-matmul.cu`

### 9.2 src0 的处理

`src0` 是来自 K cache 的 NVFP4 tensor。

由于 ggml 的 `block_nvfp4` 是“数据字节 + scale 字节”交错布局，而 cuBLASLt 的 `CUDA_R_4F_E2M1` 只接受纯 FP4 packed data，因此 native path 会把 `src0` 拆成两条 channel：

1. `data channel`
2. `scale channel`

拆分逻辑在：

- `ggml_cuda_nvfp4_split_blocks_cuda()`

### 9.3 src1 的处理

`src1` 是 Q，输入时仍是 `F32`。

当前 native path 会在 matmul 前把 Q 实时量化成 NVFP4：

1. 先按每个 query row 计算 `amax_rows`
2. 每个 row 独立算 `global_scale_q`
3. 每个 row 独立量化成 NVFP4
4. 同时保存每个 row 的：
   - `input_scale_q = out_scale / global_scale_q`

这里的 `out_scale` 目前来自权重侧 `weight_scale` 绑定；对 KQ 路径来说通常是 `1.0`

关键点：

- 现在也是 per-row，不再整批 Q 共用一个 scale

### 9.4 cuBLASLt 调用

native path 最终调用 `cublasLtMatmul()`，A/B 两侧都设置：

- `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`

也就是：

- packed FP4 data 走 `CUDA_R_4F_E2M1`
- block scale 走单独的 E4M3 scale channel

### 9.5 输出补偿

当前补偿分成两部分：

1. `K` 侧：
   - matmul 之后在图里乘 `k_scale`
2. `Q` 侧：
   - 如果是静态绑定 scale，则直接放进 `alpha`
   - 如果是 runtime dynamic scale，则先用 `alpha = 1` 或仅保留静态补偿，再在 matmul 后对每个输出列乘 `input_scale_q`

对应代码：

- `ggml/src/ggml-cuda/nvfp4-matmul.cu`
- `src/llama-graph.cpp`

这样做的原因是：

- `Q` 的 `global_scale_q` 现在是每列一个值
- 单个 `alpha` 无法表达 per-column scale
- 所以必须改成 matmul 后按列补偿

## 10. batched / permuted KQ 的处理

真实注意力图里，`k` 往往不是简单的 2D 连续矩阵，而是经过 `reshape/permute/view` 的 4D 张量切片。native path 对这类输入的处理是：

1. 按 batch/head 取 `src0_slice/src1_slice/dst_slice`
2. 如果 slice 不是连续矩阵，则先 materialize 成临时连续矩阵
3. 再递归调用 2D native path

对应代码：

- `ggml_cuda_nvfp4_make_matrix_slice()`
- `ggml_cuda_nvfp4_materialize_contiguous_matrix()`
- `ggml_cuda_mul_mat_nvfp4_native()`

## 11. 调试过程中修掉的一个关键 bug

### 11.1 现象

一开始单测能过，但真实推理里：

- `KQ = NVFP4*NVFP4` 输出严重退化
- batched / permuted 的真实图路径比最小 2D case 更差

### 11.2 根因

问题出在 `src0` repack cache。

此前 repack cache 只根据：

- `src0->data`
- `ne[]`
- `nb[]`

来判断是否可复用。

但 batched/permuted 路径里，materialize 后的临时连续 slice 使用的是 CUDA pool 临时内存：

- 不同 slice 可能复用同一个地址
- repack cache 会把“不同 head 的临时 slice”误认为同一个矩阵
- 最终把错误的 repacked K 输入送给 cuBLASLt

### 11.3 修复

现在的策略是：

- 只有 `src0->buffer != nullptr` 的稳定张量才允许进入 repack cache
- materialized 临时 slice 会显式设置 `slice.buffer = nullptr`
- 因而临时 slice 不参与 repack cache

对应代码：

- `ggml/src/ggml-cuda/nvfp4-matmul.cu`

这是把 `backend-permuted-lhs` 回归用例拉回正确结果的关键修复。

## 12. 为什么之前 `-ub 1` 能恢复质量

在修复 per-row scale 之前，`-ub 1` 的意义不是“native path 本身更正确”，而是它绕开了一个实现缺陷：

- 默认 prompt 处理会把多个 token 放进同一批
- K cache 写入和 Q 动态量化都用“整批共用一个 scale”
- `-ub 1` 把这个批人为切成单 token
- 于是“整批 scale”退化成“单 token scale”，质量自然恢复

这说明根因是 scale 粒度错了，而不是仅仅某个常量取值不对。

## 13. 测试与验证

### 13.1 单测

当前新增/扩展的关键测试在：

- `tests/test-nvfp4-matmul.cu`

覆盖点包括：

- 基础 native FP4 matmul
- integration-style 布局
- dynamic rhs
- batched dynamic rhs
- permuted lhs
- per-row dynamic scale

本地当前结果：

- `build_cuda_wt/bin/test-nvfp4-matmul` 全通过

### 13.2 smoke

使用统一 prompt：

`Write one short sentence about CUDA.`

当前观察到：

- 修复前，默认 `ubatch` 下 `KQ=NVFP4` 会退化成重复文本或明显异常输出
- 修复后，默认 `ubatch=256` 下 `KQ=NVFP4` 已能输出正常句子

本地最近一次 smoke：

- `KQ=NVFP4, -ctk nvfp4 -ctv f16`
  - 输出正常
  - `prompt_ms ~= 110.7`
  - `decode ~= 26.3 tok/s`
- 基线 `f16/f16`
  - 输出正常
  - `prompt_ms ~= 36.9`
  - `decode ~= 87.4 tok/s`

结论：

- 正确性已明显改善
- 性能目前仍落后于基线

## 14. 当前已知限制

### 14.1 功能限制

- `V cache = nvfp4` 当前关闭
- `flash_attn + K cache = nvfp4` 当前不支持
- 仅 CUDA 路径支持

### 14.2 性能限制

当前 native `KQ NVFP4*NVFP4` 还存在明显额外开销，主要来自：

- Q 的 runtime per-row quantization
- B channel repack
- dynamic per-column post-scale
- batched/permuted 场景下的 materialize

因此现在更像是“正确性验证版本”，不是性能最优版本。

### 14.3 测试环境限制

当前 `test-nvfp4-matmul` 假设编译工具链和运行 GPU 都支持 native FP4/cuBLASLt。对不支持 FP4 的工具链/GPU，还需要额外补 gate/skip 逻辑。

## 15. 下一步建议

如果继续做这条线，优先级建议如下：

1. 先把 `test-nvfp4-matmul` 对旧 toolkit / 非 FP4 GPU 的 gate 和 skip 补齐
2. 再做 native path 的性能 profiling
3. 重点看 Q runtime quant + post-scale 是否可以融合
4. 再决定是否重新打开 `V cache = nvfp4`

在当前状态下，最重要的结论是：

- `K cache` 的 NVFP4 写入和 `KQ` 的 NVFP4*NVFP4 主链路已经能工作
- 关键正确性问题来自 scale 粒度和临时 slice 的 repack cache 复用
- 这两处都已经在当前代码中修正
