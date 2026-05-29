# FP4 KQ Outlier 计算流程

本文档描述当前 CUDA 实验路径中，`K * Q` 走 FP4/NVFP4 时，K-cache
outlier sidecar 如何参与计算。重点覆盖：

- K 的 outlier 抽取；
- K residual 的 NVFP4 量化；
- Q 的运行时 NVFP4 量化；
- FP4 矩阵乘；
- outlier correction 如何补回 KQ。

主要源码入口：

- `src/llama-kv-cache-unified.cpp`
- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-set-rows.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-matmul.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/kcache-outlier.cu`

## 1. 总体路径

当前 FP4 KQ outlier 路径可以概括为：

```text
K cache: NVFP4 residual cache + sparse outlier sidecar
Q:       F32 输入，在 KQ runtime 量化成 NVFP4
KQ:      cuBLASLt FP4 matmul，F32 accumulate / F32 output
fixup:   用 sidecar 中的 outlier 对 KQ 做 sparse correction
```

数学上想恢复的是：

```text
K_original = K_residual + K_outlier_sparse

K_original * Q
  = K_residual * Q
  + K_outlier_sparse * Q
```

实现上 dense 部分走 FP4 matmul，sparse outlier 部分单独用原始 F32 Q 做
correction。

F16 K-cache outlier 实验复用了同样的 `count/index/value` sidecar 结构和
correction kernel 逻辑，但它不涉及 NVFP4 K 量化和 native FP4 KQ。

## 2. K Cache 和 Sidecar 结构

每层主 K cache tensor：

```text
cache_k_l{layer}: [n_embd_k_gqa, kv_size, n_stream]
```

NVFP4 K cache 还会有一份 K scale sidecar：

```text
cache_k_gscale_l{layer}: F32[kv_size * n_stream]
```

虽然名字里是 `gscale`，当前存储语义是：

```text
k_scale[dst_row] = input_scale = 1 / global_scale
```

开启 K-cache outlier 后，每层额外分配：

```text
cache_k_outlier_count_l{layer}: I32[kv_size * n_stream]
cache_k_outlier_index_l{layer}: I32[max_outliers, kv_size * n_stream]
cache_k_outlier_value_l{layer}: F32[max_outliers, kv_size * n_stream]
```

默认逻辑结构是固定 per-row slot：

```text
count[kv_pos]       = 当前 K row 检测到的 outlier 数量
index[kv_pos][slot] = outlier 在完整 K row 里的维度编号
value[kv_pos][slot] = outlier 的原始 signed K 值
```

开启 `LLAMA_NVFP4_KCACHE_OUTLIER_COMPACT=1` 后，sidecar 改为固定容量稀疏池：

```text
cache_k_outlier_count_l{layer}:  I32[kv_size * n_stream]
cache_k_outlier_offset_l{layer}: I32[kv_size * n_stream]
cache_k_outlier_cursor_l{layer}: I32[n_stream]
cache_k_outlier_index_l{layer}:  I32[capacity, n_stream]
cache_k_outlier_value_l{layer}:  F32[capacity, n_stream]
```

compact 逻辑结构是：

```text
count[kv_pos]        = 当前 K row 已存储 outlier 数量
offset[kv_pos]       = 当前 K row 在 sparse pool 中的起始位置，-1 表示没有存储段
index[offset+i]      = outlier 在完整 K row 里的维度编号
value[offset+i]      = outlier 的原始 signed K 值
cursor[stream]       = sparse pool 的单调追加位置
```

compact pool 容量由 `LLAMA_NVFP4_KCACHE_OUTLIER_CAPACITY_RATIO` 控制：

```text
capacity = max(kv_size, ceil(kv_size * n_embd_k_gqa * ratio))
```

compact pool 在 KV cache 生命周期内单调追加，row 重写时会更新该 row 的
`offset/count`，但旧 pool 段不回收；`clear(true)` 清零 KV buffer 后 cursor 也归零。
如果 pool 容量耗尽，后续 row 会保留 count/offset 状态中的可存储部分，未存储 outlier
无法参与 KQ correction。

这里的 `index` 不是 byte offset，也不是整个 tensor 的扁平 offset，而是
当前 K row 内的维度坐标：

```text
n_embd_k_gqa = n_head_kv * head_dim
index range  = [0, n_embd_k_gqa)
```

`llama_kv_cache_unified::cpy_k()` 会把 `k_scale` 和 outlier sidecar 绑定到
`ggml_set_rows` 节点上。后续 `llama_kv_cache_unified::get_k()` 返回 K view
时，也会把对应 sidecar view 绑定到这个 K view 上，供 `ggml_mul_mat(k, q)`
的 CUDA 路径读取。

## 3. K 写入时的 Outlier 抽取

K 写入 KV cache 时，图层入口是：

```text
llama_kv_cache_unified::cpy_k()
  -> ggml_set_rows(ctx, k, k_cur, k_idxs)
  -> CUDA set_rows
```

当目标 K cache 是 NVFP4，并且 outlier sidecar 已绑定时，
`ggml_cuda_set_rows_nvfp4_common()` 会先调用：

```text
ggml_cuda_nvfp4_kcache_outlier_extract()
```

核心逻辑：

```cpp
counts[dst_row] = 0;

for col in 0 .. ne00-1:
    v = K_f32[row][col]

    if abs(v) > threshold:
        slot = atomicAdd(counts + dst_row, 1)

        if slot < max_outliers:
            index[dst_row][slot] = col
            value[dst_row][slot] = v
    else:
        residual_amax[row] = max(residual_amax[row], abs(v))
```

关键点：

- `threshold` 来自 outlier 实验开关，默认是 `16`。
- `count[dst_row]` 记录检测到的真实 outlier 数量。
- 如果 `count[dst_row] > max_outliers`，说明 sidecar 槽位溢出，只保存了前
  `max_outliers` 个 outlier。
- `value` 保存的是原始 signed F32 K 值。
- `residual_amax` 只统计非 outlier 值。启用 NVFP4 K-cache outlier 后，K 写入
  默认使用 threshold 作为 per-tensor amax。

## 4. K Residual 的 NVFP4 量化

outlier 抽取完成后，同一个 `set_rows` 会继续写主 K cache。此时传给
NVFP4 quant kernel 的 `zero_outliers=true`。

每个 K 元素先做 residual 化：

```cpp
raw_xi = K_f32[row][col]
xi = abs(raw_xi) > threshold ? 0.0f : raw_xi
```

也就是说，被 sidecar 抽走的 outlier，在主 K cache 中写成 `0`。

然后对 residual row 做 NVFP4 量化。原始 NVFP4 K cache 会按 row residual
amax 计算：

```text
global_scale = 1344 / residual_amax[row]
block_scale  = quantize_e4m3(global_scale * block_absmax / 6)
fp4_code     = nearest_nvfp4_code(xi * global_scale / block_scale)
```

启用 `LLAMA_NVFP4_KCACHE_OUTLIER=1` 后，K 写入 cache 使用 threshold 作为
per-tensor amax，所有写入 row 共用同一个 global scale：

```text
global_scale = 1344 / threshold
block_scale  = quantize_e4m3(global_scale * block_absmax / 6)
fp4_code     = nearest_nvfp4_code(xi * global_scale / block_scale)
```

其中：

- `1344 = FP4_MAX * E4M3_HALF_MAX = 6 * 224`
- block 内每 16 个值共享一个 E4M3 scale byte；
- FP4 数据本身按 2 个 4-bit 值打包到 1 byte。

同时写入 K scale sidecar。原始 per-row 模式：

```text
k_scale[dst_row] = residual_amax[row] / 1344
```

NVFP4 K-cache outlier 模式：

```text
k_scale[dst_row] = threshold / 1344
```

写入完成后，K 的表达可以理解为：

```text
K_original = K_residual_nvfp4 + K_outlier_sparse
```

主 K cache 只保存 residual NVFP4，outlier 值只保存在 sidecar 中。

## 5. Q 的运行时 NVFP4 量化

KQ 的输入 Q 仍然是 F32。native NVFP4 KQ 路径在
`ggml_cuda_mul_mat_nvfp4_native()` 内部把 Q 量化成 NVFP4。

动态 scale 模式下，普通 native NVFP4 matmul 仍然按每个 Q row 独立计算：

```text
amax_q[row]        = max(abs(Q[row][:]))
global_scale_q     = 1344 / amax_q[row]
input_scale_q[row] = out_scale / global_scale_q
Q_nvfp4[row]       = quantize_nvfp4(Q_f32[row], global_scale_q)
```

当 K operand 绑定了 NVFP4 K-cache outlier sidecar 且
`LLAMA_NVFP4_KCACHE_OUTLIER=1` 时，Q 改为对当前 Q 矩阵计算一个
动态 per-tensor amax，并用同一个 global scale 量化所有 Q row：

```text
amax_q_tensor      = max(abs(Q[:][:]))
global_scale_q     = 1344 / amax_q_tensor
input_scale_q[row] = out_scale / global_scale_q
Q_nvfp4[row]       = quantize_nvfp4(Q_f32[row], global_scale_q)
```

这个 Q scale 来自 Q 自身的运行时数据，不使用 K outlier threshold。

如果存在 bound input scale，则使用绑定的 scale，而不是对 Q row 动态计算
`amax_q`。

ggml 的 `block_nvfp4` 布局是：

```text
packed FP4 data + in-band E4M3 scale byte
```

但 cuBLASLt 的 FP4 matmul 需要 data channel 和 scale channel 分开。因此
native KQ 会把 K 和 Q 都拆成：

```text
data channel:  packed CUDA_R_4F_E2M1 values
scale channel: E4M3 scale bytes
```

K 侧可以使用 repack cache，因为 K cache 会被重复读取。Q 侧是 runtime 输入，
每次 KQ 前现场量化并 repack。

## 6. FP4 矩阵乘

Q 量化、K/Q repack 完成后，native 路径调用 cuBLASLt FP4 matmul：

```text
dst = K_residual_nvfp4 * Q_nvfp4
```

输出是 F32，accumulate 也是 F32。必要时会对动态 token 维度做 padding，以满足
cuBLASLt 的对齐要求。

此时 `dst` 中只有 dense residual 的贡献，缺少被置零的 outlier 部分：

```text
missing = K_outlier_sparse * Q_original_f32
```

另外，NVFP4 K cache 在图上还有一步 K-side scale compensation：

```text
kq = ggml_mul(kq, k_scale)
```

outlier correction 是在 CUDA matmul 内部、这一步图级 `kq * k_scale` 之前加到
`dst` 上的。

## 7. Outlier Correction

native FP4 KQ 返回后，`ggml_cuda_mul_mat_nvfp4_native()` 会检查 K tensor 是否
携带 outlier sidecar：

```text
outlier_counts
outlier_indices
outlier_values
```

如果存在，就调用：

```text
ggml_cuda_nvfp4_kcache_outlier_apply_correction()
```

correction kernel 中，一个 CUDA thread 对应一个输出元素：

```text
KQ[kv_pos, q_pos]
```

核心逻辑：

```cpp
kv_pos = idx % kv_len;
q_pos  = idx / kv_len;

gqa     = q_heads / kv_heads;
kv_head = q_head / gqa;

head_begin = kv_head * head_dim;
head_end   = head_begin + head_dim;

n = min(count[kv_pos], max_outliers);
corr = 0;

for i in 0 .. n-1:
    global_dim = index[kv_pos][i];

    if global_dim < head_begin || global_dim >= head_end:
        continue;

    local_dim = global_dim - head_begin;
    corr += value[kv_pos][i] * Q_f32[q_pos][local_dim];

if k_scale exists:
    corr /= k_scale[kv_pos];

KQ[kv_pos][q_pos] += corr;
```

### 7.1 为什么要按 head 过滤

`index` 存的是完整 K row 内的位置：

```text
[0, n_head_kv * head_dim)
```

而当前 KQ slice 是某个 query head 的计算。GQA 下，一个 query head 映射到一个
KV head：

```text
kv_head = q_head / (q_heads / kv_heads)
```

因此 correction 只能使用当前 KV head 范围内的 outlier：

```text
[kv_head * head_dim, (kv_head + 1) * head_dim)
```

落在其他 KV head 的 outlier 必须跳过。

### 7.2 为什么 correction 读 F32 Q

outlier sidecar 中保存的是被主 K cache 置零的大值 K。补偿这部分时直接使用
原始 F32 Q：

```text
corr += outlier_value * Q_f32[local_dim]
```

这样 outlier 路径不会再引入一次 Q 的 FP4 量化误差。

### 7.3 为什么要除以 k_scale

这是 NVFP4 K cache 的 scale 时序决定的。

源码路径中，correction 是在 CUDA matmul 内部加到 `dst` 上的；但图上后面还会对
整个 KQ 结果执行：

```text
kq = kq * k_scale
```

如果直接把真实 correction 加到 `dst`，后面会被 `k_scale` 再乘一次，导致
outlier correction 被错误缩放。

所以 CUDA correction 内部先做：

```text
dst += corr / k_scale
```

图上的后续 scale multiply 之后变成：

```text
(corr / k_scale) * k_scale = corr
```

这样最终输出中的 sparse outlier 贡献才是正确的。

## 8. End-To-End 顺序

完整执行顺序：

1. F32 K row 通过 `ggml_set_rows` 写入 KV cache。
2. 如果 outlier sidecar 开启，先抽取 `abs(K) > threshold` 的值。
3. sidecar 写入 `count/index/value`。
4. 被抽取的位置在 residual K 中置零。
5. residual K row 量化成 NVFP4。
6. 每个 K row 写入 `k_scale = 1 / global_scale`。
7. attention 读取 K view，并携带 `k_scale` 和 outlier sidecar view。
8. native KQ 在 runtime 把 F32 Q 量化成 NVFP4。
9. K/Q NVFP4 block 拆成 cuBLASLt FP4 data/scale channel。
10. cuBLASLt 计算 dense residual `K * Q`，输出 F32 KQ。
11. CUDA correction 用 sidecar 中的 outlier 和原始 F32 Q 补回 sparse 贡献。
12. 图上继续执行 `kq = kq * k_scale`，完成 K-side scale compensation。

## 9. 正确性关键点

这条路径正确依赖以下不变量：

- 抽取和 correction 使用同一套 K row 坐标系，`index` 不能被解释成 byte
  offset。
- outlier 在主 residual K cache 中必须写成 `0`。
- `count` 必须记录真实检测数量，用于 correction 读取有效 slot 和判断 overflow。
- correction 必须按 GQA 映射过滤到当前 KV head。
- NVFP4 K path 中，correction 必须在图级 `kq * k_scale` 前预除以 `k_scale`。
- 如果 `count > max_outliers`，当前 row 的 correction 不完整，PPL/精度结果需要
  结合 overflow 统计判断。
