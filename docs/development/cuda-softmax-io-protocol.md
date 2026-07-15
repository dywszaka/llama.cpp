# CUDA Softmax input/output protocol

本文记录当前 CUDA `GGML_OP_SOFT_MAX` 的输入、算法和输出约定，只覆盖前向路径
`ggml_cuda_op_soft_max()`。`GGML_OP_SOFT_MAX_BACK`、flash-attention 内部 softmax、
采样端 CPU softmax 和其他融合路径不在本文范围内。

协议本身不以 `ggml_tensor` 结构体作为输入输出协议。当前 llama.cpp 会从 `dst`、
`dst->src[0..2]`、`dst->op_params` 和 `ctx.stream()` 提取下面这些字段；重写实现时应把
这些字段视为真正的 CUDA Softmax 协议。

## 输入

基本参数：

| 参数 | 含义 |
| --- | --- |
| `input` | CUDA 可访问的 logits 基地址，当前来自 `dst->src[0]->data`。 |
| `mask` | 可选 CUDA 可访问的 mask 基地址，当前来自 `dst->src[1]->data`；可以为 null。 |
| `sinks` | 可选 CUDA 可访问的 F32 attention sink 基地址，当前来自 `dst->src[2]->data`；可以为 null。 |
| `output` | CUDA 可访问的输出基地址，当前来自 `dst->data`。 |
| `input_dtype` | 当前必须是 `F32`。 |
| `mask_dtype` | 可选；当前只支持 `F16` 或 `F32`。 |
| `output_dtype` | 当前必须是 `F32`。 |
| `ne00..ne03` | `input` 的 4 维元素数。 |
| `ncols` | `ne00`，每个 softmax row 的列数。 |
| `ne01` | grid x 维，即 dim1 row 数。 |
| `ne02` | grid y 维，通常对应 head 数。 |
| `ne03` | grid z 维，通常对应 outer batch / token group。 |
| `nrows_x` | `ggml_nrows(input)`，当前作为派生元数据保留。 |
| `nrows_y` | `input->ne[1]`，当前作为派生元数据保留。 |
| `nheads` | `input->ne[2]`。 |
| `n_head_log2` | `1u << floor(log2(nheads))`，用于 ALiBi slope。 |
| `nb11..nb13` | `mask` 的 dim1..dim3 字节步长；没有 mask 时为 `1`。 |
| `ne12..ne13` | `mask` 的 dim2..dim3 元素数；没有 mask 时为 `1`。 |
| `scale` | F32 标量，当前从 `dst->op_params[0]` 读取。 |
| `max_bias` | F32 标量，当前从 `dst->op_params[1]` 读取，用于 ALiBi slope。 |
| `m0`, `m1` | 由 `max_bias` 和 `n_head_log2` 派生的 ALiBi slope 参数。 |
| `stream` | CUDA stream，当前来自 `ctx.stream()`。 |

维度语义：

```text
input shape: [ne00, ne01, ne02, ne03]
grid:        (ne01, ne02, ne03)
row index:   rowx = i01 + i02*ne01 + i03*ne01*ne02
row input:   input  + rowx*ncols
row output:  output + rowx*ncols
```

当前 kernel 对 `input` 和 `output` 使用 dense row layout，row 内 dim0 连续，row 与 row
之间按 `ncols` 线性推进。源码中仍有 `TODO: noncontigous inputs/outputs`；因此外部
实现不应假设现有前向 CUDA 路径支持任意 strided `input` 或 `output`。

`mask` 可以在 dim1..dim3 上带 stride。对第 `(i01, i02, i03)` 个输出 row，mask 起始
地址为：

```text
mask + (i01*nb11 + (i02 % ne12)*nb12 + (i03 % ne13)*nb13) / sizeof(mask_dtype)
```

如果没有 `mask`，mask 加法项为 0。如果没有 `sinks`，sink 只作为 null 处理。

## 算法

CUDA launch 使用：

```text
block.x = min(next_power_of_two_at_least(ncols), CUDA_SOFT_MAX_BLOCK_SIZE)
grid    = (ne01, ne02, ne03)
```

当共享内存足够时，kernel 会把 row 中间值放入共享内存，并对常见列数
`32, 64, 128, 256, 512, 1024, 2048, 4096` 使用模板特化；否则使用低共享内存路径，
中间值暂存到 `output`。

对每个 `(i01, i02, i03)` row，核心处理流程：

```text
slope = get_alibi_slope(max_bias, i02, n_head_log2, m0, m1)
max_val = sinks ? sinks[i02] : -inf

for col in 0:ncols:
    z[col] = input[rowx, col] * scale
    if mask:
        z[col] += slope * mask[row_mask, col]
    max_val = max(max_val, z[col])

sum = 0
for col in 0:ncols:
    e[col] = exp(z[col] - max_val)
    sum += e[col]

if sinks:
    sum += exp(sinks[i02] - max_val)

for col in 0:ncols:
    output[rowx, col] = e[col] / sum
```

实现细节：

- `ncols` 是归约长度，要求为正。
- `scale` 先作用于 `input`，再加 mask/ALiBi 项。
- mask 值按其 dtype 转为 F32 后参与计算。
- `sinks[i02]` 参与最大值和分母，但当前不向 `output` 写入 sink 概率；启用 sinks
  时，`output` 当前 row 内的概率和通常小于 1。
- in-place 情况下低共享内存路径会把中间指数值写入 `output` 后再归一化；外部实现若
  支持 alias，需要保持等价行为。

## 输出

输出参数和数据：

| 参数 | 含义 |
| --- | --- |
| `output` | CUDA 可访问的输出基地址，写入 F32 softmax 概率。 |
| `output_dtype` | `F32`。 |
| `output_dims` | 与 `input` 相同：`[ne00, ne01, ne02, ne03]`。 |
| `output_elements` | `ne00 * ne01 * ne02 * ne03`。 |
| `output_bytes` | `output_elements * sizeof(float)`。 |
| `row_layout` | Dense/contiguous row layout，dim0 最快。 |

第 `(i01, i02, i03)` 个输出 row 位于：

```text
output + (i01 + i02*ne01 + i03*ne01*ne02)*ncols
```

没有 `sinks` 时，每个输出 row 满足普通 softmax 约束：

```text
sum(output[row, 0:ncols]) ~= 1
```

有 `sinks` 时，未写出的 sink 概率也属于分母的一部分：

```text
sum(output[row, 0:ncols]) + sink_probability ~= 1
```

当前前向实现不写额外输出参数，也不产生 side data。若未来支持非连续 `input` 或
`output` stride，需要把对应 byte stride / element stride 明确加入协议；当前实现没有
这部分协议。

## 实验框架

`ggml/src/ggml-cuda/expt/softmax-cim.cu` 提供了与 RMS_NORM CIM 比较路径一致的占位
框架：

- `GGML_CUDA_SOFT_MAX_CIM_MODE=cuda`：默认 CUDA-only 路径。
- `GGML_CUDA_SOFT_MAX_CIM_MODE=cim`：只运行外部/CIM 占位路径，并使用其结果。
- `GGML_CUDA_SOFT_MAX_CIM_MODE=compare_cuda`：CUDA 与外部/CIM 占位路径双跑，记录
  RMSE，并使用 CUDA 结果。
- `GGML_CUDA_SOFT_MAX_CIM_MODE=compare_cim`：CUDA 与外部/CIM 占位路径双跑，记录
  RMSE，并使用外部/CIM 结果。

后续接入真实实现时，优先替换 `ggml_cim_op_soft_max()` 内部逻辑。该函数当前已经接收
完整 `ggml_cuda_soft_max_cim_params`、`dst_tensor`、输出元素数和 CUDA stream，并已把
`src0`、可选 `src1`、可选 `src2` 按原 tensor 字节数 staging 到 host，用于模拟 RPC/IO
请求边界。真实实现返回与 `output` 协议相同的 dense F32 row layout 即可。
