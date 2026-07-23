# CUDA RMS Norm input/output protocol

本文记录当前 CUDA `GGML_OP_RMS_NORM` 的输入、算法和输出约定，只覆盖未融合路径
`ggml_cuda_op_rms_norm()`。融合的 RMS Norm + multiply、反向传播和其他 Norm 算子不在
本文范围内。

协议本身不以 `ggml_tensor` 结构体作为输入输出协议。当前 llama.cpp 会从 `dst`、
`dst->src[0]`、`dst->op_params` 和 `ctx.stream()` 提取下面这些字段；重写实现时应把
这些字段视为真正的 CUDA RMS Norm 协议。

## 输入

基本参数：

| 参数 | 含义 |
| --- | --- |
| `input` | CUDA 可访问的输入基地址，当前来自 `dst->src[0]->data`。 |
| `output` | CUDA 可访问的输出基地址，当前来自 `dst->data`。 |
| `dtype` | 输入和输出当前都必须是 `F32`。 |
| `ne0` | dim0 元素数，即每个向量的列数；RMS Norm 在该维度上归约。 |
| `ne1` | dim1 元素数，即独立 row 数。 |
| `ne2` | dim2 元素数，即独立 channel 数。 |
| `ne3` | dim3 元素数，即独立 sample / outer batch 数。 |
| `nb0` | dim0 字节步长；当前要求等于 `sizeof(float)`，所以每个归约向量内部连续。 |
| `nb1..nb3` | dim1..dim3 字节步长；当前转换为 F32 元素步长后传入 kernel。 |
| `s1..s3` | 输入元素步长，由 `nb1..nb3 / sizeof(float)` 得到。 |
| `ncols` | `ne0`，每个归约向量的元素数。 |
| `nrows` | `ne1`，grid x 维。 |
| `nchannels` | `ne2`，grid y 维。 |
| `nsamples` | `ne3`，grid z 维。 |
| `total_elements` | `ne0 * ne1 * ne2 * ne3`，输入和输出的总元素数。 |
| `row_elements` | `ne0`，单个 RMS Norm 向量的元素数。 |
| `eps` | F32 标量，要求 `eps >= 0.0f`。 |
| `stream` | CUDA stream，当前来自 `ctx.stream()`。 |

维度语义：

```text
2D: [ne0, ne1]           => ncols=ne0, nrows=ne1, nchannels=1,   nsamples=1
3D: [ne0, ne1, ne2]      => ncols=ne0, nrows=ne1, nchannels=ne2, nsamples=1
4D: [ne0, ne1, ne2, ne3] => ncols=ne0, nrows=ne1, nchannels=ne2, nsamples=ne3
```

输入可以在 dim1..dim3 上是 strided layout，因为 kernel 使用 `s1..s3` 计算每个向量
的起始地址；dim0 必须连续，因为 kernel 在向量内部直接读取 `x[col]`。当前未融合路径
不使用 `src[1]`、`src[2]` 或额外 `op_params` 字节。

当前实现从 `dst->op_params` 的前 `sizeof(float)` 字节读取 `eps`，并断言输入
`src0->type == GGML_TYPE_F32`、输出 `dst->type == GGML_TYPE_F32`。

## 算法

CUDA launch 使用：

```text
grid  = (nrows, nchannels, nsamples)
block = (WARP_SIZE, 1, 1) if ncols < 1024 else (1024, 1, 1)
```

一个 CUDA block 处理一个 `(row, channel, sample)` 对应的向量。该向量的输入和输出
基地址为：

```text
input_base  = input  + sample*s3 + channel*s2 + row*s1
output_base = output + ((sample*nchannels + channel)*nrows + row)*ncols
```

核心处理流程：

```text
sum = 0
for col in columns assigned to this thread:
    v = input_base[col]
    sum += v * v

sum = block_reduce_sum(sum)
scale = rsqrt(sum / ncols + eps)

for col in columns assigned to this thread:
    output_base[col] = input_base[col] * scale
```

实现细节：

- `ncols` 是归约长度，要求为正；当前代码对 `sum / ncols` 不做额外保护。
- `block_reduce_sum` 先做 warp 内归约；1024-thread 路径再用共享内存归约每个 warp
  的部分和。
- 输入 dim1..dim3 的 strided 地址由 `s1..s3` 决定。
- 输出地址按 dense row layout 线性递增，不使用输出张量的 stride 字段。
- in-place 情况下 `output` 可与 `input` alias；当前算法先完成该向量的平方和归约，
  再写回同一向量。

## 输出

输出参数和数据：

| 参数 | 含义 |
| --- | --- |
| `output` | CUDA 可访问的输出基地址，写入归一化后的 F32 数据。 |
| `output_dtype` | `F32`。 |
| `output_dims` | 与输入相同：`[ne0, ne1, ne2, ne3]`。 |
| `output_elements` | `ne0 * ne1 * ne2 * ne3`。 |
| `output_bytes` | `output_elements * sizeof(float)`。 |
| `row_layout` | Dense/contiguous row layout，dim0 最快。 |

每个 `(row, channel, sample)` 输出向量占连续 `ne0` 个 F32 元素。

因此，第 `(row, channel, sample)` 个输出向量位于：

```text
output + ((sample*nchannels + channel)*nrows + row)*ncols
```

其中每个元素满足：

```text
output[col] = input[col] * rsqrt(mean(input[0:ncols]^2) + eps)
```

当前未融合实现不写额外输出参数，也不产生 side data。若未来支持非连续输出 stride，
需要把输出 byte stride / element stride 明确加入协议；当前实现没有这部分输出协议。

## QEMU / qemu_cuda canonical protocol

实验路径由 `GGML_CUDA_RMS_NORM_QEMU_MODE` 控制，默认 `cuda` 时上述原始协议完全不变。
启用 `qemu`、`qemu_cuda` 或 `compare` 后，CUDA preprocess 使用 `s1..s3` 读取原始
strided F32 输入，并生成如下 canonical tensor：

```text
dtype  = BF16 bit pattern
layout = dense [nsamples, nchannels, nrows, ncols]
bytes  = ncols * nrows * nchannels * nsamples * sizeof(uint16_t)
round  = F32-to-BF16 RZ/truncation: bf16_bits = f32_bits >> 16
```

QEMU/RVV 和 qemu_cuda 消费完全相同的 canonical BF16 input，并产生同布局 BF16
output。RMS_NORM 在这里遵循所有后续 QEMU/RVV 算子的通用 canonical input 规则，并非
特殊例外：上游 F32 tensor 直接保留 bit pattern 的高 16 位，不对低 16 位做进位，也
不额外改写 NaN payload。
`eps` 以 F32 标量传输，在算子入口按 RNE 量化为 BF16；`1/ncols` 也按 RNE 量化为
BF16。返回 llama.cpp 下游前，CUDA 在 device 上把 BF16 output 转换回 dense F32
`dst`。

当前 BF16 RMS_NORM 数值模型直接参考
`decode/rms_norm/ybxkernel/eu_rms_norm.cl`，并固定为 NI900 VLEN=512 的 32 个 e16 lane：

1. lane `i` 按 `i, i+32, ...` 读取列，并对每一步执行 BF16 fused multiply-add；
2. `vfredusum` 按 lane 0 到 31 的顺序执行 BF16 加法；
3. sum、BF16 `1/ncols` 和 BF16 `eps` 依次执行 BF16 multiply/add/sqrt/reciprocal；
4. 第二遍按 BF16 multiply 缩放输入并直接输出 BF16。

qemu_cuda 使用 `ggml/src/ggml-cuda/expt/rms-norm-bf16-core.cuh` 复刻相同 lane 映射、
归约顺序和每一步 RNE 舍入。QEMU 与 qemu_cuda 比较原始 `uint16_t`，契约为 bit
mismatch 数量等于 0。llama.cpp 的未融合 `GGML_OP_RMS_NORM` 不包含 weight 乘法，
因此 canonical RPC 不传 weight。

`qemu_cuda` 纯路径只使用 device buffer 和调用方 CUDA stream，不创建 ZMQ socket，也
不执行 D2H/H2D。显式启用 `GGML_CUDA_RMS_NORM_QEMU_TIMING` 时会创建 CUDA event 并
同步当前调用，以记录 preprocess、BF16 operator、BF16-to-F32 和 total 时间。
