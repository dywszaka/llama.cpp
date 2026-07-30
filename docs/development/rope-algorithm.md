# CPU 前向 RoPE 算法与参数说明

本文档以当前 ggml CPU 前向实现为准，说明 `GGML_OP_ROPE` 的数学公式、
张量布局、全部算子参数、固定常量、Normal/NeoX/M-RoPE/Vision 模式差异、
YaRN 缩放以及 CPU 执行方式。

主要代码入口：

- 算子创建与参数编码：[`ggml/src/ggml.c`](../../ggml/src/ggml.c)
- CPU 算法实现：[`ggml/src/ggml-cpu/ops.cpp`](../../ggml/src/ggml-cpu/ops.cpp)
- CPU 算子分发与工作区规划：[`ggml/src/ggml-cpu/ggml-cpu.c`](../../ggml/src/ggml-cpu/ggml-cpu.c)
- 公共 API 与模式常量：[`ggml/include/ggml.h`](../../ggml/include/ggml.h)
- llama.cpp 运行参数解析：[`src/llama-context.cpp`](../../src/llama-context.cpp)

## 1. 算法目标

RoPE（Rotary Position Embedding）把输入向量中的两个标量组成一个二维向量，
再根据 token 的位置和该维度对应的频率进行旋转。对二维输入

\[
\mathbf{x}=
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
\]

旋转后的结果为

\[
\begin{bmatrix}
y_0 \\
y_1
\end{bmatrix}
=
\begin{bmatrix}
c & -s \\
s &  c
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix},
\]

即

\[
y_0=x_0c-x_1s,
\qquad
y_1=x_0s+x_1c.
\]

标准 RoPE 中，\(c=\cos\theta\)、\(s=\sin\theta\)。当前实现还允许通过
`attn_factor` 和 YaRN 对二者施加相同的幅度缩放。

对位置分别为 \(p\) 和 \(q\) 的 Query、Key，有

\[
(R(p)Q)^T(R(q)K)=Q^TR(q-p)K,
\]

因此旋转后的注意力内积能够直接表达相对位置 \(q-p\)。

## 2. 输入、输出与张量维度

RoPE 节点使用三个输入：

| 名称 | ggml 源张量 | 类型 | 形状或长度 | 含义 |
| --- | --- | --- | --- | --- |
| `a` | `src0` | `F32` 或 `F16` | `[ne0, ne1, ne2, ne3]` | 要旋转的 Q、K 或其他特征 |
| `b` | `src1` | `I32` | 普通模式为 `ne2`；M-RoPE/Vision 为 `4 * ne2` | position ID |
| `c` | `src2` | 可选 `F32` | API 要求至少 `n_dims / 2` | 每个旋转对的额外频率因子 |

维度语义通常为：

| 维度 | 常见含义 |
| --- | --- |
| `ne0` | 每个 attention head 的通道数，即 `head_dim` |
| `ne1` | attention head 数量 |
| `ne2` | token 数量或 sequence length |
| `ne3` | batch 维度 |

输出张量与 `a` 具有相同的形状和数据类型。非 in-place API 创建同形状输出，
in-place API 返回 `a` 的 view。

位置只按照 `i2`（token 维）索引；同一个 `i2` 的位置会在所有 `i1` head 和
所有 `i3` batch 上复用。当前位置张量不带独立的 batch 维。

## 3. 全部算子参数

### 3.1 公共 API 参数

| 参数 | 类型 | 作用 | 当前约束或默认行为 |
| --- | --- | --- | --- |
| `a` | tensor | 被旋转的输入 | CPU 仅实现 `F32`、`F16` |
| `b` | tensor | position ID | 必须是 `I32` 向量 |
| `c` | tensor/null | `freq_factors` | 为 null 时每个因子固定为 `1.0` |
| `n_dims` | int | RoPE 参数中的旋转维度 | 必须为偶数，且 `n_dims <= ne0`；Vision 还要求 `n_dims == ne0 / 2` |
| `sections[4]` | int[4] | M-RoPE 的四个位置分区长度 | 固定为 4 项；单位是“旋转对”而不是单个标量 |
| `mode` | int | 选择维度配对和多位置模式 | 具体数值见下表 |
| `n_ctx_orig` | int | YaRN 原始上下文长度 | 用于计算 YaRN correction dimensions |
| `freq_base` | float | RoPE 基频 \(B\) | 模型缺省值为 `10000.0` |
| `freq_scale` | float | 插值角度缩放 \(S_f\) | 无缩放时为 `1.0` |
| `ext_factor` | float | YaRN 外推混合强度 \(E\) | 非 YaRN 时为 `0.0`，完整 YaRN 混合通常为 `1.0` |
| `attn_factor` | float | cos/sin 的初始幅度因子 \(M\) | llama context 默认 `1.0`，之后还会乘模型中的 attention factor |
| `beta_fast` | float | YaRN 高频 correction 阈值 | llama context 默认 `32.0` |
| `beta_slow` | float | YaRN 低频 correction 阈值 | llama context 默认 `1.0` |
| `inplace` | bool | 输出是否为输入 view | 仅由所调用的 API 变体决定，不编码为 op 参数 |

模式常量固定为：

| 模式 | 数值 | CPU 判断方式 | 配对方式 |
| --- | ---: | --- | --- |
| Normal | `0` | 其他模式均不匹配 | 相邻维度 `(2j, 2j+1)` |
| GPT-NeoX | `2` | `mode & 2` | 两个半区 `(j, j+n_dims/2)` |
| M-RoPE | `8` | `mode & 8` | 两个半区，并从四路 position ID 中选一路 |
| Vision | `24` | `mode == 24` | M-RoPE 特例；完整 head 的前后两半配对 |
| None | `-1` | llama 模型层含义，不应创建 RoPE op | 不执行 RoPE |

`mode` 的 bit 0 历史上表示跳过 `n_past`，当前已不支持；创建算子时固定要求
`(mode & 1) == 0`。CPU 数值计算当前只显式检查 bit 1、bit 3 和完整值 `24`，
其他未识别且 bit 0 为 0 的值会落入 Normal 路径。

### 3.2 `op_params` 的精确布局

RoPE 使用 15 个 `int32_t` 大小的参数槽。浮点数通过 `memcpy` 按位写入对应槽：

| 槽位 | 内容 | 类型 | 固定值或来源 |
| ---: | --- | --- | --- |
| 0 | `n_past`，已停用 | int32 | 固定为 `0` |
| 1 | `n_dims` | int32 | API 参数 |
| 2 | `mode` | int32 | API 参数 |
| 3 | `n_ctx`，已停用 | int32 | 固定为 `0` |
| 4 | `n_ctx_orig` | int32 | API 参数 |
| 5 | `freq_base` | float bits | API 参数 |
| 6 | `freq_scale` | float bits | API 参数 |
| 7 | `ext_factor` | float bits | API 参数 |
| 8 | `attn_factor` | float bits | API 参数 |
| 9 | `beta_fast` | float bits | API 参数 |
| 10 | `beta_slow` | float bits | API 参数 |
| 11..14 | `sections[0..3]` | int32 | 非 M-RoPE 固定清零 |

### 3.3 简化 API `ggml_rope()` 的固定参数

`ggml_rope(ctx, a, b, n_dims, mode)` 和对应的 in-place 版本固定使用：

```text
c            = null
n_ctx_orig   = 0
freq_base    = 10000.0
freq_scale   = 1.0
ext_factor   = 0.0
attn_factor  = 1.0
beta_fast    = 0.0
beta_slow    = 0.0
sections     = {0, 0, 0, 0}
```

因为 `ext_factor = 0.0`，此入口不会启用 YaRN 外推混合。

### 3.4 llama context 的默认值与解析规则

`llama_context_default_params()` 提供的是“用户未指定”值，其中部分值还会被模型
元数据替换：

| llama context 参数 | API 默认值 | 最终解析规则 |
| --- | ---: | --- |
| `rope_freq_base` | `0.0` | `0.0` 表示使用模型值；模型未提供时模型缺省为 `10000.0` |
| `rope_freq_scale` | `0.0` | `0.0` 表示使用模型值；模型未提供 scaling factor 时最终为 `1.0` |
| `yarn_ext_factor` | `-1.0` | 负数表示自动选择：YaRN scaling 为 `1.0`，否则为 `0.0` |
| `yarn_attn_factor` | `1.0` | 再乘模型的 `rope_attn_factor`；模型缺省也是 `1.0` |
| `yarn_beta_fast` | `32.0` | 直接传给 RoPE op |
| `yarn_beta_slow` | `1.0` | 直接传给 RoPE op |
| `yarn_orig_ctx` | `0` | 依次回退到模型 `n_ctx_orig_yarn`、模型训练上下文长度 |

如果最终 scaling type 是 `NONE`，`freq_scale` 会被强制设为 `1.0`。

## 4. 基础频率和角度

定义：

- \(B=\texttt{freq_base}\)
- \(d=\texttt{n_dims}\)
- \(j=i_0/2\) 为旋转对编号
- \(p\) 为该 token 选中的 position ID

CPU 首先计算固定的相邻频率比：

\[
\texttt{theta\_scale}=B^{-2/d}.
\]

Normal/NeoX/普通 M-RoPE 中第 \(j\) 对的基础角度为：

\[
\theta_{\text{base},j}
=p\cdot \texttt{theta\_scale}^{j}
=p\cdot B^{-2j/d}.
\]

实现没有在循环中调用 `powf`；它从 `theta = p` 开始，每处理一个旋转对执行：

```text
theta *= theta_scale
```

如果提供了 `freq_factors[j]`，则先执行：

\[
\theta_{\text{extrap},j}
=\frac{\theta_{\text{base},j}}{\texttt{freq\_factors}[j]}.
\]

未提供 `c` 时固定有：

\[
\texttt{freq\_factors}[j]=1.0.
\]

## 5. YaRN 角度和幅度缩放

### 5.1 Correction dimensions

当前实现通过下式把 `beta_fast` 和 `beta_slow` 转换为 correction dimension：

\[
\operatorname{corr\_dim}(\beta)
=
\frac{d\cdot
\ln\left(\frac{n_{\text{ctx-orig}}}{\beta\cdot 2\pi}\right)}
{2\ln B}.
\]

其中固定使用 \(2\pi\)。随后计算：

\[
\texttt{corr\_low}
=\max\left(0,\left\lfloor\operatorname{corr\_dim}(\texttt{beta\_fast})\right\rfloor\right),
\]

\[
\texttt{corr\_high}
=\min\left(d-1,\left\lceil\operatorname{corr\_dim}(\texttt{beta\_slow})\right\rceil\right).
\]

`beta_fast`、`beta_slow` 并不是直接的维度编号，而是经过上述公式转换后才得到
correction dimensions。

### 5.2 Ramp

对旋转对 \(j=i_0/2\)，CPU 计算：

\[
y_j=
\frac{j-\texttt{corr\_low}}
{\max(0.001,\texttt{corr\_high}-\texttt{corr\_low})},
\]

\[
r_j=1-\operatorname{clamp}(y_j,0,1).
\]

其中分母保护值固定为 `0.001`。最终混合权重为：

\[
m_j=r_j\cdot \texttt{ext\_factor}.
\]

代码不额外把 `ext_factor` clamp 到 `[0, 1]`；通常调用方传 `0.0` 或 `1.0`。

### 5.3 插值与外推混合

首先计算插值角度：

\[
\theta_{\text{interp},j}
=\texttt{freq\_scale}\cdot\theta_{\text{extrap},j}.
\]

当 `ext_factor == 0.0` 时：

\[
\theta_j=\theta_{\text{interp},j}.
\]

当 `ext_factor != 0.0` 时：

\[
\theta_j
=(1-m_j)\theta_{\text{interp},j}
+m_j\theta_{\text{extrap},j}.
\]

### 5.4 幅度缩放

幅度初始值为：

\[
M=\texttt{attn\_factor}.
\]

只有当 `ext_factor != 0.0` 时，CPU 才应用固定系数 `0.1` 的 YaRN 幅度修正：

\[
M
\leftarrow
M\left(1+0.1\ln\frac{1}{\texttt{freq\_scale}}\right).
\]

最终 cache 中保存：

\[
c_j=M\cos\theta_j,
\qquad
s_j=M\sin\theta_j.
\]

CPU 前向实现传入的 `sin_sign` 固定为 `+1.0`。

为避免 `logf` 和 correction dimension 公式产生无效值，实际启用 YaRN 时应保证
`freq_base > 0`、`freq_base != 1`、`freq_scale > 0`、`n_ctx_orig > 0`、
`beta_fast > 0`、`beta_slow > 0`。当前 CPU 算子本身没有逐项验证这些数值条件。

## 6. 四种维度布局

### 6.1 Normal RoPE：`mode = 0`

对 \(0\le j<d/2\)，相邻标量组成旋转对：

\[
x_0=x[2j],\qquad x_1=x[2j+1].
\]

输出为：

\[
y[2j]=x[2j]c_j-x[2j+1]s_j,
\]

\[
y[2j+1]=x[2j]s_j+x[2j+1]c_j.
\]

索引范围 `[n_dims, ne0)` 原样复制。

### 6.2 GPT-NeoX：`mode & 2 != 0`

前 `n_dims` 个标量被分成两个半区。对 \(0\le j<d/2\)：

\[
x_0=x[j],\qquad x_1=x[j+d/2].
\]

输出为：

\[
y[j]=x[j]c_j-x[j+d/2]s_j,
\]

\[
y[j+d/2]=x[j]s_j+x[j+d/2]c_j.
\]

索引范围 `[n_dims, ne0)` 原样复制。

### 6.3 M-RoPE：`mode & 8 != 0` 且 `mode != 24`

M-RoPE 使用与 NeoX 相同的两个半区配对方式，但每个旋转对从四路 position ID
中选择一路。位置张量的固定布局是：

```text
b[0 * ne2 + i2] = p_t[i2]
b[1 * ne2 + i2] = p_h[i2]
b[2 * ne2 + i2] = p_w[i2]
b[3 * ne2 + i2] = p_e[i2]
```

四路名称来自 CPU 实现：

- `t`：第一位置流，通常为 text/time；
- `h`：第二位置流，通常为 height；
- `w`：第三位置流，通常为 width；
- `e`：第四位置流，供额外的 vision encoder 坐标使用。

定义：

\[
S=s_0+s_1+s_2+s_3,
\qquad
u_j=j\bmod S,
\]

其中 \(s_k=\texttt{sections}[k]\)。按 `u_j` 所在区间选择位置流：

| `u_j` 范围 | position ID |
| --- | --- |
| `[0, s0)` | `p_t` |
| `[s0, s0+s1)` | `p_h` |
| `[s0+s1, s0+s1+s2)` | `p_w` |
| 其余 | `p_e` |

普通 M-RoPE 的四路角度都按照全局旋转对编号 \(j\) 前进。因此选择位置 \(p_k\)
后，基础角度等价于：

\[
\theta_{\text{base},j}=p_kB^{-2j/d}.
\]

`sections` 的单位是旋转对。CPU cache 初始化要求：

```text
sections[0] + sections[1] + sections[2] + sections[3] <= ne0
```

并要求前三项中至少一项大于 0。调用方还必须保证 section 总和大于 0，避免
`j % S` 中的除零；当前代码没有单独断言 `S > 0`，但前三项检查实际保证了这一点。

索引范围 `[n_dims, ne0)` 原样复制。

### 6.4 Vision RoPE：`mode = 24`

Vision 是 M-RoPE 特例，固定要求：

\[
\texttt{n\_dims}=ne0/2.
\]

这里 `n_dims` 的含义与其他模式不同：它既是两个 half 之间的元素距离，也等于
half-head 的大小；Vision 最终会旋转完整的 `ne0` 个标量，而不是只旋转前
`n_dims` 个标量。

对 \(0\le j<ne0/2\)：

\[
x_0=x[j],\qquad x_1=x[j+n_{dims}],
\]

\[
y[j]=x[j]c_j-x[j+n_{dims}]s_j,
\]

\[
y[j+n_{dims}]=x[j]s_j+x[j+n_{dims}]c_j.
\]

Vision 仍使用四路 position ID 和 `sections[4]`，但启用 independent sections。
每次 `u_j` 进入某个 section 的起点时，该位置流的角度会重置为对应 position ID，
因此 section 内的频率指数从 0 重新开始。若 \(k\) 是当前 section 内的局部旋转对
编号，则基础角度为：

\[
\theta_{\text{base}}=p_{section}B^{-2k/d}.
\]

当前 `tools/mtmd/clip.cpp` 中的 Vision M-RoPE 调用固定使用：

```text
n_dims       = d_head / 2
sections     = {d_head/4, d_head/4, d_head/4, d_head/4}
mode         = 24
n_ctx_orig   = 32768
freq_base    = 10000
freq_scale   = 1
ext_factor   = 0
attn_factor  = 1
beta_fast    = 32
beta_slow    = 1
```

这是一处具体调用的固定配置，不是 `ggml_rope_multi()` API 对所有 Vision 模型的
强制默认值。

## 7. 前向旋转

前向对每个二维分组应用：

\[
R(\theta)=
\begin{bmatrix}
c & -s \\
s & c
\end{bmatrix}.
\]

其中 \(c=M\cos\theta\)、\(s=M\sin\theta\)。当 `attn_factor = 1.0` 且未启用
YaRN 幅度修正时，\(M=1\)，矩阵就是标准二维旋转矩阵；否则它是带统一幅度
缩放的旋转变换。

## 8. CPU 实现流程

对 F32 和 F16，CPU 采用相同算法框架：

1. 从 `op_params` 解码全部参数。
2. 验证 `n_dims`、模式和可选 frequency factors。
3. 计算 `theta_scale` 和 YaRN correction dimensions。
4. 将输出的所有 row，即 `ne1 * ne2 * ne3`，按连续区间分配给 CPU worker。
5. 每个 worker 为当前 token 建立一个长度为 `ne0` 的 F32 cos/sin cache。
6. 同一 token 下被该 worker 处理的各个 attention head 复用该 cache。
7. 根据 mode 选择维度配对方式并执行二维旋转。
8. 非 Vision 模式把 `[n_dims, ne0)` 原样复制。

每个 worker 的 cache 地址按下面方式隔离：

```text
cache = wdata + (ne0 + CACHE_LINE_SIZE_F32) * worker_index
```

其中额外 cache-line 间隔用于减少不同 worker 工作区之间的 false sharing；它不改变
RoPE 数学结果。

F16 路径的数值过程固定为：

```text
F16 input -> F32 x0/x1 -> F32 rotation -> F16 output
```

cos/sin cache 始终为 F32。F32 路径直接执行 F32 读写。

## 9. 等价伪代码

下面的伪代码描述 Normal/NeoX/普通 M-RoPE 的公共角度计算：

```text
theta_scale = pow(freq_base, -2 / n_dims)
corr_low, corr_high = yarn_corr_dims(...)

for each token i2:
    choose one or four position IDs

    for j in [0, ne0/2):
        p = select_position(j, mode, sections)
        theta_base = position_frequency(p, j, mode)
        ff = freq_factors ? freq_factors[j] : 1
        theta_extrap = theta_base / ff
        theta_interp = freq_scale * theta_extrap

        if ext_factor != 0:
            ramp = 1 - clamp((j-corr_low) / max(0.001, corr_high-corr_low), 0, 1)
            mix = ramp * ext_factor
            theta = theta_interp * (1-mix) + theta_extrap * mix
            mscale = attn_factor * (1 + 0.1 * log(1/freq_scale))
        else:
            theta = theta_interp
            mscale = attn_factor

        cos_cache[j] = cos(theta) * mscale
        sin_cache[j] = sin(theta) * mscale

    for each assigned head:
        for each active rotation pair j:
            x0, x1 = load_pair(mode, j)
            y0 = x0*cos_cache[j] - x1*sin_cache[j]
            y1 = x0*sin_cache[j] + x1*cos_cache[j]
            store_pair(mode, j, y0, y1)
```

## 10. 当前实现注意事项

- `n_dims` 必须是偶数；Vision 还要求 `ne0` 可被 2 整除并满足
  `n_dims == ne0 / 2`。
- 非 M-RoPE 的 position tensor 长度固定为 `ne2`；M-RoPE 和 Vision 固定要求
  `4 * ne2`，即使具体模型只关心其中部分位置流。
- API 对 `freq_factors` 的公开检查是长度至少为 `n_dims / 2`。但当前 CPU cache
  初始化循环会一直构建到 `ne0`，并在 `c != null` 时访问到
  `freq_factors[ne0/2 - 1]`。因此在 `n_dims < ne0` 时，CPU 调用方应提供至少
  `ne0 / 2` 个元素，除非后续实现缩小 cache 初始化范围。
- `freq_factors` 在公式中作为除数；调用方应避免传入 0。
- `sections` 的数值是旋转对数量，不是字节数，也不是单个标量通道数。
- Normal、NeoX 和普通 M-RoPE 只改变前 `n_dims` 个标量；Vision 会改变完整
  `ne0` 个标量。
- 当前 CPU 实现为 F32、F16 各维护一份基本相同的循环，代码中已有去重 TODO。

## 11. 关键固定值汇总

| 固定项 | 数值 |
| --- | ---: |
| 每个旋转组的标量数 | `2` |
| M-RoPE position/section 路数 | `4` |
| Normal mode | `0` |
| NeoX mode bit | `2` |
| M-RoPE mode bit | `8` |
| Vision mode exact value | `24` |
| 已停用 `n_past` op 参数 | `0` |
| 已停用 `n_ctx` op 参数 | `0` |
| YaRN ramp 分母最小值 | `0.001` |
| YaRN 幅度修正系数 | `0.1` |
| correction dimension 周期常量 | `2*pi` |
| 无 frequency factors 时的因子 | `1.0` |
| 前向 sin 符号 | `+1.0` |
| 模型缺省 `freq_base` | `10000.0` |
| 无频率缩放时 `freq_scale` | `1.0` |
| 非 YaRN `ext_factor` | `0.0` |
| llama context 默认 `attn_factor` | `1.0` |
| llama context 默认 `beta_fast` | `32.0` |
| llama context 默认 `beta_slow` | `1.0` |
