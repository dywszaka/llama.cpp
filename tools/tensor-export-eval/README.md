# Tensor Export Eval

`llama-tensor-export-eval` 用于离线读取 tensor export 产物，并对导出的张量做复算或误差统计。

当前支持的离线算法包括：

- `nvfp4_ref`
- `attention_replay`

本 README 重点说明 layer 0 attention softmax 的导出和离线评测。

## 构建

在已有构建目录中编译：

```bash
cmake --build build-default --target llama-tensor-export-eval -j 4
```

生成的可执行文件为：

```bash
build-default/bin/llama-tensor-export-eval
```

## 运行时导出

运行时导出由既有环境变量控制：

- `LLAMA_EXPT_TENSOR_EXPORT_DIR`
  - 导出目录，运行结束后会写入 `manifest.json` 和对应的 raw `.bin`
- `LLAMA_EXPT_TENSOR_EXPORT_KINDS`
  - 逗号分隔的导出 kind 列表

对 layer 0 attention softmax 离线 replay，建议至少导出：

```text
k,q,kq,kq_softmax,kq_mask,k_attn,q_attn
```

在 `non-flash attention + --cache-type-k f16` 条件下，manifest 中会包含这些关键记录：

- `Kcur-0`
- `Qcur-0`
- `kq-0`
- `kq-softmax-0`
- `kq-mask-0`
- `k-attn-0`
- `q-attn-0`

其中 `kq-softmax-0` 的 `meta` 会记录：

- `src_k`
- `src_q`
- `src_kq`
- `src_mask`
- `kq_scale`
- `max_bias`

## 导出示例

下面示例会运行 `llama-cli`，并把 layer 0 attention 相关张量导出到指定目录：

```bash
EXPORT_DIR=/tmp/layer0-export
MODEL=/path/to/model.gguf
PROMPT='Hello'

mkdir -p "$EXPORT_DIR"

LLAMA_EXPT_TENSOR_EXPORT_DIR="$EXPORT_DIR" \
LLAMA_EXPT_TENSOR_EXPORT_KINDS="k,q,kq,kq_softmax,kq_mask,k_attn,q_attn" \
./build-default/bin/llama-cli \
  -m "$MODEL" \
  -p "$PROMPT" \
  -n 1 \
  -c 512 \
  --batch-size 32 \
  --ubatch-size 32 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --kv-unified \
  -t 4
```

导出完成后，检查：

```bash
ls "$EXPORT_DIR"
cat "$EXPORT_DIR/manifest.json"
```

## 离线评测

离线 replay 命令：

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/export/manifest.json \
  --algorithm attention_replay
```

输出为 JSON，主要字段包括：

- `records[].k_record`
- `records[].q_record`
- `records[].kq_record`
- `records[].softmax_record`
- `records[].kq_metrics`
- `records[].softmax_metrics`
- `records[].max_abs_err_kq`
- `records[].max_abs_err_softmax`

## attention_replay 如何复算

`attention_replay` 的实现位于：

- [tensor-export-eval.cpp](/home/allen/host_workspace/develop/llama.cpp/src/expt/tensor-export-eval.cpp:874)
- [tensor-export-eval.cpp](/home/allen/host_workspace/develop/llama.cpp/tools/tensor-export-eval/tensor-export-eval.cpp:1)

离线评测流程如下：

1. 读取 `manifest.json`
2. 查找 kind 为 `kq_softmax` 且名称为 `kq-softmax-0` 的记录
3. 从该记录的 `meta` 中取出 `src_k`、`src_q`、`src_kq`、`src_mask`、`kq_scale`、`max_bias`
4. 读取对应 raw F32 导出文件
5. 按运行时相同的 attention layout 做 `q reshape_4d + permute`、`k permute`
6. 把导出的 `k-attn-0` 先做一次 `fp32 -> f16` 回放，以匹配 `--cache-type-k f16`
7. 用 ggml CPU backend 重算：
   - `ggml_mul_mat`
   - `ggml_soft_max_ext`
8. 将离线结果与运行时导出的 `kq-0`、`kq-softmax-0` 比较，输出误差

## 现成实验脚本

仓库中已有一份可复用的实验脚本：

- [run.sh](/home/allen/host_workspace/develop/llama.cpp/experiments/20260708T080134Z-layer0-attn-softmax-export/run.sh)

它会：

1. 调 `llama-cli` 导出 layer 0 attention 张量
2. 调 `llama-tensor-export-eval --algorithm attention_replay`
3. 生成：
   - `export/manifest.json`
   - `logs/llama-cli.stdout.log`
   - `logs/llama-cli.stderr.log`
   - `logs/attention-replay.json`

直接运行：

```bash
./experiments/20260708T080134Z-layer0-attn-softmax-export/run.sh
```

## 当前参考结果

参考实验目录：

- [summary.md](/home/allen/host_workspace/develop/llama.cpp/experiments/20260708T080134Z-layer0-attn-softmax-export/summary.md)
- [attention-replay.json](/home/allen/host_workspace/develop/llama.cpp/experiments/20260708T080134Z-layer0-attn-softmax-export/logs/attention-replay.json)

当前通过结果为：

```text
max_abs_err_kq=0.0
max_abs_err_softmax=0.0
```

## 约束

目前这套说明和验证结论只覆盖：

- layer 0
- non-flash attention
- `--cache-type-k f16`

不覆盖：

- flash-attention 路径
- 其他 `cache-type-k`
- 全层泛化保证
