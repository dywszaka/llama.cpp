# K/Q/V 张量导出与量化评估规范

## 范围

本文定义 LLAMA_CPP-12 实验的约定：导出选定的 attention 张量，并在离线流程中评估量化误差。该实验必须显式开启；未设置相关开关时，不应改变运行时推理数学行为。

当前实现从 decode graph 中导出选定的已计算 F32 图张量，然后用 NVFP4 参考量化/反量化 roundtrip 做离线评估。运行时导出钩子位于 `src/llama-context.cpp` 的 `llama_context::process_ubatch()` 中；实验逻辑、manifest 解析、指标计算和初始 NVFP4 evaluator 位于 `src/expt/tensor-export-eval.{h,cpp}`。CLI 入口是 `tools/tensor-export-eval/llama-tensor-export-eval`。

## 数据导出

导出默认关闭。仅当 `LLAMA_EXPT_TENSOR_EXPORT_DIR` 设置为非空目录时启用。运行时应尽可能创建该目录，并把全部导出产物写入其中。

`LLAMA_EXPT_TENSOR_EXPORT_KINDS` 可选地过滤张量种类。它是逗号分隔列表，支持以下值：

- `k`
- `q`
- `v`
- `kq`
- `kqv`

当 `LLAMA_EXPT_TENSOR_EXPORT_KINDS` 未设置或为空时，导出器会尝试所有支持的种类：`k,q,v,kq,kqv`。该 kind 开关只过滤已识别的张量名；如果没有设置 `LLAMA_EXPT_TENSOR_EXPORT_DIR`，它不会单独启用导出。

通用 op 导出模式由 `LLAMA_EXPT_TENSOR_EXPORT_OP` 开启，并由
`LLAMA_EXPT_TENSOR_EXPORT_TYPE=decode|prefill` 选择首次 prompt/prefill graph
或 prompt 之后的首次单 token decode graph；position 0 的单 token prompt 也
归类为 prefill。op 名按 `ggml_op_name()` 匹配，忽略大小写并接受可选的
`GGML_OP_` 前缀。该模式会导出所有匹配节点的 `dst`、`dst->src[0]` 和
`dst->src[1]`；不存在的 source 不生成 record。它保留每个 tensor 的原始
dtype 和带 stride 的内存 span，不要求 F32 或连续布局，也不使用 kind
过滤器。

op 模式的 manifest 格式为 `llama_expt_op_tensor_export_v1`，顶层包含
`type`、`op`、`matched_nodes` 和 `records`。每条 record 包含
`node_index`、`op`、`role`、`name`、`dtype`、`ne`、`nb`、`path`、
`byte_size`、`contiguous` 和 `view_offset`。这些 raw 文件用于 op 级调试，
不能直接交给只接受 F32 K/Q/V manifest 的离线 NVFP4 evaluator。

`LLAMA_EXPT_TENSOR_EXPORT_LAYER` 可选地把 op 导出限制到一个从 0 开始的
模型层。匹配依据是 graph tensor 名中的稳定 layer 标记，例如 `norm-0`、
`blk.0.*` 或 `cache_k_l0`。op manifest 顶层用 `layer` 记录实际过滤值；未
设置时为 `-1`，表示导出全部匹配层。

导出器在 backend scheduler 同步后扫描已完成的 graph node。它从稳定的 attention 相关张量名识别 kind，按 graph node 指针去重，并为每个支持的张量写一个 raw 文件。导出文件是无 header 的连续 F32 little-endian 数据。文件大小必须等于 `ggml_nelements(tensor) * sizeof(float)`。

导出目录中的 manifest 文件名为 `manifest.json`，格式如下：

```json
{
  "format": "llama_expt_tensor_export_v1",
  "records": [
    {
      "name": "k-synthetic",
      "kind": "k",
      "dtype": "f32",
      "ne": [16, 1, 1, 1],
      "nb": [4, 64, 64, 64],
      "path": "k0.bin",
      "byte_size": 64
    }
  ]
}
```

每条 record 包含以下字段：

- `name`：原始 ggml 张量名。
- `kind`：归一化后的张量种类，取值为 `k`、`q`、`v`、`kq` 或 `kqv`。
- `dtype`：导出的数据类型。当前约定仅支持 `f32`。
- `ne`：四个 ggml 逻辑维度。
- `nb`：源张量的四个 ggml byte stride。
- `path`：相对 manifest 目录的 raw `.bin` 文件路径。
- `byte_size`：raw 文件字节数。

导出器当前对不支持的张量采取跳过策略，而不是让推理失败。以下情况应跳过：kind 未识别或未被选择、张量不是 `GGML_TYPE_F32`、graph node 指针重复、张量非连续。对于不支持的 dtype 和非连续张量，应打印 warning，便于 run record 解释缺失的导出产物。

`experiments/` 下的实验目录应保存 manifest、raw `.bin` 文件、生成和评估它们的命令或脚本、输入引用、原始工具输出、解析后的指标和简短总结。来自 `llama-server` 验证的运行时导出还应保存请求 payload/data、server response、server logs 和验证结果。长期保留的 tensor-export 说明、脚本和轻量示例元数据放在 `mylab/tensor-export/` 下；大型或临时 raw tensor 文件不应搬入该目录，除非它们是明确需要版本管理的 fixture。

## 模型与运行参数选择

导出器本身不指定模型，也不指定推理参数。导出钩子运行在当前进程的 `llama_context::process_ubatch()` 内，并在 graph compute 之后导出当前进程、当前模型和当前运行参数实际产生的张量。

导出只由环境变量控制：

- `LLAMA_EXPT_TENSOR_EXPORT_DIR`：非空时启用导出，并指定导出目录。
- `LLAMA_EXPT_TENSOR_EXPORT_KINDS`：可选，限制导出的 kind。

模型和运行时参数由启动 `llama-server`、`llama-perplexity` 或其他 llama.cpp 工具的命令指定。基线来源是 `expt-baseline.md`；新实验脚本应从其中的基线命令开始，只替换实验必需的参数或新增实验开关。

当前基线模型是 `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`。共享基线参数包括 CUDA device 0、`--n_gpu_layers 40`、`-t 32`、`--cache-type-k f16`、`--cache-type-v f16`、flash attention 关闭、KQV offload 开启、`--kv-unified`。对于 `llama-server`，基线还包括 `--batch-size 512`、`--ubatch-size 512` 和 `-c 8192`。

下面示例只是在 `llama-server` 基线命令上增加导出环境变量；它不是新的永久基线：

```bash
CUDA_VISIBLE_DEVICES=0 \
LLAMA_EXPT_TENSOR_EXPORT_DIR="${WORKSPACE}/experiments/LLAMA_CPP-12-export" \
LLAMA_EXPT_TENSOR_EXPORT_KINDS="k,q,v,kq,kqv" \
LLAMA_STDOUT_FILE="${WORKSPACE}/gpu.log" \
  "${WORKSPACE}/build_cuda/bin/llama-server" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    --n_gpu_layers 40 \
    --host 127.0.0.1 \
    --batch-size 512 \
    --ubatch-size 512 \
    --port 8080 \
    -t 32 \
    -c 8192 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --kv-unified \
    --log-file "${WORKSPACE}/gpu.log"
```

## 算法实现

基线 evaluator 算法是 `nvfp4_ref`。它通过复用以下函数模拟当前 CPU 参考 NVFP4 activation roundtrip：

- `quantize_row_nvfp4_ref()`
- `dequantize_row_nvfp4()`

这两个函数定义在 `ggml/src/ggml-quants.c` 中，并通过本地实验 evaluator 暴露。复用这些函数可以让离线基线与仓库中的参考 NVFP4 编码和解码行为保持一致，避免在 CLI 中复制一套独立模型。

evaluator 将每条 manifest record 作为连续 F32 加载，量化为临时 `block_nvfp4` 数据，再反量化回 F32，然后把 roundtrip 后的值与导出的参考值比较。当前 CLI 接受 `--global-scale N`，默认值为 `1.0`。该值会原样传给参考 quantizer 和 dequantizer。默认不指定算法时仍输出 `nvfp4_ref` JSON。

K-channel 排序 evaluator 是离线比较算法，不改变运行时推理路径。CLI 支持：

- `--k-channel-sort` 或 `--algorithm nvfp4_k_channel_sort`：仅接受 `kind == "k"` 的 records，把 `ne[0]` 视为 channel 数，用第一行的 `abs(value)` 降序生成 channel order，绝对值相同按 channel index 升序稳定打破平局。
- `--k-channel-mean-sort` 或 `--algorithm nvfp4_k_channel_mean_sort`：仅接受 `kind == "k"` 的 records，把 `ne[0]` 视为 channel 数，剩余维度视为 rows/tokens；先对每个 channel 计算跨全部 rows 的 signed mean，再按 `abs(mean)` 降序生成 channel order，绝对值相同按 channel index 升序稳定打破平局。输出中的 `sort_basis` 为 `abs_mean`，表示排序依据是平均值绝对值而不是 signed mean 本身。

两个 K-channel 排序 evaluator 都会先计算未排序 `nvfp4_ref` roundtrip 指标，再按 channel order 重排每一行、执行 NVFP4 roundtrip、按反向 order 还原布局，最后与原始 F32 reference 比较。输出包含 `baseline_metrics`、`sorted_metrics`、`delta_metrics`、`channel_count`、`row_count`、`sort_basis` 和 `channel_order`。这种离线流程用于判断 channel 分组对 NVFP4 roundtrip 误差的影响；它不保证运行时 PPL 可直接运行，因为运行时 K-cache channel 重排必须同时对 KQ 中的 Q 应用同一 per-layer channel order 才能保持 dot-product 语义。

在 NVFP4 运行时路径中，`input_scale` 通常表示 `1 / global_scale`。后续如果做使用绑定张量 scale 的运行时派生评估，应明确导出值是 `input_scale` 还是 `global_scale`，并在 evaluator 边界完成转换。当前 manifest 不存储每条 record 的 scale，因此 `--global-scale` 是 run-level 评估参数，应随命令输出一起记录。

后续量化算法应通过一个小的算法选择边界接入离线 evaluator，例如增加显式 CLI algorithm 选项，并增加一个接收已加载 F32 向量和算法参数的 evaluator 函数。新算法不得改变运行时默认行为，不得隐式启用导出；实质性实现应放在 `src/expt/` 或其他实验归属目录下，而不是把算法逻辑加入 upstream-tracked runtime 文件。运行时集成应保持为导出、dispatch 或开关 plumbing 的窄钩子。

## 评估方法

evaluator 读取 `manifest.json`，验证每条 record，从 manifest 目录相对路径加载 raw F32 文件，执行选定量化 roundtrip，并输出 JSON。`nvfp4_ref` 输出包括：

- `algorithm`
- `global_scale`
- `records`
- `aggregate_by_kind`

每条 record 的输出保留 manifest 字段，并增加 `metrics`。`aggregate_by_kind` 会合并相同 `kind` 的全部 records。

K-channel 排序算法的输出仍保留 `algorithm`、`global_scale`、`records` 和 `aggregate_by_kind`，另外顶层和 record 层都有 `sort_basis`。每条 record 使用 `baseline_metrics`、`sorted_metrics` 和 `delta_metrics` 代替单一 `metrics` 字段；`delta_metrics` 定义为 `sorted - baseline`。

兼容性检查会拒绝不支持或不一致的数据：

- `dtype` 必须是 `f32`。
- `ne` 和 `nb` 必须各包含四个元素。
- `ne` 中所有 extent 必须为正。
- `ne[0]` 必须能被 NVFP4 block size 整除。
- `byte_size` 必须等于由 shape 推导出的 F32 字节数。
- raw 文件大小必须匹配 `byte_size`。
- 指标输入必须非空，且元素数量相等。

对每个张量，令 `reference[i]` 为导出的 F32 值，`actual[i]` 为 roundtrip 后的值。若元素数为 `n`：

```text
MAE  = sum(abs(actual[i] - reference[i])) / n
MSE  = sum((actual[i] - reference[i])^2) / n
RMSE = sqrt(MSE)
```

按 kind 聚合的指标按元素数加权，而不是按张量数加权：

```text
kind_MAE  = sum(record_MAE * record_n) / sum(record_n)
kind_MSE  = sum(record_MSE * record_n) / sum(record_n)
kind_RMSE = sqrt(kind_MSE)
```

合成 fixture 验证应覆盖指标公式、开关检测、manifest 加载、按 kind 聚合，以及对不兼容 dtype、byte-size mismatch 和 NVFP4 不兼容 shape 的拒绝。当前实现的 focused test 是 `tests/test-expt-tensor-export-eval.cpp`。轻量示例记录目录 `mylab/tensor-export/examples/LLAMA_CPP-12-tensor-export-eval-sample/` 是 fixture 验证记录，不是模型运行时导出，也不是性能测量；其中保留命令、manifest、输出和总结，不保留 raw `.bin` 样本。

实验总结应区分观察到的指标和解释。有效的 run record 应写明 code revision、导出开关、选定 kinds、评估命令、global scale、输入 workload 或 fixture 来源、原始输出路径、解析后指标路径和已知混杂因素。直接 A/B 对比必须按照 `expt-baseline.md` 和 `docs/development/experiment-records.md` 固定无关基线参数。

## 限制与已知风险

该实验只导出连续 F32 graph 张量。它不导出非 F32 张量、非连续 view、张量 sidecar scales、per-layer metadata 或完整运行时上下文。导出张量会增加同步、host copy、磁盘 I/O 和日志，因此除非导出开销就是实验主题，否则导出运行不应被当作性能测量。

kind 检测基于名称。新的 graph 张量名可能需要显式映射后才会出现在导出产物中。反过来，当 active graph 没有创建匹配的 F32 连续张量时，即使选择了某些 kinds，也可能产生零条 records。

当前 NVFP4 evaluator 是离线 roundtrip 基线。它不模拟所有 CUDA native matmul 细节、cuBLASLt 行为、K-cache sidecar compensation 或 generic dequantization 路径。对于 correctness-sensitive 的运行时调试，应把该 evaluator 作为一个信号，并结合运行时配置证据、代码路径确认、focused CUDA 或 server 验证 artifact 一起判断。

op 导出在 graph compute 完成后读取 raw tensor span，可能显著增加同步、
显存到主机拷贝和磁盘占用。特别是 `MUL_MAT` 一类 op 的 `src0` 可能是大
权重 tensor；一次导出可能接近模型大小。该模式仍属于诊断运行，不应用于
性能结论。
