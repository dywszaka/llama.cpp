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

evaluator 将每条 manifest record 作为连续 F32 加载，量化为临时 `block_nvfp4` 数据，再反量化回 F32，然后把 roundtrip 后的值与导出的参考值比较。当前 CLI 接受 `--global-scale N`，默认值为 `1.0`。该值会原样传给参考 quantizer 和 dequantizer。

在 NVFP4 运行时路径中，`input_scale` 通常表示 `1 / global_scale`。后续如果做使用绑定张量 scale 的运行时派生评估，应明确导出值是 `input_scale` 还是 `global_scale`，并在 evaluator 边界完成转换。当前 manifest 不存储每条 record 的 scale，因此 `--global-scale` 是 run-level 评估参数，应随命令输出一起记录。

后续量化算法应通过一个小的算法选择边界接入离线 evaluator，例如增加显式 CLI algorithm 选项，并增加一个接收已加载 F32 向量和算法参数的 evaluator 函数。新算法不得改变运行时默认行为，不得隐式启用导出；实质性实现应放在 `src/expt/` 或其他实验归属目录下，而不是把算法逻辑加入 upstream-tracked runtime 文件。运行时集成应保持为导出、dispatch 或开关 plumbing 的窄钩子。

## 评估方法

evaluator 读取 `manifest.json`，验证每条 record，从 manifest 目录相对路径加载 raw F32 文件，执行选定量化 roundtrip，并输出 JSON。当前输出包括：

- `algorithm`
- `global_scale`
- `records`
- `aggregate_by_kind`

每条 record 的输出保留 manifest 字段，并增加 `metrics`。`aggregate_by_kind` 会合并相同 `kind` 的全部 records。

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
