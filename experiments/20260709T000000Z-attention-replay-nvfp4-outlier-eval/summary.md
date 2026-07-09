# NVFP4 outlier attention replay eval

- 输入 manifest：`experiments/20260708T080134Z-layer0-attn-softmax-export/export/manifest.json`。
- 复现实验脚本：`run.sh`。
- baseline replay 输出：`logs/attention-replay.json`。
- NVFP4 outlier replay 输出：`logs/attention-replay-nvfp4-outlier.json`。
- KLD 定义：以导出的 `kq-softmax-0` 为 reference distribution，reference 和 actual 概率在取 log 前都 clamp 到 `epsilon=1e-12`。

## 结果

- 代码版本：本任务提交前工作区构建，`build-default/bin/llama-tensor-export-eval`。
- baseline `attention_replay`：`max_abs_err_kq=0.0`，`max_abs_err_softmax=0.0`，确认现有 replay 行为未回归。
- 新算法 `attention_replay_nvfp4_outlier`：K 量化模式为 `nvfp4_outlier_threshold_layer0`，Q 量化模式为 `nvfp4_dynamic_row_amax`。
- layer 0 K threshold：`256.0`；K `global_scale=5.25`；本次导出数据中 `k_outlier_count=0`。
- softmax 对比：`softmax_mse=0.002568306835889052`，`softmax_kld=23.936649096022784`。
- 结论：新 evaluator 可复用既有 layer 0 attention replay 的 reshape/permute、mask 和 softmax 路径，在 K/Q 进入 `ggml_mul_mat` 前加入 NVFP4 roundtrip，并输出可复查的 softmax MSE/KLD。
