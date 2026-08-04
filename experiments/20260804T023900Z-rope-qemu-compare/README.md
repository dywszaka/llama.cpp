# ROPE QEMU validation, 2026-08-04

This validation used `/home/lerong.chen/qwen3-8b-q4_0.gguf`, context size 8192,
prompt `hello`, one generated token, all layers on an RTX 5090, and the static
table from `/home/lerong.chen/0729-rope-node4`.

This is a focused implementation validation, not a direct comparison with the
NVFP4 server baseline in `expt-baseline.md`: the locally available Q4_0 model
and llama-cli workload were chosen to reproduce the captured RoPE parameters.

Observed results:

- 72 RoPE records (Q and K across 36 layers);
- QEMU/RVV versus qemu_cuda BF16 bit mismatch sum: `0`;
- no mismatch artifact was created;
- maximum original-CUDA versus QEMU RMSE: `0.0527223`;
- maximum original-CUDA versus QEMU absolute error: `0.96814`;
- compare mode kept the original CUDA output downstream.
- a separate pure `qemu` llama run completed 72 successful RPC requests and
  used the returned RVV values downstream.

The larger original-CUDA difference is expected from canonical BF16 input
truncation on activation values. The validation checks hardware-model
consistency with raw BF16 comparison separately from this lossy boundary.

Raw per-node metrics are in `compare.jsonl`. QEMU service logs were captured at
`/home/lerong.chen/qemu/rope-fp32-llama-logs` during the run.
