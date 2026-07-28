# CUDA RoPE cos/sin export, context 8192

- Model parameters: Qwen3-8B, GPT-NeoX RoPE, `n_dims=128`, `freq_base=1000000`, `freq_scale=1`.
- Context/position range: `[0, 8192)`.
- Channel index range: `[0, 64)`; `channel_idx` is the rotary pair index (`i0 / 2`).
- Data layout: little-endian F32 `[position][channel_idx][cos,sin]`.
- Generation method: run the current CUDA `GGML_OP_ROPE` kernel with a GPT-NeoX basis input whose first rotary half is 1 and second half is 0. The output halves are therefore the kernel's actual cos and sin values.
- Baseline difference: this is a focused operator export rather than a model inference comparison; model RoPE parameters and requested context size match the local 8192 baseline.
- Exported values: 1,048,576 F32 values / 4,194,304 bytes.
- SHA-256: `9638aa97a8e0d31064cc423809bbb998d6f1be0200bc3d57b0bd55c8911a22d6`.
- Numerical validation: maximum `abs(cos^2 + sin^2 - 1)` is `5.00036066e-07`.
- Switch validation: `GGML_CUDA_ROPE_QEMU_ENABLED=1` reaches the new hook, disables CUDA graph reuse for RoPE, and uses the documented CUDA fallback. Enabled and disabled exports are bit-identical.
- Build validation: `ggml-cuda` and `llama-cli` both rebuilt successfully in `build_cuda`.
- Model smoke test: `llama-cli` completed with `-c 8192`, the baseline model/cache/GPU parameters, and the RoPE QEMU hook enabled; see `llama-cli-smoke.log`.

Query example (ranges use an exclusive end):

```bash
./query-rope-cos-sin.py --position 100:104 --channel-idx 0:8
```
