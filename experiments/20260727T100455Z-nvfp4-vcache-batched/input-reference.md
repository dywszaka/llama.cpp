# Input Reference

- Baseline contract: /home/allen/host_workspace/develop/llama.cpp-vp-opt/expt-baseline.md
- Baseline data directory: /home/allen/host_workspace/develop/llama.cpp-vp-opt/experiments/kld-baseline-data
- Baseline log-prob files: /home/allen/host_workspace/develop/llama.cpp-vp-opt/experiments/kld-baseline-data/baseline-logprobs/ubatch_<N>.kld
- Dataset manifest: /home/allen/host_workspace/develop/llama.cpp-vp-opt/experiments/kld-baseline-data/data/wikitext-small.manifest.json
- Small prompt: /home/allen/host_workspace/develop/llama.cpp-vp-opt/experiments/kld-baseline-data/data/wikitext-small.raw
- Binary: /home/allen/host_workspace/develop/llama.cpp-vp-opt/build_cuda/bin/llama-perplexity
- Model: /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf
- CUDA device: CUDA_VISIBLE_DEVICES=0
- Fixed args: --n_gpu_layers 40, --batch-size 512, -t 32, -c 512, --kv-unified, --no-warmup, --chunks 8
- Experiment matrix: native_slice:f16:nvfp4:- batched:f16:nvfp4:GGML_CUDA_NVFP4_VCACHE_BATCHED=1
- Experiment per case: --kl-divergence with the matching baseline-logprobs/ubatch_<N>.kld from /home/allen/host_workspace/develop/llama.cpp-vp-opt/experiments/kld-baseline-data
- Tooling: tools/kld
- Diagnostic contract: tools/kld/collection-contract.md
- Diagnostic artifact policy: keep histograms and bounded row samples only; do not dump full K, V, attention-score, probability, or logits tensors.
- NVFP4 V-cache runtime requirements: flash attention disabled by omission, KQV offload enabled by omission, --kv-unified enabled.
