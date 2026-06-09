# Input Reference

- baseline=expt-baseline.md
- model=/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf
- data=data/wikitext/wikitext-2-raw/wiki.test.raw
- binary=build_cuda/bin/llama-perplexity
- changed_from_baseline=--cache-type-k nvfp4, LLAMA_NVFP4_KCACHE_OUTLIER=1
- threshold_profile=balanced default from src/llama-kv-cache-nvfp4-outlier-config.h
- capacity_profile=balanced default from src/llama-kv-cache-nvfp4-outlier-config.h
