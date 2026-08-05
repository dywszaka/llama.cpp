# Environment

- UTC run family: `2026-07-27T10:04:55Z`
- Source base: `36bdb74eff4661a2c1ba19d33dd9306c73b26479`
- Branch: `expt/nvfp4-vp-batched`
- Build: `Release`, CUDA enabled, native CUDA architecture
- GPU used: NVIDIA GeForce RTX 5090, compute capability 12.0
- Driver: 595.80
- CUDA toolkit: 13.0.88
- Device selection: `CUDA_VISIBLE_DEVICES=0`
- Direct A/B variable: `GGML_CUDA_NVFP4_VCACHE_BATCHED` unset versus `1`
- V-cache benchmark change from the repository baseline: `--cache-type-v nvfp4`, because V-cache P*V is the subject of the experiment.
- KLD baseline: locally generated f16/f16 sparse log-prob baseline for ubatch 512 from the fixed eight-document Wikitext sample.
