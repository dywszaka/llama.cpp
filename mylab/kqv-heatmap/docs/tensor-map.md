# Tensor Map

This lab captures graph callback tensors from the non-flash attention path.

## Q

- Source tensor name: `Qcur-*`
- Runtime shape from graph log: `ne=[128,512,32,1]`
- Heatmap shape: `512 tokens x 4096 channels`
- Channel flattening: `head * 128 + head_dim`

## KQ

- Source tensor name: `kq-*`
- Produced by `ggml_mul_mat(ctx0, k, q)`
- Runtime shape from graph log: `ne=[512,512,32,1]`
- Heatmap shape: `512 query tokens x 16384 flattened key/head channels`
- Channel flattening: `head * 512 + key_token`

## V

- Source tensor name: `Vcur-*`
- Runtime shape: `ne=[128,8,512,1]`
- Heatmap shape: `512 tokens x 1024 channels`
- Channel flattening: `kv_head * 128 + head_dim`

## VP

- Source tensor name: `kqv-*`
- Produced by `ggml_mul_mat(ctx0, v, kq)` after KQ softmax
- Runtime shape from graph log: `ne=[128,512,32,1]`
- Heatmap shape: `512 tokens x 4096 channels`
- Channel flattening: `head * 128 + head_dim`

VP is the attention value result before final permutation/contiguous reshape and
before the output projection.
