# NVFP4 Debug Context (llama.cpp)

## Summary
- Added a CPU-only NVFP4 activation quant->dequant simulation and inserted it into the Qwen3 graph for all NVFP4 matmuls (Q/K/V, WO, FFN up/gate/down).
- Exported `ggml_nvfp4_to_f32_op` for test use and added a unit test that validates correctness vs. a reference implementation.
- CMake tests were reconfigured in `build` with `-DLLAMA_BUILD_TESTS=ON` and `test-nvfp4-op` passes.

## Current Behavior
- With `LLAMA_NVFP4_WEIGHT_SCALE_2_RECIP=1` (treat `weight_scale_2` as reciprocal global scale), NaNs go away, but output is still incorrect (e.g., repeated `ffffffff`).
- After inserting activation quant->dequant, model still needs verification for correctness.

## Key Code Changes

### NVFP4 activation quant/dequant op (CPU only)
- File: `src/llama-model.cpp`
- New helpers inside anonymous namespace:
  - `nvfp4_quantize_row_ref`, `nvfp4_dequantize_row_ref`, `nvfp4_to_f32_ref`
  - `nvfp4_fp32_to_e4m3`, `nvfp4_fp4_quantize` (E2M1 values), `nvfp4_clampf`
- New op implementation: `nvfp4_to_f32_op_impl` (uses `ggml_map_custom2`)
- Exported symbol:
  - `extern "C" void ggml_nvfp4_to_f32_op(...)` (wrapper calling the internal impl)
- Behavior:
  - Only handles contiguous F32 2D tensors (ne2/3 == 1) and `ncols % QK_NVFP4 == 0`.
  - Uses `global_scale` from the second tensor (input_scale). If `global_scale == 0` or shape not supported, it copies input to output.
  - Honors `LLAMA_NVFP4_WEIGHT_SCALE_2_RECIP=1` by inverting `global_scale` before use.

### Qwen3 graph integration
- File: `src/llama-model.cpp` (`llm_build_qwen3`)
- `build_lora_mm_scaled` signature changed to accept `w_inp_scale`.
- For NVFP4 weights, before `ggml_mul_mat`:
  - `x_used = ggml_map_custom2(ctx0, x_used, w_inp_scale, ggml_nvfp4_to_f32_op, GGML_N_TASKS_MAX, nullptr)`
- All Q/K/V/WO/FFN matmuls now pass the corresponding `*_inp_scale`.
- Output scaling still uses `weight_scale_2` with `LLAMA_NVFP4_WEIGHT_SCALE_2_RECIP` toggle:
  - if set, multiply by `weight_scale_2`, else divide.

### Test exposure + unit test
- `src/llama-nvfp4.h` added with `ggml_nvfp4_to_f32_op` declaration.
- `tests/test-nvfp4-op.cpp` added:
  - compares `ggml_nvfp4_to_f32_op` vs reference quant->dequant
  - validates copy behavior for `global_scale == 0` and non-multiple of `QK_NVFP4`
- `tests/CMakeLists.txt` updated to add `test-nvfp4-op`.

## Environment Variables
- `LLAMA_NVFP4_WEIGHT_SCALE_2_RECIP=1`:
  - Treats `weight_scale_2` as reciprocal of global scale.
  - Also inverts `global_scale` in the activation quant->dequant op.

## Tests
- Build configuration (from repo root):
  - `cmake -S . -B build -DLLAMA_BUILD_TESTS=ON`
- Build test:
  - `cmake --build build --target test-nvfp4-op`
- Run:
  - `ctest -R test-nvfp4-op -V` (in `build`)
- Result: `test-nvfp4-op` passed.

## Notes / Open Items
- Activation quant->dequant is CPU-only and will slow inference.
- Output still incorrect despite stable numeric ranges; likely remaining scale mismatch or missing step in NVFP4 simulation.
- `weight_scale_2` values are ~2e-4; treating them as reciprocal (`LLAMA_NVFP4_WEIGHT_SCALE_2_RECIP=1`) avoids NaNs but output is still wrong.

## Debug Analysis (2026-01-12)

### Identified Issues

1. **kvalues_nvfp4 range mismatch**:
   - `kvalues_nvfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}`
   - Range is **[-12, 12]** (comment says "e2m1 values (doubled)")
   - But `nvfp4_fp4_quantize` uses `k_fp4_max = 6`, range is **[-6, 6]**
   - This is a 2x mismatch that could cause incorrect quantization

2. **Weight dequantization format** (`ggml/src/ggml-quants.c:443-461`):
   ```c
   const float d = GGML_E4M3_TO_FP32_HALF(x[i].e);  // per-block scale
   y[i*qk + j] = kvalues_nvfp4[index] * d;
   ```

3. **Activation quantize scale formula** (`src/llama-model.cpp:116-122`):
   ```cpp
   float scale = global_scale * (vmax / k_fp4_max);
   const float inv_scale = global_scale / scale_f;
   ```
   - This computes per-block scale AND uses global_scale
   - May not match the NVFP4 tensor core matmul semantics

4. **Activation dequantize scale formula** (`src/llama-model.cpp:144`):
   ```cpp
   const float out_scale = scale_f / global_scale;
   ```

### Debug Plan

1. **Add detailed logging** to track scale values at each step
2. **Fix kvalues_nvfp4 range mismatch** (12 vs 6)
3. **Simplify activation path** - test direct multiplication by global_scale instead of quant->dequant
4. **Analyze weight quantization formula** to determine correct activation scaling
5. **Verify correctness** after fix

### NVFP4 Matmul Semantics (to be verified)

The NVFP4 tensor core matmul may expect:
- Weights: NVFP4 format (per-block E4M3 scale + 4-bit E2M1 values)
- Activation: FP32 multiplied by global_scale (1/weight_scale_2)
- Output: needs scaling by weight_scale_2 to recover correct values

