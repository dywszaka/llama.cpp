# Summary

Sample LLAMA_CPP-12 offline evaluator fixture.

The run evaluates one synthetic K tensor against the current CPU NVFP4 reference
quantize/dequantize baseline. This is a fixture validation artifact, not a model
runtime export or performance measurement.

Observed metrics from the current tool run:

- kind: `k`
- n: 16
- MAE: 0.390625
- MSE: 0.2841796875
- RMSE: 0.5330850659134994

Validation command:

```text
build-llama-cpp-12/bin/llama-tensor-export-eval --manifest experiments/LLAMA_CPP-12-tensor-export-eval-sample/manifest.json --global-scale 1
```
