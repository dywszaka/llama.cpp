# Validation

Expected checks after running the lab:

```bash
tests/test-kcache-mean-chunks-source.sh
cmake --build build_cuda --target llama-kcache-mean -j 8
python3 mylab/kqv-heatmap/scripts/validate_outputs.py
```

Expected raw tensor file sizes:

- Q: 36 files, each `8,388,608` bytes
- KQ: 36 files, each `33,554,432` bytes
- V: 36 files, each `2,097,152` bytes
- VP: 36 files, each `8,388,608` bytes

Expected heatmap image dimensions:

- Q layer 0: `602 x 4158`
- KQ layer 0: `602 x 16446`
- V layer 0: `602 x 1086`
- VP layer 0: `602 x 4158`
