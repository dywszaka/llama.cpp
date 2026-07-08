#!/bin/bash

rm -rf build build_cuda build-cuda-c100 log/
cmake -B build -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug -j8

cmake --build build --target ggml-c100 -j"$(nproc)"

cuda_arch="${CMAKE_CUDA_ARCHITECTURES:-80}"
cmake -S . -B build_cuda \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build_cuda --config Release -j "$(nproc)" --target llama-perplexity

cat <<'EOF'

[NOTE] Run from the repository root through Docker:
docker run --rm -v "$PWD:/app" -w /app llama.cpp.sim bash -lc \
  'cd ext/llama.cpp && bash run-build.sh'
EOF
exit
