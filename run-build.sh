#!/bin/bash

rm -rf build
cmake -B build -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug -j8

cmake --build . --target ggml-c100 -j$(nproc)
