FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        device-tree-compiler \
        libcurl4 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*
