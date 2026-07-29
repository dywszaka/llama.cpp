#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HCU_CPP="${ROOT_DIR}/c100-sim/src/top/hcu.cpp"
SPIKE_DEVICE_ADDR="${ROOT_DIR}/c100-sim/ext/riscv-isa-sim-lib-demo/src/top/common/device_addr.h"

grep -Fq 'add_host_memory_region(DeviceAddress::C100_SU_DRAM_REMAP_BASE,' "${HCU_CPP}"
grep -Fq 'DeviceAddress::C100_SU_DRAM_REMAP_SIZE,' "${HCU_CPP}"
grep -Fq 'global_ptr + DeviceAddress::C100_SU_DRAM_REMAP_TARGET_OFFSET' "${HCU_CPP}"
grep -Fq '#define GLOBAL_SHM_BASE      0x100000000ULL' "${SPIKE_DEVICE_ADDR}"
grep -Fq '#define GLOBAL_SHM_SIZE      0xC0000000ULL' "${SPIKE_DEVICE_ADDR}"
