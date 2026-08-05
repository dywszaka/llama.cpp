#pragma once

#include <cstdint>

#define ROPE_FP32_RPC_MAGIC UINT32_C(0x31505252)
#define ROPE_FP32_RPC_VERSION UINT16_C(1)
#define ROPE_FP32_TABLE_POSITIONS UINT32_C(8192)
#define ROPE_FP32_TABLE_CHANNELS UINT32_C(64)

enum rope_fp32_rpc_dtype {
    ROPE_FP32_RPC_DTYPE_INVALID = 0,
    ROPE_FP32_RPC_DTYPE_BF16 = 1,
    ROPE_FP32_RPC_DTYPE_I32 = 2,
    ROPE_FP32_RPC_DTYPE_F32 = 3,
};

enum rope_fp32_rpc_flags {
    ROPE_FP32_RPC_CANONICAL_DENSE = 1u << 0,
    ROPE_FP32_RPC_STATIC_F32_TABLE = 1u << 1,
};

enum rope_fp32_rpc_status {
    ROPE_FP32_RPC_STATUS_OK = 0,
    ROPE_FP32_RPC_STATUS_BAD_REQUEST = 1,
    ROPE_FP32_RPC_STATUS_NOT_READY = 2,
    ROPE_FP32_RPC_STATUS_QEMU_ERROR = 3,
    ROPE_FP32_RPC_STATUS_TIMEOUT = 4,
    ROPE_FP32_RPC_STATUS_INTERNAL_ERROR = 5,
};

#pragma pack(push, 1)

struct rope_fp32_rpc_request_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t request_id;
    uint32_t flags;
    uint32_t src0_type;
    uint32_t pos_type;
    uint32_t dst_type;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    uint32_t position_count;
    uint32_t table_positions;
    uint32_t table_channels;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4];
    uint64_t src0_bytes;
    uint64_t position_bytes;
    uint64_t dst_bytes;
};

struct rope_fp32_rpc_response_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t request_id;
    uint32_t status;
    uint32_t error_code;
    uint64_t output_bytes;
    uint64_t elapsed_ns;
};

#pragma pack(pop)

static_assert(sizeof(rope_fp32_rpc_request_v1) == 152,
        "ROPE FP32 RPC request layout changed");
static_assert(sizeof(rope_fp32_rpc_response_v1) == 40,
        "ROPE FP32 RPC response layout changed");
