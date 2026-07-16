#pragma once

#include <cstdint>

static constexpr uint32_t SOFTMAX_RPC_MAGIC = UINT32_C(0x31584d53);
static constexpr uint16_t SOFTMAX_RPC_VERSION = UINT16_C(1);

enum softmax_rpc_mask_type : uint32_t {
    SOFTMAX_RPC_MASK_NONE = 0,
    SOFTMAX_RPC_MASK_F16 = 1,
    SOFTMAX_RPC_MASK_F32 = 2,
};

enum softmax_rpc_request_flags : uint32_t {
    SOFTMAX_RPC_REQUEST_HAS_MASK = 1u << 0,
    SOFTMAX_RPC_REQUEST_HAS_SINKS = 1u << 1,
    SOFTMAX_RPC_REQUEST_BF16_IO = 1u << 2,
};

enum softmax_rpc_status : uint32_t {
    SOFTMAX_RPC_STATUS_OK = 0,
    SOFTMAX_RPC_STATUS_BAD_REQUEST = 1,
    SOFTMAX_RPC_STATUS_NOT_READY = 2,
    SOFTMAX_RPC_STATUS_QEMU_ERROR = 3,
    SOFTMAX_RPC_STATUS_TIMEOUT = 4,
    SOFTMAX_RPC_STATUS_INTERNAL_ERROR = 5,
};

#pragma pack(push, 1)

struct softmax_rpc_request_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t request_id;
    uint32_t mask_type;
    uint32_t flags;

    int64_t nheads;
    uint32_t n_head_log2;
    uint32_t reserved0;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;
    int64_t ne12;
    int64_t ne13;

    float scale;
    float max_bias;
    float m0;
    float m1;

    uint64_t src0_bytes;
    uint64_t src1_bytes;
    uint64_t src2_bytes;
    uint64_t dst_bytes;
};

struct softmax_rpc_response_v1 {
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

static_assert(sizeof(softmax_rpc_request_v1) == 184, "softmax RPC request layout changed");
static_assert(sizeof(softmax_rpc_response_v1) == 40, "softmax RPC response layout changed");
