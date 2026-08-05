#pragma once

#include <cstdint>

static constexpr uint32_t RMS_NORM_RPC_MAGIC = UINT32_C(0x31534d52);
static constexpr uint16_t RMS_NORM_RPC_VERSION = UINT16_C(1);

enum rms_norm_rpc_dtype : uint32_t {
    RMS_NORM_RPC_DTYPE_INVALID = 0,
    RMS_NORM_RPC_DTYPE_BF16 = 1,
};

enum rms_norm_rpc_request_flags : uint32_t {
    RMS_NORM_RPC_REQUEST_CANONICAL_DENSE = 1u << 0,
};

enum rms_norm_rpc_status : uint32_t {
    RMS_NORM_RPC_STATUS_OK = 0,
    RMS_NORM_RPC_STATUS_BAD_REQUEST = 1,
    RMS_NORM_RPC_STATUS_NOT_READY = 2,
    RMS_NORM_RPC_STATUS_QEMU_ERROR = 3,
    RMS_NORM_RPC_STATUS_TIMEOUT = 4,
    RMS_NORM_RPC_STATUS_INTERNAL_ERROR = 5,
};

#pragma pack(push, 1)

struct rms_norm_rpc_request_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t request_id;
    uint32_t flags;
    uint32_t src_type;
    uint32_t dst_type;
    uint32_t reserved0;

    int64_t ncols;
    int64_t nrows;
    int64_t nchannels;
    int64_t nsamples;

    float eps;
    uint32_t reserved1;

    uint64_t src0_bytes;
    uint64_t dst_bytes;
};

struct rms_norm_rpc_response_v1 {
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

static_assert(sizeof(rms_norm_rpc_request_v1) == 88, "RMS_NORM RPC request layout changed");
static_assert(sizeof(rms_norm_rpc_response_v1) == 40, "RMS_NORM RPC response layout changed");
