#pragma once

#include <cstdint>

static constexpr uint32_t GLU_RPC_MAGIC = UINT32_C(0x31554c47);
static constexpr uint16_t GLU_RPC_VERSION = UINT16_C(1);

enum glu_rpc_dtype : uint32_t {
    GLU_RPC_DTYPE_INVALID = 0,
    GLU_RPC_DTYPE_BF16 = 1,
};

enum glu_rpc_request_flags : uint32_t {
    GLU_RPC_REQUEST_CANONICAL_DENSE = 1u << 0,
    GLU_RPC_REQUEST_SWIGLU = 1u << 1,
};

enum glu_rpc_status : uint32_t {
    GLU_RPC_STATUS_OK = 0,
    GLU_RPC_STATUS_BAD_REQUEST = 1,
    GLU_RPC_STATUS_NOT_READY = 2,
    GLU_RPC_STATUS_QEMU_ERROR = 3,
    GLU_RPC_STATUS_TIMEOUT = 4,
    GLU_RPC_STATUS_INTERNAL_ERROR = 5,
};

#pragma pack(push, 1)

struct glu_rpc_request_v1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t request_id;
    uint32_t flags;
    uint32_t src0_type;
    uint32_t src1_type;
    uint32_t dst_type;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    uint64_t src0_bytes;
    uint64_t src1_bytes;
    uint64_t dst_bytes;
};

struct glu_rpc_response_v1 {
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

static_assert(sizeof(glu_rpc_request_v1) == 88, "GLU RPC request layout changed");
static_assert(sizeof(glu_rpc_response_v1) == 40, "GLU RPC response layout changed");
