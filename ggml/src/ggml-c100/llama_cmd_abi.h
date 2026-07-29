#ifndef LLAMA_CMD_ABI_H
#define LLAMA_CMD_ABI_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

// Shared llama.cpp CMD/RESULT ABI.
// This file is intentionally duplicated in firmware/ and ext/llama.cpp/.

enum {
    LLAMA_CMD_REGION_BASE = 0x20FF0000u,
    LLAMA_RESULT_REGION_BASE = 0x20FF1000u,
};

#define LLAMA_CMD_MAGIC 0x43313030
#define LLAMA_RESULT_MAGIC 0x43313031

#define LLAMA_CMD_ID_SOFTMAX 0x01
#define LLAMA_CMD_ID_ADD 0x02
#define LLAMA_CMD_ID_MUL 0x03
#define LLAMA_CMD_ID_RMS_NORM 0x04
#define LLAMA_CMD_ID_SILU 0x05
#define LLAMA_CMD_ID_ROPE 0x06
#define LLAMA_CMD_ID_EXT_PARAM_DEBUG 0x07
#define LLAMA_CMD_ID_GLU 0x08
#define LLAMA_CMD_ID_REDUCE_ADD 0x09

#define LLAMA_CMD_ID_MUL_MAT 0x20

#define LLAMA_CMD_ID_GET_ROWS 0x10
#define LLAMA_CMD_ID_RESHAPE 0x11
#define LLAMA_CMD_ID_VIEW 0x12
#define LLAMA_CMD_ID_CPY 0x13
#define LLAMA_CMD_ID_TRANSPOSE 0x14
#define LLAMA_CMD_ID_PERMUTE 0x15
#define LLAMA_CMD_ID_CONT 0x16

#define LLAMA_SOFTMAX_FLAG_HAS_MASK 0x1u
#define LLAMA_GLU_FLAG_HAS_SRC1 0x1u
#define LLAMA_GLU_FLAG_SWAPPED 0x2u

#define LLAMA_CMD_FLAG_EXT_PARAM 0x1u

#define LLAMA_CPY_MODE_SHIFT 8u
#define LLAMA_CPY_MODE_MASK (0x3u << LLAMA_CPY_MODE_SHIFT)
#define LLAMA_CPY_MODE_CE 0u
#define LLAMA_CPY_MODE_S2M 1u
#define LLAMA_CPY_MODE_M2S 2u
#define LLAMA_CPY_DMA_ID_MASK 0x3u

static inline uint32_t llama_cmd_cpy_mode(uint32_t flags) {
    return (flags & LLAMA_CPY_MODE_MASK) >> LLAMA_CPY_MODE_SHIFT;
}

static inline uint32_t llama_cmd_cpy_flags(uint32_t flags, uint32_t mode) {
    return (flags & ~LLAMA_CPY_MODE_MASK) |
           ((mode << LLAMA_CPY_MODE_SHIFT) & LLAMA_CPY_MODE_MASK);
}

#define LLAMA_STATUS_IDLE 0
#define LLAMA_STATUS_RUNNING 1
#define LLAMA_STATUS_DONE 2
#define LLAMA_STATUS_ERROR 3

#define LLAMA_CMD_STATUS_OFFSET 0x34

#define LLAMA_ERROR_SUCCESS 0
#define LLAMA_ERROR_INVALID_CMD_ID 1
#define LLAMA_ERROR_INVALID_ADDRESS 2
#define LLAMA_ERROR_VE_EXECUTION_FAILED 3
#define LLAMA_ERROR_TIMEOUT 4
#define LLAMA_ERROR_SU_VE_COMM_FAILED 5

#define LLAMA_EXT_PARAM_MAGIC 0x45585042u
#define LLAMA_EXT_PARAM_VERSION 1u
#define LLAMA_EXT_PARAM_HEADER_SIZE 32u

#define LLAMA_EXT_PARAM_DEBUG_PAYLOAD_MAGIC 0x45504447u
#define LLAMA_EXT_PARAM_DEBUG_PAYLOAD_VERSION 1u
#define LLAMA_EXT_PARAM_DEBUG_CHECK_XOR 0xC100E001u

#define LLAMA_SOFTMAX_EXT_PAYLOAD_MAGIC 0x534D5845u
#define LLAMA_SOFTMAX_EXT_PAYLOAD_VERSION 1u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t total_size;
    uint32_t cmd_id;
    uint32_t flags;
    uint32_t payload_size;
    uint32_t reserved;
} llama_ext_param_header_t;

typedef struct {
    uint32_t payload_magic;
    uint32_t payload_version;
    uint32_t value0;
    uint32_t value1;
    uint32_t checksum;
} llama_ext_param_debug_payload_t;

typedef struct {
    uint32_t payload_magic;
    uint32_t payload_version;
    uint32_t mask_ne2;
    uint32_t mask_ne3;
} llama_softmax_ext_payload_t;

static inline uint32_t llama_ext_param_debug_checksum(uint32_t value0, uint32_t value1) {
    return value0 ^ value1 ^ LLAMA_EXT_PARAM_DEBUG_CHECK_XOR;
}

static inline int llama_ext_param_validate_header(const llama_ext_param_header_t* block,
                                                  uint32_t cmd_id,
                                                  uint32_t min_payload_size) {
    if (block == 0) {
        return 0;
    }
    if (block->magic != LLAMA_EXT_PARAM_MAGIC) {
        return 0;
    }
    if (block->version != LLAMA_EXT_PARAM_VERSION) {
        return 0;
    }
    if (block->header_size != LLAMA_EXT_PARAM_HEADER_SIZE) {
        return 0;
    }
    if (block->total_size < LLAMA_EXT_PARAM_HEADER_SIZE) {
        return 0;
    }
    if (block->cmd_id != cmd_id) {
        return 0;
    }
    if (block->payload_size < min_payload_size) {
        return 0;
    }
    if (block->payload_size > block->total_size - LLAMA_EXT_PARAM_HEADER_SIZE) {
        return 0;
    }
    return 1;
}

static inline const void* llama_ext_param_payload(const llama_ext_param_header_t* block) {
    return (const void*)((const uint8_t*)block + block->header_size);
}

typedef struct {
    uint32_t cmd_magic;
    uint32_t cmd_id;
    uint64_t src0_addr;
    uint32_t src0_size;
    uint64_t src1_addr;
    uint32_t src1_size;
    uint64_t dst_addr;
    uint32_t dst_size;
    uint32_t status;
    uint32_t flags;
    uint64_t ext_param_addr;
    uint32_t ext_param_size;
    uint32_t params[8];
} llama_cmd_header_t;

typedef struct {
    uint32_t result_magic;
    uint32_t result_code;
    uint32_t cycles;
    uint32_t reserved;
} llama_result_t;

static inline int llama_cmd_is_valid(const llama_cmd_header_t* cmd) {
    return cmd->cmd_magic == LLAMA_CMD_MAGIC;
}

static inline int llama_cmd_is_done(const llama_cmd_header_t* cmd) {
    return cmd->status == LLAMA_STATUS_DONE;
}

static inline int llama_cmd_is_error(const llama_cmd_header_t* cmd) {
    return cmd->status == LLAMA_STATUS_ERROR;
}

#endif  // LLAMA_CMD_ABI_H
