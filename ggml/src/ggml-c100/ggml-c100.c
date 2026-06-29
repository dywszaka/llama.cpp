/**
 * @file ggml-c100.c
 * @brief C100 GGML Backend implementation
 *
 * This backend enables llama.cpp to offload compute operations
 * to the C100 simulator (SU + VE architecture).
 */

#include "ggml-c100.h"
#include "ggml-c100-impl.h"
#include "llama-cmd.h"  // C-compatible command structures
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ============================================================================
// External C API from llama_cpp.cpp
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Simulator singleton access
void* get_simulator_instance(void);

// llama_cpp.cpp C API functions
bool c100_llama_write_cmd(const LlamaCmdHeader* cmd);
bool c100_llama_read_cmd(LlamaCmdHeader* cmd);
bool c100_llama_read_result(LlamaResult* result);
bool c100_llama_write_result(const LlamaResult* result);
uint32_t c100_llama_read_cmd_status(void);

// Tensor memory allocation in simulator space
uint64_t c100_llama_alloc_tensor(size_t size);
uint64_t c100_llama_alloc_global(size_t size);
uint64_t c100_llama_alloc_local(size_t size);
void* c100_llama_get_host_ptr(uint64_t sim_addr);
uint64_t c100_llama_create_ext_param_block(void* cmd,
                                           uint32_t cmd_id,
                                           uint32_t flags,
                                           const void* payload,
                                           uint32_t payload_size);

#ifdef __cplusplus
}
#endif

// ============================================================================
// Global State
// ============================================================================

static const char* GGML_C100_NAME = "C100";

// Default polling configuration
#define C100_DEFAULT_POLL_INTERVAL_US  1000   // 1ms
#define C100_DEFAULT_MAX_POLL_ITERATIONS 50000  // 50 seconds max

static uint32_t c100_float_to_u32(float value) {
    union {
        float f32;
        uint32_t u32;
    } converted = {.f32 = value};
    return converted.u32;
}

// ============================================================================
// Forward declarations
// ============================================================================

ggml_backend_t ggml_backend_c100_init(void);
ggml_backend_buffer_type_t ggml_backend_c100_buffer_type(void);

static bool c100_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// ============================================================================
// C100 Buffer Context
// ============================================================================

typedef struct c100_buffer_context {
    uint64_t sim_addr;   // Simulator address (0x20000000+)
    void* host_ptr;      // Host pointer for direct access
    size_t size;         // Buffer size
} c100_buffer_context_t;

// ============================================================================
// CMD/RESULT Operations
// ============================================================================

/**
 * @brief Get simulator address from tensor
 *
 * The tensor->data points to host memory, but we need the simulator address
 * for the firmware. Calculate it from the buffer context.
 */
static uint64_t c100_get_sim_addr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->buffer || !tensor->buffer->context) {
        return 0;
    }

    c100_buffer_context_t* ctx = (c100_buffer_context_t*)tensor->buffer->context;
    uint8_t* base_host = (uint8_t*)ctx->host_ptr;
    uint8_t* tensor_host = (uint8_t*)tensor->data;

    // Calculate offset within buffer
    uint64_t offset = tensor_host - base_host;

    // Return simulator address
    return ctx->sim_addr + offset;
}

/**
 * @brief Prepare a CMD header for SoftMax operation
 */
static void c100_prepare_softmax_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_SOFTMAX;  // 0x01

    // Get simulator addresses
    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    const struct ggml_tensor* mask = dst->src[1];
    const float* op_params = (const float*)dst->op_params;
    const float scale = op_params ? op_params[0] : 1.0f;
    const float max_bias = op_params ? op_params[1] : 0.0f;
    uint32_t rows = src->ne[1] > 0 ? src->ne[1] : 1;
    uint32_t cols = src->ne[0];

    if (mask) {
        cmd->src1_addr = c100_get_sim_addr(mask);
        cmd->src1_size = ggml_nbytes(mask);
    }

    bool is_f32 = src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
                  (!mask || mask->type == GGML_TYPE_F32);
    bool requires_extended = mask || scale != 1.0f || max_bias != 0.0f || is_f32;
    uint32_t flags = 0;
    if (mask) {
        flags |= CMD_SOFTMAX_FLAG_HAS_MASK;
    }
    if (is_f32) {
        flags |= CMD_SOFTMAX_FLAG_F32;
    }
    if (requires_extended) {
        flags |= CMD_SOFTMAX_FLAG_SHAPE4D;
    }

    // params[0] keeps the legacy rows/cols encoding for plain SoftMax.
    cmd->params[0] = ((uint32_t)cols << 16) | rows;
    if (requires_extended) {
        cmd->params[0] = CMD_SOFTMAX_PACK_U16_PAIR(src->ne[0], src->ne[1]);
        cmd->params[1] = CMD_SOFTMAX_EXT_MAGIC;
        cmd->params[2] = flags;
        cmd->params[3] = c100_float_to_u32(scale);
        cmd->params[4] = c100_float_to_u32(max_bias);
        cmd->params[5] = CMD_SOFTMAX_PACK_U16_PAIR(src->ne[2], src->ne[3]);
        cmd->params[6] = mask ? CMD_SOFTMAX_PACK_U16_PAIR(mask->ne[1], mask->ne[2]) : 0;
        cmd->params[7] = mask ? CMD_SOFTMAX_PACK_U16_PAIR(mask->ne[3], 0) : 0;
    }

    cmd->status = CMD_STATUS_RUNNING;  // Set to RUNNING so firmware processes it
}

/**
 * @brief Prepare a CMD header for Add operation (element-wise)
 */
static void c100_prepare_add_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_ADD;  // 0x02

    // Get simulator addresses for both inputs
    cmd->src0_addr = c100_get_sim_addr(src0);
    cmd->src0_size = ggml_nbytes(src0);
    cmd->src1_addr = c100_get_sim_addr(src1);
    cmd->src1_size = ggml_nbytes(src1);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    // Total elements in params (uint32_t, limited to 32-bit value)
    uint32_t total_elements = (uint32_t)ggml_nelements(src0);
    cmd->params[0] = total_elements;

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for Mul operation (element-wise)
 */
static void c100_prepare_mul_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_MUL;  // 0x03

    cmd->src0_addr = c100_get_sim_addr(src0);
    cmd->src0_size = ggml_nbytes(src0);
    cmd->src1_addr = c100_get_sim_addr(src1);
    cmd->src1_size = ggml_nbytes(src1);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    uint64_t total_elements = ggml_nelements(src0);
    cmd->params[0] = total_elements;

    // Test mode: set flag if environment variable is set
    if (getenv("C100_TEST_MODE") != NULL) {
        cmd->params[1] = 0xDEADBEEF;  // Test mode flag: skip actual VE execution
    }

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for SiLU operation
 */
static void c100_prepare_silu_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_SILU;  // 0x05

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    uint64_t total_elements = ggml_nelements(src);
    cmd->params[0] = total_elements;

    // Test mode: set flag if environment variable is set
    if (getenv("C100_TEST_MODE") != NULL) {
        cmd->params[1] = 0xDEADBEEF;  // Test mode flag: skip actual VE execution
    }

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for RoPE operation
 */
static void c100_prepare_rope_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst,
    int n_dims,
    int mode
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_ROPE;  // 0x06

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    // params: lower 16 bits = n_dims, upper 16 bits = mode (uint32_t)
    cmd->params[0] = ((uint32_t)mode << 16) | (uint32_t)n_dims;

    // Test mode: set flag if environment variable is set
    if (getenv("C100_TEST_MODE") != NULL) {
        cmd->params[1] = 0xDEADBEEF;  // Test mode flag: skip actual VE execution
    }

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for CPY operation (Copy Engine)
 */
static void c100_prepare_cpy_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_CPY;  // 0x13

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    // Encode type information
    // params[0] = (dst_type << 16) | src_type
    uint32_t src_type = (uint32_t)src->type;
    uint32_t dst_type = (uint32_t)dst->type;
    cmd->params[0] = (dst_type << 16) | src_type;

    // params[1] = src_type_size (element size in bytes)
    // params[2] = src_block_size (for quantized types)
    // params[3] = dst_type_size
    cmd->params[1] = (uint32_t)ggml_type_size(src->type);
    cmd->params[2] = (uint32_t)ggml_blck_size(src->type);
    cmd->params[3] = (uint32_t)ggml_type_size(dst->type);

    // params[4-5] = src ne dimensions (packed as hi:lo 16-bit each)
    // params[6-7] = dst ne dimensions
    // ne values truncated to uint16_t (max 65535), sufficient for current test shapes
    uint32_t src_ne0 = (uint32_t)(src->ne[0] > 65535 ? 65535 : src->ne[0]);
    uint32_t src_ne1 = (uint32_t)(src->ne[1] > 65535 ? 65535 : src->ne[1]);
    uint32_t src_ne2 = (uint32_t)(src->ne[2] > 65535 ? 65535 : src->ne[2]);
    uint32_t src_ne3 = (uint32_t)(src->ne[3] > 65535 ? 65535 : src->ne[3]);
    uint32_t dst_ne0 = (uint32_t)(dst->ne[0] > 65535 ? 65535 : dst->ne[0]);
    uint32_t dst_ne1 = (uint32_t)(dst->ne[1] > 65535 ? 65535 : dst->ne[1]);
    uint32_t dst_ne2 = (uint32_t)(dst->ne[2] > 65535 ? 65535 : dst->ne[2]);
    uint32_t dst_ne3 = (uint32_t)(dst->ne[3] > 65535 ? 65535 : dst->ne[3]);

    cmd->params[4] = (src_ne1 << 16) | src_ne0;
    cmd->params[5] = (src_ne3 << 16) | src_ne2;
    cmd->params[6] = (dst_ne1 << 16) | dst_ne0;
    cmd->params[7] = (dst_ne3 << 16) | dst_ne2;

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for TRANSPOSE operation (Copy Engine)
 */
static void c100_prepare_transpose_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_TRANSPOSE;  // 0x14

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for PERMUTE operation (Copy Engine)
 */
static void c100_prepare_permute_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_PERMUTE;  // 0x15

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for GET_ROWS operation (Copy Engine)
 */
static void c100_prepare_get_rows_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    const struct ggml_tensor* rows,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_GET_ROWS;  // 0x10

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->src1_addr = c100_get_sim_addr(rows);
    cmd->src1_size = ggml_nbytes(rows);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for MUL_MAT operation (Tensor Engine)
 */
static void c100_prepare_mul_mat_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_MUL_MAT;  // 0x20 (Tensor Engine)

    cmd->src0_addr = c100_get_sim_addr(src0);
    cmd->src0_size = ggml_nbytes(src0);
    cmd->src1_addr = c100_get_sim_addr(src1);
    cmd->src1_size = ggml_nbytes(src1);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    // Fill matrix dimensions for GEMM configuration
    // ggml_mul_mat(A, B): A is M×K, B is N×K (transposed storage), result is M×N
    // src0->ne[0] = K, src0->ne[1] = M
    // src1->ne[0] = K, src1->ne[1] = N
    cmd->params[0] = src0->ne[1];  // M
    cmd->params[1] = src0->ne[0];  // K
    cmd->params[2] = src1->ne[1];  // N

    // Type sizes for tensor_descriptor
    cmd->params[3] = (uint32_t)ggml_type_size(src0->type);  // type_size_A
    cmd->params[4] = (uint32_t)ggml_type_size(src1->type);  // type_size_B
    cmd->params[5] = (uint32_t)ggml_type_size(dst->type);   // type_size_C

    // Batch dimensions for 4D tensor support
    // ne[2] = batch dimension (bs), ne[3] = num repeats (nr)
    // params[6] = bs[0] (batch size), params[7] = nr[0] (num repeats)
    uint32_t batch0 = (src0->ne[2] > 0) ? (uint32_t)src0->ne[2] : 1;
    uint32_t batch1 = (src0->ne[3] > 0) ? (uint32_t)src0->ne[3] : 1;
    cmd->params[6] = batch0;  // batch size (bs)
    cmd->params[7] = batch1;  // num repeats (nr)

    if (getenv("C100_TEST_MODE") != NULL) {
        cmd->params[7] = 0xDEADBEEF;  // Test mode flag: skip actual TE execution
    }

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for RESHAPE operation (Copy Engine - metadata, may be NOP)
 */
static void c100_prepare_reshape_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_RESHAPE;  // 0x11

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for VIEW operation (Copy Engine - metadata, may be NOP)
 */
static void c100_prepare_view_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_VIEW;  // 0x12

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Prepare a CMD header for CONT operation (Copy Engine)
 */
static void c100_prepare_cont_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_CONT;  // 0x16

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    cmd->status = CMD_STATUS_RUNNING;
}

/**
 * @brief Poll CMD status until done
 */
static void c100_prepare_rms_norm_cmd(
    LlamaCmdHeader* cmd,
    const struct ggml_tensor* src,
    struct ggml_tensor* dst,
    float eps
) {
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_magic = CMD_MAGIC;
    cmd->cmd_id = CMD_ID_RMS_NORM;  // 0x04

    cmd->src0_addr = c100_get_sim_addr(src);
    cmd->src0_size = ggml_nbytes(src);
    cmd->dst_addr = c100_get_sim_addr(dst);
    cmd->dst_size = ggml_nbytes(dst);

    // RMSNorm params: params[0] = element count, params[1] = eps (full 32-bit)
    uint32_t total_elements = (uint32_t)ggml_nelements(src);
    uint32_t eps_bits = *((uint32_t*)&eps);
    cmd->params[0] = total_elements;   // element count (full 32-bit)
    cmd->params[1] = eps_bits;         // eps (full 32-bit float encoding)

    // Test mode: set flag if environment variable is set
    if (getenv("C100_TEST_MODE") != NULL) {
        cmd->params[3] = 0xDEADBEEF;  // Test mode flag: skip actual VE execution
    }

    cmd->status = CMD_STATUS_RUNNING;
}
static bool c100_poll_cmd_done(uint32_t* cycles) {
    int iterations = 0;
    while (iterations < C100_DEFAULT_MAX_POLL_ITERATIONS) {
        uint32_t status = c100_llama_read_cmd_status();
        if (status == CMD_STATUS_DONE) {
            if (cycles) {
                LlamaResult result;
                if (c100_llama_read_result(&result)) {
                    *cycles = result.cycles;
                }
            }
            return true;
        }
        if (status == CMD_STATUS_ERROR) {
            return false;
        }
        usleep(C100_DEFAULT_POLL_INTERVAL_US);
        iterations++;
    }
    return false;  // Timeout
}

// ============================================================================
// Backend Context (using definition from ggml-c100-impl.h)
// ============================================================================

static ggml_backend_c100_context* c100_context_create(void) {
    ggml_backend_c100_context* ctx = calloc(1, sizeof(*ctx));
    if (ctx) {
        ctx->use_polling = 1;
        ctx->poll_interval_us = C100_DEFAULT_POLL_INTERVAL_US;
        ctx->max_poll_iterations = C100_DEFAULT_MAX_POLL_ITERATIONS;
    }
    return ctx;
}

static void c100_context_free(ggml_backend_c100_context* ctx) {
    if (ctx) {
        free(ctx);
    }
}

// ============================================================================
// Buffer Interface
// ============================================================================

static void c100_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer && buffer->context) {
        // Note: Memory is managed by simulator, just free the context struct
        free(buffer->context);
    }
}

static void * c100_buffer_get_base(ggml_backend_buffer_t buffer) {
    c100_buffer_context_t* ctx = (c100_buffer_context_t*)buffer->context;
    return ctx ? ctx->host_ptr : NULL;
}

static void c100_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    (void)buffer;
}

static void c100_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);
    (void)buffer;
}

static const struct ggml_backend_buffer_i c100_buffer_i = {
    /* .free_buffer     = */ c100_buffer_free,
    /* .get_base        = */ c100_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ c100_buffer_set_tensor,
    /* .get_tensor      = */ c100_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ NULL,
    /* .reset           = */ NULL,
};

// ============================================================================
// Buffer Type Interface
// ============================================================================

// Default buffer type (GLOBAL)
static const char* c100_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return GGML_C100_NAME;
}

static ggml_backend_buffer_t c100_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    (void)buft;

    // Add alignment padding to buffer size to account for initial offset alignment
    size_t alignment = 64;
    size_t padded_size = size + alignment - 1;

    // Allocate in simulator's global memory
    uint64_t sim_addr = c100_llama_alloc_tensor(padded_size);
    if (sim_addr == 0) {
        fprintf(stderr, "[ERROR] C100: Failed to allocate tensor in simulator memory\n");
        return NULL;
    }

    // Get host pointer for direct access
    void* host_ptr = c100_llama_get_host_ptr(sim_addr);
    if (!host_ptr) {
        fprintf(stderr, "[ERROR] C100: Failed to get host pointer for simulator address 0x%lx\n", sim_addr);
        return NULL;
    }

    // Create buffer context
    c100_buffer_context_t* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->sim_addr = sim_addr;
    ctx->host_ptr = host_ptr;
    ctx->size = padded_size;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, c100_buffer_i, ctx, padded_size);
    return buffer;
}

static size_t c100_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64;  // 64-byte alignment for cache line optimization
}

static size_t c100_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 512 * 1024 * 1024;  // 512MB max
}

static size_t c100_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

static bool c100_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true;
}

// ============================================================================
// GLOBAL Buffer Type Interface
// ============================================================================

static const char* c100_global_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "C100_GLOBAL";
}

static ggml_backend_buffer_t c100_global_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    (void)buft;

    size_t alignment = 64;
    size_t padded_size = size + alignment - 1;

    // Allocate in simulator's GLOBAL memory
    uint64_t sim_addr = c100_llama_alloc_global(padded_size);
    if (sim_addr == 0) {
        fprintf(stderr, "[ERROR] C100: Failed to allocate tensor in simulator GLOBAL memory\n");
        return NULL;
    }

    void* host_ptr = c100_llama_get_host_ptr(sim_addr);
    if (!host_ptr) {
        fprintf(stderr, "[ERROR] C100: Failed to get host pointer for simulator address 0x%lx\n", sim_addr);
        return NULL;
    }

    c100_buffer_context_t* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->sim_addr = sim_addr;
    ctx->host_ptr = host_ptr;
    ctx->size = padded_size;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, c100_buffer_i, ctx, padded_size);
    return buffer;
}

static size_t c100_global_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64;
}

static size_t c100_global_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 512 * 1024 * 1024;  // 512MB max
}

static size_t c100_global_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

// ============================================================================
// LOCAL Buffer Type Interface
// ============================================================================

static const char* c100_local_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "C100_LOCAL";
}

static ggml_backend_buffer_t c100_local_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    (void)buft;

    size_t alignment = 64;
    size_t padded_size = size + alignment - 1;

    // Allocate in simulator's LOCAL memory
    uint64_t sim_addr = c100_llama_alloc_local(padded_size);
    if (sim_addr == 0) {
        fprintf(stderr, "[ERROR] C100: Failed to allocate tensor in simulator LOCAL memory\n");
        return NULL;
    }

    void* host_ptr = c100_llama_get_host_ptr(sim_addr);
    if (!host_ptr) {
        fprintf(stderr, "[ERROR] C100: Failed to get host pointer for simulator address 0x%lx\n", sim_addr);
        return NULL;
    }

    c100_buffer_context_t* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->sim_addr = sim_addr;
    ctx->host_ptr = host_ptr;
    ctx->size = padded_size;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, c100_buffer_i, ctx, padded_size);
    return buffer;
}

static size_t c100_local_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64;
}

static size_t c100_local_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 16 * 1024 * 1024;  // 16MB max (LOCAL_MEM size)
}

static size_t c100_local_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor* tensor) {
    (void)buft;
    return ggml_nbytes(tensor);
}

// ============================================================================
// Backend Interface
// ============================================================================

static const char* c100_backend_get_name(ggml_backend_t backend) {
    (void)backend;
    return GGML_C100_NAME;
}

static void c100_backend_free(ggml_backend_t backend) {
    if (backend) {
        if (backend->context) {
            c100_context_free(backend->context);
        }
        free(backend);
    }
}

static ggml_backend_buffer_type_t c100_backend_get_default_buffer_type(ggml_backend_t backend) {
    (void)backend;
    return ggml_backend_c100_buffer_type();
}

static enum ggml_status c100_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    if (!backend || !cgraph) {
        return GGML_STATUS_FAILED;
    }

    LlamaCmdHeader cmd;
    LlamaResult result;
    (void)result;  // May be used later for cycle counting

    // Iterate through compute graph nodes
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;
        if (c100_is_view_op(node->op)) continue;

        switch (node->op) {
            case GGML_OP_SOFT_MAX: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                // Prepare and send CMD
                c100_prepare_softmax_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write SoftMax CMD\n");
                    return GGML_STATUS_FAILED;
                }

                // Poll for completion
                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: SoftMax execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_ADD: {
                struct ggml_tensor* src0 = node->src[0];
                struct ggml_tensor* src1 = node->src[1];
                if (!src0 || !src1) continue;

                // Prepare and send CMD
                c100_prepare_add_cmd(&cmd, src0, src1, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write Add CMD\n");
                    return GGML_STATUS_FAILED;
                }

                // Poll for completion
                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: Add execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_MUL: {
                struct ggml_tensor* src0 = node->src[0];
                struct ggml_tensor* src1 = node->src[1];
                if (!src0 || !src1) continue;

                c100_prepare_mul_cmd(&cmd, src0, src1, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write Mul CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: Mul execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_RMS_NORM: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                float eps = 1e-5f;
                if (GGML_MAX_OP_PARAMS >= sizeof(float)) {
                    eps = *(float*)node->op_params;
                }

                c100_prepare_rms_norm_cmd(&cmd, src0, node, eps);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write RMSNorm CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: RMSNorm execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_UNARY: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                enum ggml_unary_op uop = ggml_get_unary_op(node);
                if (uop == GGML_UNARY_OP_SILU) {
                    c100_prepare_silu_cmd(&cmd, src0, node);

                    if (!c100_llama_write_cmd(&cmd)) {
                        fprintf(stderr, "[ERROR] C100: Failed to write SiLU CMD\n");
                        return GGML_STATUS_FAILED;
                    }

                    if (!c100_poll_cmd_done(NULL)) {
                        fprintf(stderr, "[ERROR] C100: SiLU execution failed or timeout\n");
                        return GGML_STATUS_FAILED;
                    }
                } else {
                    fprintf(stderr, "[WARN] C100: Unsupported unary op: %d\n", uop);
                }
                break;
            }
            case GGML_OP_ROPE: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                int n_dims = 128;
                int mode = 0;
                // params[0] = n_past, params[1] = n_dims, params[2] = mode
                if (GGML_MAX_OP_PARAMS >= 3 * sizeof(int32_t)) {
                    n_dims = ((int32_t*)node->op_params)[1];
                    mode = ((int32_t*)node->op_params)[2];
                }

                c100_prepare_rope_cmd(&cmd, src0, src0, n_dims, mode);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write RoPE CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: RoPE execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            // CE (Copy Engine) operators
            case GGML_OP_CPY: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_cpy_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write CPY CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: CPY execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_TRANSPOSE: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_transpose_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write TRANSPOSE CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: TRANSPOSE execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_PERMUTE: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_permute_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write PERMUTE CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: PERMUTE execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_GET_ROWS: {
                struct ggml_tensor* src0 = node->src[0];
                struct ggml_tensor* src1 = node->src[1];
                if (!src0 || !src1) continue;

                c100_prepare_get_rows_cmd(&cmd, src0, src1, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write GET_ROWS CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: GET_ROWS execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_RESHAPE: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_reshape_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write RESHAPE CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: RESHAPE execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_VIEW: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_view_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write VIEW CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: VIEW execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            case GGML_OP_CONT: {
                struct ggml_tensor* src0 = node->src[0];
                if (!src0) continue;

                c100_prepare_cont_cmd(&cmd, src0, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write CONT CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: CONT execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            // TE (Tensor Engine) operators
            case GGML_OP_MUL_MAT: {
                struct ggml_tensor* src0 = node->src[0];
                struct ggml_tensor* src1 = node->src[1];
                if (!src0 || !src1) continue;

                c100_prepare_mul_mat_cmd(&cmd, src0, src1, node);

                if (!c100_llama_write_cmd(&cmd)) {
                    fprintf(stderr, "[ERROR] C100: Failed to write MUL_MAT CMD\n");
                    return GGML_STATUS_FAILED;
                }

                if (!c100_poll_cmd_done(NULL)) {
                    fprintf(stderr, "[ERROR] C100: MUL_MAT execution failed or timeout\n");
                    return GGML_STATUS_FAILED;
                }

                break;
            }
            default:
                // Unsupported operation - skip
                fprintf(stderr, "[WARN] C100: Unsupported op: %s\n", ggml_op_name(node->op));
                break;
        }
    }

    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Device Interface
// ============================================================================

static const char* c100_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_C100_NAME;
}

static const char* c100_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "C100 Simulator Backend";
}

static void c100_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    (void)dev;
    *free = 512 * 1024 * 1024;  // 512MB
    *total = 512 * 1024 * 1024;
}

static enum ggml_backend_dev_type c100_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void c100_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    props->name = c100_device_get_name(dev);
    props->description = c100_device_get_description(dev);
    props->type = c100_device_get_type(dev);
    c100_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = (struct ggml_backend_dev_caps){
        .async = false,
        .host_buffer = false,
        .buffer_from_host_ptr = false,
        .events = false,
    };
}

static ggml_backend_t c100_device_init_backend(ggml_backend_dev_t dev, const char* params) {
    (void)params;
    return ggml_backend_c100_init();
}

static ggml_backend_buffer_type_t c100_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return ggml_backend_c100_buffer_type();
}

static bool c100_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    switch (op->op) {
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return true;
        default:
            return false;
    }
}

static bool c100_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return buft == ggml_backend_c100_buffer_type() ||
           buft == ggml_backend_c100_global_buffer_type() ||
           buft == ggml_backend_c100_local_buffer_type();
}

// ============================================================================
// Interface Tables
// ============================================================================

static const struct ggml_backend_buffer_type_i c100_buffer_type_i = {
    /* .get_name      = */ c100_buffer_type_get_name,
    /* .alloc_buffer  = */ c100_buffer_type_alloc_buffer,
    /* .get_alignment = */ c100_buffer_type_get_alignment,
    /* .get_max_size  = */ c100_buffer_type_get_max_size,
    /* .get_alloc_size= */ c100_buffer_type_get_alloc_size,
    /* .is_host       = */ c100_buffer_type_is_host,
};

static const struct ggml_backend_buffer_type_i c100_global_buffer_type_i = {
    /* .get_name      = */ c100_global_buffer_type_get_name,
    /* .alloc_buffer  = */ c100_global_buffer_type_alloc_buffer,
    /* .get_alignment = */ c100_global_buffer_type_get_alignment,
    /* .get_max_size  = */ c100_global_buffer_type_get_max_size,
    /* .get_alloc_size= */ c100_global_buffer_type_get_alloc_size,
    /* .is_host       = */ c100_buffer_type_is_host,
};

static const struct ggml_backend_buffer_type_i c100_local_buffer_type_i = {
    /* .get_name      = */ c100_local_buffer_type_get_name,
    /* .alloc_buffer  = */ c100_local_buffer_type_alloc_buffer,
    /* .get_alignment = */ c100_local_buffer_type_get_alignment,
    /* .get_max_size  = */ c100_local_buffer_type_get_max_size,
    /* .get_alloc_size= */ c100_local_buffer_type_get_alloc_size,
    /* .is_host       = */ c100_buffer_type_is_host,
};

static const struct ggml_backend_i c100_backend_i = {
    /* .get_name             = */ c100_backend_get_name,
    /* .free                 = */ c100_backend_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ c100_backend_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static const struct ggml_backend_device_i c100_device_i = {
    /* .get_name             = */ c100_device_get_name,
    /* .get_description      = */ c100_device_get_description,
    /* .get_memory           = */ c100_device_get_memory,
    /* .get_type             = */ c100_device_get_type,
    /* .get_props            = */ c100_device_get_props,
    /* .init_backend         = */ c100_device_init_backend,
    /* .get_buffer_type      = */ c100_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ c100_device_supports_op,
    /* .supports_buft        = */ c100_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================================
// Registration Interface
// ============================================================================

static const char* c100_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return GGML_C100_NAME;
}

static size_t c100_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return ggml_backend_c100_is_available() ? 1 : 0;
}

static ggml_backend_dev_t c100_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    if (index > 0) return NULL;

    static struct ggml_backend_device c100_device = {
        /* .iface   = */ c100_device_i,
        /* .reg     = */ NULL,  // Set during registration
        /* .context = */ NULL,
    };

    return &c100_device;
}

static const struct ggml_backend_reg_i c100_reg_i = {
    /* .get_name         = */ c100_reg_get_name,
    /* .get_device_count = */ c100_reg_get_device_count,
    /* .get_device       = */ c100_reg_get_device,
    /* .get_proc_address = */ NULL,
};

// ============================================================================
// Public API Implementation
// ============================================================================

bool ggml_backend_c100_is_available(void) {
    return get_simulator_instance() != NULL;
}

size_t ggml_backend_c100_get_device_count(void) {
    return ggml_backend_c100_is_available() ? 1 : 0;
}

ggml_backend_reg_t ggml_backend_c100_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ c100_reg_i,
        /* .context     = */ NULL,
    };

    return &reg;
}

static ggml_guid_t c100_guid(void) {
    static ggml_guid guid = { 0x52, 0x31, 0x30, 0x30, 0x53, 0x49, 0x4d, 0x55,
                              0x4c, 0x41, 0x54, 0x4f, 0x52, 0x00, 0x00, 0x01 };
    return &guid;
}

ggml_backend_buffer_type_t ggml_backend_c100_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ c100_buffer_type_i,
        /* .device  = */ NULL,  // Set during device init
        /* .context = */ NULL,
    };
    return &buft;
}

ggml_backend_buffer_type_t ggml_backend_c100_global_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ c100_global_buffer_type_i,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &buft;
}

ggml_backend_buffer_type_t ggml_backend_c100_local_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ c100_local_buffer_type_i,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &buft;
}

ggml_backend_t ggml_backend_c100_init(void) {
    if (!ggml_backend_c100_is_available()) {
        fprintf(stderr, "[ERROR] C100: Simulator not available\n");
        return NULL;
    }

    struct ggml_backend_c100_context* ctx = c100_context_create();
    if (!ctx) {
        return NULL;
    }

    ggml_backend_t backend = calloc(1, sizeof(*backend));
    if (!backend) {
        c100_context_free(ctx);
        return NULL;
    }

    *backend = (struct ggml_backend){
        /* .guid    = */ c100_guid(),
        /* .iface   = */ c100_backend_i,
        /* .device  = */ c100_reg_get_device(ggml_backend_c100_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

// Dynamic loading support
GGML_BACKEND_DL_IMPL(ggml_backend_c100_reg)
