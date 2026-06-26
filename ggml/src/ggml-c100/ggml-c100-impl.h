/**
 * @file ggml-c100-impl.h
 * @brief C100 GGML Backend internal implementation header
 */

#ifndef GGML_C100_IMPL_H
#define GGML_C100_IMPL_H

#include "ggml-c100.h"
#include "llama-cmd.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"  // For struct ggml_backend definition

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// C100 Backend Context
// ============================================================================

/**
 * @brief C100 backend context structure
 *
 * Contains all state needed for C100 backend operations.
 * Note: Simulator pointer is obtained via global singleton,
 * not stored here (to avoid C/C++ pointer complexity).
 */
typedef struct ggml_backend_c100_context {
    ggml_backend_t backend;           // Parent backend reference
    ggml_backend_buffer_type_t buft;  // Buffer type for this backend

    // Memory management
    void* global_mem_base;            // C100 GLOBAL memory base (0x20000000)
    size_t global_mem_size;           // Total GLOBAL memory size (4GB)
    size_t allocated_offset;          // Current allocation offset

    // Command tracking
    uint32_t last_cmd_status;         // Last read CMD status
    uint64_t cmd_count;               // Total commands processed

    // Configuration
    int use_polling;                  // 1 = poll for completion, 0 = async
    int poll_interval_us;             // Polling interval in microseconds
    int max_poll_iterations;          // Maximum poll iterations before timeout
} ggml_backend_c100_context;

// ============================================================================
// Buffer Management
// ============================================================================

/**
 * @brief C100 buffer structure
 *
 * Wraps a memory region in C100 GLOBAL memory.
 */
typedef struct ggml_backend_c100_buffer {
    ggml_backend_buffer_t base;  // Base buffer interface
    void* data;                  // Pointer to GLOBAL memory
    size_t size;                 // Buffer size
    uint64_t c100_phys_addr;     // C100 physical address (0x20000000 + offset)
} ggml_backend_c100_buffer;

// C100 GLOBAL memory base address
#define C100_GLOBAL_MEM_BASE  0x20000000ULL
#define C100_GLOBAL_MEM_SIZE  0x100000000ULL  // 4GB

// CMD/RESULT region addresses
#define C100_CMD_BASE       0x20FF0000ULL
#define C100_CMD_SIZE       0x00010000ULL  // 64KB
#define C100_RESULT_BASE    0x20FF1000ULL
#define C100_RESULT_SIZE    0x00010000ULL  // 64KB

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Get C100 backend context from backend
 */
static inline ggml_backend_c100_context* ggml_c100_ctx(ggml_backend_t backend) {
    return (ggml_backend_c100_context*)backend->context;
}

/**
 * @brief Convert host pointer to C100 physical address
 */
static inline uint64_t ggml_c100_host_to_phys(void* host_ptr, void* global_base) {
    return C100_GLOBAL_MEM_BASE + ((uint8_t*)host_ptr - (uint8_t*)global_base);
}

/**
 * @brief Convert C100 physical address to host pointer
 */
static inline void* ggml_c100_phys_to_host(uint64_t phys_addr, void* global_base) {
    return (void*)((uint8_t*)global_base + (phys_addr - C100_GLOBAL_MEM_BASE));
}

#ifdef __cplusplus
}
#endif

#endif // GGML_C100_IMPL_H
