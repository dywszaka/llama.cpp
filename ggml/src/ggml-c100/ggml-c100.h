/**
 * @file ggml-c100.h
 * @brief C100 GGML Backend interface
 *
 * This backend enables llama.cpp to offload compute operations
 * to the C100 simulator (SU + VE architecture).
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C100 backend initialization
 *
 * Initializes the C100 backend and registers it with GGML.
 * This should be called before using any C100 backend functions.
 *
 * @return ggml_backend_reg_t Registration handle, NULL on failure
 */
GGML_API ggml_backend_reg_t ggml_backend_c100_reg(void);

/**
 * @brief Check if C100 backend is available
 *
 * @return true if C100 backend is available, false otherwise
 */
GGML_API bool ggml_backend_c100_is_available(void);

/**
 * @brief Get C100 backend device count
 *
 * @return Number of C100 devices (currently always 1)
 */
GGML_API size_t ggml_backend_c100_get_device_count(void);

/**
 * @brief Get C100 buffer type
 *
 * Returns the buffer type for C100's unified memory.
 * Tensors allocated with this type will be placed in C100's
 * GLOBAL memory region (0x21000000+ by default).
 *
 * @return ggml_backend_buffer_type_t Buffer type handle
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_c100_buffer_type(void);

/**
 * @brief Get C100 GLOBAL buffer type
 *
 * Returns the buffer type for C100's GLOBAL memory (0x21000000+).
 * Use this for large tensors that need to be accessible across layers.
 *
 * @return ggml_backend_buffer_type_t Buffer type handle
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_c100_global_buffer_type(void);

/**
 * @brief Get C100 LOCAL buffer type
 *
 * Returns the buffer type for C100's LOCAL memory (0x10000000+).
 * Use this for temporary tensors within a single layer.
 *
 * @return ggml_backend_buffer_type_t Buffer type handle
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_c100_local_buffer_type(void);

/**
 * @brief Initialize C100 backend
 *
 * Creates a new C100 backend instance.
 *
 * @return ggml_backend_t Backend handle, NULL on failure
 */
GGML_API ggml_backend_t ggml_backend_c100_init(void);

#ifdef __cplusplus
}
#endif
