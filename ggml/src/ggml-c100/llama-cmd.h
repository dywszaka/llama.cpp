#ifndef LLAMA_CMD_H
#define LLAMA_CMD_H

#include <stdint.h>

#include "llama_cmd_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef llama_cmd_header_t LlamaCmdHeader;
typedef llama_result_t LlamaResult;

#define CMD_STATUS_IDLE     LLAMA_STATUS_IDLE
#define CMD_STATUS_RUNNING  LLAMA_STATUS_RUNNING
#define CMD_STATUS_DONE     LLAMA_STATUS_DONE
#define CMD_STATUS_ERROR    LLAMA_STATUS_ERROR

#define CMD_STATUS_OFFSET LLAMA_CMD_STATUS_OFFSET

#define CMD_ID_SOFTMAX          LLAMA_CMD_ID_SOFTMAX
#define CMD_ID_ADD              LLAMA_CMD_ID_ADD
#define CMD_ID_MUL              LLAMA_CMD_ID_MUL
#define CMD_ID_RMS_NORM         LLAMA_CMD_ID_RMS_NORM
#define CMD_ID_SILU             LLAMA_CMD_ID_SILU
#define CMD_ID_ROPE             LLAMA_CMD_ID_ROPE
#define CMD_ID_EXT_PARAM_DEBUG  LLAMA_CMD_ID_EXT_PARAM_DEBUG

#define CMD_ID_MUL_MAT LLAMA_CMD_ID_MUL_MAT

#define CMD_ID_GET_ROWS   LLAMA_CMD_ID_GET_ROWS
#define CMD_ID_RESHAPE    LLAMA_CMD_ID_RESHAPE
#define CMD_ID_VIEW       LLAMA_CMD_ID_VIEW
#define CMD_ID_CPY        LLAMA_CMD_ID_CPY
#define CMD_ID_TRANSPOSE  LLAMA_CMD_ID_TRANSPOSE
#define CMD_ID_PERMUTE    LLAMA_CMD_ID_PERMUTE
#define CMD_ID_CONT       LLAMA_CMD_ID_CONT

#define CMD_SOFTMAX_FLAG_HAS_MASK   LLAMA_SOFTMAX_FLAG_HAS_MASK
#define CMD_FLAG_EXT_PARAM          LLAMA_CMD_FLAG_EXT_PARAM

#define CMD_MAGIC LLAMA_CMD_MAGIC
#define RESULT_MAGIC LLAMA_RESULT_MAGIC

#ifdef __cplusplus
}
#endif

#endif  // LLAMA_CMD_H
