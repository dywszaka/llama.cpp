#pragma once

#include "llama-batch.h"
#include "llama-graph.h"

#include "ggml-backend.h"
#include "ggml.h"

struct llama_decode_attn_dump_state;

bool llama_decode_attn_dump_enabled();

bool llama_decode_attn_dump_pending();

bool llama_decode_attn_dump_ubatch_is_first_decode(const llama_ubatch & ubatch, llm_graph_type gtype);

void llama_decode_attn_dump_log_enabled_once();

void llama_decode_attn_dump_mark_softmax(
        const llama_ubatch & ubatch,
        llm_graph_type      gtype,
        ggml_tensor       * tensor,
        int                 il);

llama_decode_attn_dump_state * llama_decode_attn_dump_prepare(
        const llama_ubatch & ubatch,
        llm_graph_type      gtype,
        ggml_backend_sched_eval_callback user_cb,
        void              * user_data);

ggml_backend_sched_eval_callback llama_decode_attn_dump_eval_callback();

void * llama_decode_attn_dump_eval_user_data(llama_decode_attn_dump_state * state);

void llama_decode_attn_dump_finish(llama_decode_attn_dump_state * state);
