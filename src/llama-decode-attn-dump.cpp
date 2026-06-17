#include "llama-decode-attn-dump.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

static constexpr const char * LLAMA_DECODE_ATTN_DUMP_ENV = "LLAMA_DUMP_FIRST_DECODE_ATTN_SOFTMAX";
static constexpr const char * LLAMA_DECODE_ATTN_DUMP_DIR = "experiments/first-decode-attn-softmax-dump";
static constexpr const char * LLAMA_DECODE_ATTN_SOFTMAX_NAME = "first_decode_attn_softmax";

struct llama_decode_attn_dump_state {
    ggml_backend_sched_eval_callback user_cb   = nullptr;
    void *                           user_data = nullptr;
    ggml_tensor *                    last_ask_tensor = nullptr;
    bool                             last_user_need  = false;
};

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0;
}

static bool & dump_done() {
    static bool value = false;
    return value;
}

static int & dump_layer() {
    static int value = -1;
    return value;
}

static ggml_tensor *& dump_input_tensor() {
    static ggml_tensor * value = nullptr;
    return value;
}

static std::vector<unsigned char> & dump_input_data() {
    static std::vector<unsigned char> value;
    return value;
}

static bool & dump_input_captured() {
    static bool value = false;
    return value;
}

static bool create_dir_if_needed(const char * path) {
#if defined(_WIN32)
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return true;
    }
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return true;
    }
#endif
    LLAMA_LOG_ERROR("%s: failed to create dump directory '%s': %s\n", __func__, path, std::strerror(errno));
    return false;
}

static bool write_file(const std::string & path, const void * data, size_t size) {
    FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path.c_str(), std::strerror(errno));
        return false;
    }

    const size_t nwritten = size == 0 ? 0 : std::fwrite(data, 1, size, fp);
    const bool ok = nwritten == size && std::fclose(fp) == 0;
    if (!ok) {
        LLAMA_LOG_ERROR("%s: failed to write '%s'\n", __func__, path.c_str());
    }
    return ok;
}

static std::vector<unsigned char> copy_tensor_to_bytes(const ggml_tensor * t) {
    std::vector<unsigned char> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    return data;
}

static bool copy_tensor_to_file(const ggml_tensor * t, const std::string & path) {
    const std::vector<unsigned char> data = copy_tensor_to_bytes(t);
    return write_file(path, data.data(), data.size());
}

static void write_metadata(const ggml_tensor * input, const ggml_tensor * output) {
    const std::string path = std::string(LLAMA_DECODE_ATTN_DUMP_DIR) + "/metadata.json";
    FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path.c_str(), std::strerror(errno));
        return;
    }

    std::fprintf(fp,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"dump\": \"first-decode-attn-softmax\",\n"
            "  \"trigger_env\": \"%s\",\n"
            "  \"directory\": \"%s\",\n"
            "  \"attention_layer\": %d,\n"
            "  \"tensors\": [\n",
            LLAMA_DECODE_ATTN_DUMP_ENV,
            LLAMA_DECODE_ATTN_DUMP_DIR,
            dump_layer());

    const ggml_tensor * tensors[2] = { input, output };
    const char * ids[2] = { "input", "output" };
    const char * paths[2] = { "attn_softmax_input.bin", "attn_softmax_output.bin" };
    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * t = tensors[i];
        std::fprintf(fp,
                "    {\n"
                "      \"id\": \"%s\",\n"
                "      \"path\": \"%s\",\n"
                "      \"tensor_name\": \"%s\",\n"
                "      \"dtype\": \"%s\",\n"
                "      \"shape\": [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "],\n"
                "      \"strides_bytes\": [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "],\n"
                "      \"nbytes\": %zu\n"
                "    }%s\n",
                ids[i],
                paths[i],
                ggml_get_name(t),
                ggml_type_name(t->type),
                t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3],
                ggml_nbytes(t),
                i == 0 ? "," : "");
    }

    std::fprintf(fp, "  ]\n}\n");
    if (std::fclose(fp) != 0) {
        LLAMA_LOG_ERROR("%s: failed to close '%s'\n", __func__, path.c_str());
    }
}

static bool is_marked_softmax(const ggml_tensor * t) {
    return t != nullptr &&
        t->op == GGML_OP_SOFT_MAX &&
        std::strcmp(ggml_get_name(t), LLAMA_DECODE_ATTN_SOFTMAX_NAME) == 0;
}

static bool is_marked_softmax_input(const ggml_tensor * t) {
    return t != nullptr && t == dump_input_tensor();
}

static bool capture_input_tensor(const ggml_tensor * input) {
    dump_input_data() = copy_tensor_to_bytes(input);
    dump_input_captured() = true;
    return true;
}

static bool dump_tensor_pair(const ggml_tensor * output) {
    if (!create_dir_if_needed("experiments") || !create_dir_if_needed(LLAMA_DECODE_ATTN_DUMP_DIR)) {
        return false;
    }

    const ggml_tensor * input = output->src[0];
    if (input == nullptr) {
        LLAMA_LOG_ERROR("%s: softmax node has no input tensor\n", __func__);
        return false;
    }

    const std::string input_path = std::string(LLAMA_DECODE_ATTN_DUMP_DIR) + "/attn_softmax_input.bin";
    const std::string output_path = std::string(LLAMA_DECODE_ATTN_DUMP_DIR) + "/attn_softmax_output.bin";

    if (!dump_input_captured()) {
        LLAMA_LOG_ERROR("%s: softmax input tensor was not captured before softmax execution\n", __func__);
        return false;
    }

    if (!write_file(input_path, dump_input_data().data(), dump_input_data().size())) {
        return false;
    }
    if (!copy_tensor_to_file(output, output_path)) {
        return false;
    }

    write_metadata(input, output);

    LLAMA_LOG_INFO("%s: dumped first decode attention softmax tensors to %s\n",
            __func__, LLAMA_DECODE_ATTN_DUMP_DIR);
    return true;
}

bool llama_decode_attn_dump_enabled() {
    static const bool enabled = env_flag_enabled(LLAMA_DECODE_ATTN_DUMP_ENV);
    return enabled;
}

bool llama_decode_attn_dump_pending() {
    return llama_decode_attn_dump_enabled() && !dump_done();
}

bool llama_decode_attn_dump_ubatch_is_first_decode(const llama_ubatch & ubatch, llm_graph_type gtype) {
    if (gtype != LLM_GRAPH_TYPE_DECODER || ubatch.n_tokens != 1 || ubatch.pos == nullptr) {
        return false;
    }

    return ubatch.pos[0] > 0;
}

void llama_decode_attn_dump_log_enabled_once() {
    if (!llama_decode_attn_dump_enabled()) {
        return;
    }

    static bool logged = false;
    if (!logged) {
        LLAMA_LOG_INFO("%s: %s enabled, output directory: %s\n",
                __func__, LLAMA_DECODE_ATTN_DUMP_ENV, LLAMA_DECODE_ATTN_DUMP_DIR);
        logged = true;
    }
}

void llama_decode_attn_dump_mark_softmax(
        const llama_ubatch & ubatch,
        llm_graph_type      gtype,
        ggml_tensor       * tensor,
        int                 il) {
    if (!llama_decode_attn_dump_pending() ||
            !llama_decode_attn_dump_ubatch_is_first_decode(ubatch, gtype)) {
        return;
    }
    if (dump_layer() >= 0) {
        return;
    }

    ggml_set_name(tensor, LLAMA_DECODE_ATTN_SOFTMAX_NAME);
    dump_layer() = il;
    dump_input_tensor() = tensor->src[0];
}

llama_decode_attn_dump_state * llama_decode_attn_dump_prepare(
        const llama_ubatch & ubatch,
        llm_graph_type      gtype,
        ggml_backend_sched_eval_callback user_cb,
        void              * user_data) {
    if (!llama_decode_attn_dump_pending() ||
            !llama_decode_attn_dump_ubatch_is_first_decode(ubatch, gtype)) {
        return nullptr;
    }

    llama_decode_attn_dump_state * state = new llama_decode_attn_dump_state;
    state->user_cb = user_cb;
    state->user_data = user_data;
    return state;
}

static bool llama_decode_attn_dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    llama_decode_attn_dump_state * state = (llama_decode_attn_dump_state *) user_data;

    const bool dump_need_input = llama_decode_attn_dump_pending() && !dump_input_captured() && is_marked_softmax_input(t);
    const bool dump_need_output = llama_decode_attn_dump_pending() && is_marked_softmax(t);

    if (ask) {
        const bool user_need = state->user_cb ? state->user_cb(t, true, state->user_data) : false;
        state->last_ask_tensor = t;
        state->last_user_need = user_need;
        return user_need || dump_need_input || dump_need_output;
    }

    bool keep_going = true;
    const bool user_need = state->last_ask_tensor == t ?
        state->last_user_need :
        (state->user_cb ? state->user_cb(t, true, state->user_data) : false);
    if (user_need && state->user_cb) {
        keep_going = state->user_cb(t, false, state->user_data);
    }

    if (dump_need_input) {
        capture_input_tensor(t);
    }

    if (dump_need_output) {
        if (dump_tensor_pair(t)) {
            dump_done() = true;
        }
    }

    return keep_going;
}

ggml_backend_sched_eval_callback llama_decode_attn_dump_eval_callback() {
    return llama_decode_attn_dump_cb;
}

void * llama_decode_attn_dump_eval_user_data(llama_decode_attn_dump_state * state) {
    return state;
}

void llama_decode_attn_dump_finish(llama_decode_attn_dump_state * state) {
    delete state;
}
