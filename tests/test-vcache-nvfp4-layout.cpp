#include <ggml.h>

#include <cstdlib>
#include <cstdio>

static bool run_case() {
    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    const int64_t n_kv_padded = 32;
    const int64_t n_head_kv = 8;
    const int64_t n_head_v = 128;
    const int64_t n_embd_v_gqa = n_head_kv * n_head_v;
    const int64_t n_tokens = 1;

    ggml_tensor * v_store = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, n_kv_padded, n_embd_v_gqa, 1);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_v_gqa * (n_kv_padded / 16), 1);
    ggml_tensor * v = ggml_view_4d(ctx, v_store,
            n_kv_padded, n_head_kv, n_head_v, 1,
            ggml_row_size(v_store->type, n_kv_padded),
            ggml_row_size(v_store->type, n_kv_padded * n_head_v),
            ggml_row_size(v_store->type, n_kv_padded * n_embd_v_gqa),
            0);
    ggml_tensor * scale = ggml_view_4d(ctx, v_scale,
            n_kv_padded / 16, n_head_kv, n_head_v, 1,
            (int64_t) (n_kv_padded / 16) * sizeof(float),
            (int64_t) (n_kv_padded / 16) * n_head_v * sizeof(float),
            (int64_t) (n_kv_padded / 16) * n_embd_v_gqa * sizeof(float),
            0);
    ggml_tensor_set_nvfp4_scale(v, scale);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head_v, n_head_kv, n_tokens);
    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2], 1);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);

    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_head_v, n_head_kv, n_kv_padded, 1);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor_set_nvfp4_scale(v, scale);

    const bool legacy_v_trans = v->nb[1] > v->nb[2];
    if (!legacy_v_trans) {
        v = ggml_cont(ctx, ggml_transpose(ctx, v));
    }

    const bool can_mul = (v->ne[0] == kq->ne[0]) && (kq->ne[2] % v->ne[2] == 0) && (kq->ne[3] % v->ne[3] == 0);

    if (!can_mul) {
        std::fprintf(stderr, "expected permuted NVFP4 V-cache layout to stay mul_mat-compatible\n");
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);
    return true;
}

static bool run_recent_f16_metadata_case() {
    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 128, 32);
    ggml_tensor * recent_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 32);
    ggml_tensor * recent_active = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32);
    ggml_tensor * recent_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32);

    ggml_tensor_set_nvfp4_kcache_recent_f16(k, recent_f16, recent_active, recent_pos);

    if (ggml_tensor_get_nvfp4_kcache_recent_f16(k) != recent_f16) {
        std::fprintf(stderr, "recent F16 shadow metadata did not round-trip\n");
        ggml_free(ctx);
        return false;
    }
    if (ggml_tensor_get_nvfp4_kcache_recent_f16_active(k) != recent_active) {
        std::fprintf(stderr, "recent F16 active metadata did not round-trip\n");
        ggml_free(ctx);
        return false;
    }
    if (ggml_tensor_get_nvfp4_kcache_recent_f16_pos(k) != recent_pos) {
        std::fprintf(stderr, "recent F16 position metadata did not round-trip\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * view = ggml_view_tensor(ctx, k);
    ggml_tensor_set_nvfp4_kcache_recent_f16(view, recent_f16, recent_active, recent_pos);
    if (ggml_tensor_get_nvfp4_kcache_recent_f16(view) != recent_f16 ||
            ggml_tensor_get_nvfp4_kcache_recent_f16_active(view) != recent_active ||
            ggml_tensor_get_nvfp4_kcache_recent_f16_pos(view) != recent_pos) {
        std::fprintf(stderr, "recent F16 metadata did not round-trip on view tensor\n");
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);
    return true;
}

int main() {
    if (!run_case()) {
        return 1;
    }
    if (!run_recent_f16_metadata_case()) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-layout: ok");
    return 0;
}
