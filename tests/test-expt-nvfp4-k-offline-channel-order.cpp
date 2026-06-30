#include "../src/expt/nvfp4-k-offline-channel-order.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static bool expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "%s\n", msg);
        return false;
    }
    return true;
}

static std::string temp_dir() {
    const char * base = std::getenv("TMPDIR");
    if (!base) {
        base = "/tmp";
    }
    return std::string(base) + "/llama-expt-nvfp4-k-order-test";
}

static void write_file(const std::string & path, const std::string & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    out << data;
}

static bool test_loads_records_by_layer_name() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/order.json";
    write_file(path,
            "{\n"
            "  \"algorithm\": \"nvfp4_k_channel_mean_sort\",\n"
            "  \"records\": [\n"
            "    {\"name\":\"Kcur-1\",\"sort_basis\":\"abs_mean\",\"channel_order\":[2,0,1,3]},\n"
            "    {\"name\":\"Kcur-0\",\"sort_basis\":\"abs_mean\",\"channel_order\":[3,1,0,2]}\n"
            "  ]\n"
            "}\n");

    const auto loaded = llama_expt::nvfp4_k_offline_channel_order_load(path, 2, 4);

    return expect(loaded.path == path, "path mismatch") &&
           expect(loaded.per_layer.size() == 2, "layer count mismatch") &&
           expect(loaded.per_layer[0] == std::vector<int32_t>({ 3, 1, 0, 2 }), "layer 0 order mismatch") &&
           expect(loaded.per_layer[1] == std::vector<int32_t>({ 2, 0, 1, 3 }), "layer 1 order mismatch");
}

static bool test_rejects_non_permutation() {
    const std::string dir = temp_dir();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/bad-order.json";
    write_file(path,
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"Kcur-0\",\"sort_basis\":\"abs_mean\",\"channel_order\":[0,1,1,3]}\n"
            "  ]\n"
            "}\n");

    try {
        (void) llama_expt::nvfp4_k_offline_channel_order_load(path, 1, 4);
    } catch (const std::exception &) {
        return true;
    }

    std::fprintf(stderr, "expected non-permutation order to throw\n");
    return false;
}

static bool test_gqa_indices_repeat_per_head() {
    const std::vector<int32_t> order = { 2, 0, 1, 3 };
    const std::vector<int32_t> indices = llama_expt::nvfp4_k_offline_channel_order_gqa_indices(order, 4, 2);
    const std::vector<int32_t> expected = { 2, 0, 1, 3, 6, 4, 5, 7 };
    return expect(indices == expected, "GQA flattened indices mismatch");
}

int main() {
    if (!test_loads_records_by_layer_name()) {
        return 1;
    }
    if (!test_rejects_non_permutation()) {
        return 1;
    }
    if (!test_gqa_indices_repeat_per_head()) {
        return 1;
    }

    std::puts("test-expt-nvfp4-k-offline-channel-order: ok");
    return 0;
}
