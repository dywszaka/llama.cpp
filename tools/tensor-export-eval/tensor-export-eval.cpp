#include "../../src/expt/tensor-export-eval.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s --manifest path/to/manifest.json [--global-scale N] "
            "[--algorithm nvfp4_ref|attention_replay|attention_replay_nvfp4_outlier|attention_replay_fp8_e4m3_e8m0|nvfp4_k_channel_sort|nvfp4_k_channel_mean_sort] "
            "[--k-channel-sort] [--k-channel-mean-sort]\n",
            argv0);
}

int main(int argc, char ** argv) {
    std::string manifest_path;
    std::string algorithm = "nvfp4_ref";
    float global_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--global-scale" && i + 1 < argc) {
            global_scale = std::strtof(argv[++i], nullptr);
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (arg == "--k-channel-sort") {
            algorithm = "nvfp4_k_channel_sort";
        } else if (arg == "--k-channel-mean-sort") {
            algorithm = "nvfp4_k_channel_mean_sort";
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (manifest_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        if (algorithm == "nvfp4_ref") {
            const llama_expt::eval_report report = llama_expt::evaluate_manifest(manifest_path, global_scale);
            std::printf("%s\n", llama_expt::format_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay") {
            const llama_expt::attention_replay_eval_report report =
                llama_expt::evaluate_manifest_attention_replay(manifest_path);
            std::printf("%s\n", llama_expt::format_attention_replay_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay_nvfp4_outlier") {
            const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
                llama_expt::evaluate_manifest_attention_replay_nvfp4_outlier(manifest_path);
            std::printf("%s\n", llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay_fp8_e4m3_e8m0") {
            const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
                llama_expt::evaluate_manifest_attention_replay_fp8_e4m3_e8m0(manifest_path);
            std::printf("%s\n", llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report).c_str());
        } else if (algorithm == "nvfp4_k_channel_sort") {
            const llama_expt::k_channel_sort_eval_report report =
                llama_expt::evaluate_manifest_k_channel_sort(
                        manifest_path,
                        llama_expt::k_channel_sort_basis::FIRST_ROW_ABS,
                        global_scale);
            std::printf("%s\n", llama_expt::format_k_channel_sort_eval_report_json(report).c_str());
        } else if (algorithm == "nvfp4_k_channel_mean_sort") {
            const llama_expt::k_channel_sort_eval_report report =
                llama_expt::evaluate_manifest_k_channel_sort(
                        manifest_path,
                        llama_expt::k_channel_sort_basis::ABS_MEAN,
                        global_scale);
            std::printf("%s\n", llama_expt::format_k_channel_sort_eval_report_json(report).c_str());
        } else {
            std::fprintf(stderr, "unknown algorithm: %s\n", algorithm.c_str());
            print_usage(argv[0]);
            return 2;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-tensor-export-eval: %s\n", e.what());
        return 1;
    }

    return 0;
}
