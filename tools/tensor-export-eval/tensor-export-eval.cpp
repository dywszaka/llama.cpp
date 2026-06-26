#include "../../src/expt/tensor-export-eval.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

static void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --manifest path/to/manifest.json [--global-scale N]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string manifest_path;
    float global_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--global-scale" && i + 1 < argc) {
            global_scale = std::strtof(argv[++i], nullptr);
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
        const llama_expt::eval_report report = llama_expt::evaluate_manifest(manifest_path, global_scale);
        std::printf("%s\n", llama_expt::format_eval_report_json(report).c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-tensor-export-eval: %s\n", e.what());
        return 1;
    }

    return 0;
}
