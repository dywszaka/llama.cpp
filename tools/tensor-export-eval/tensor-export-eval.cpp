#include "../../src/expt/tensor-export-eval.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s --manifest path/to/manifest.json [--global-scale N] [--csv path/to/metrics.csv] "
            "[--algorithm nvfp4_ref|attention_replay|attention_replay_nvfp4_outlier|attention_replay_fp8_e4m3_e8m0]\n",
            argv0);
}

static std::string csv_escape(const std::string & value) {
    bool needs_quote = false;
    for (char ch : value) {
        if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        return value;
    }

    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += "\"";
    return out;
}

static std::string csv_number(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}

static void append_csv_row(
        std::ofstream & out,
        const std::string & algorithm,
        const std::string & record,
        const std::string & target,
        double mse,
        double nmse,
        double max_abs_err,
        const std::string & kld) {
    out << csv_escape(algorithm) << ','
        << csv_escape(record) << ','
        << csv_escape(target) << ','
        << csv_number(mse) << ','
        << csv_number(nmse) << ','
        << csv_number(max_abs_err) << ','
        << kld
        << '\n';
}

static void append_eval_csv(
        const std::string & csv_path,
        const std::string & algorithm,
        const llama_expt::eval_report & report) {
    if (csv_path.empty()) {
        return;
    }

    const bool write_header = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open csv output '" + csv_path + "': " + std::strerror(errno));
    }
    if (write_header) {
        out << "algorithm,record,target,mse,nmse,max_abs_err,kld\n";
    }
    for (const llama_expt::eval_record_report & rr : report.records) {
        append_csv_row(out, algorithm, rr.record.name, rr.record.kind,
                rr.metrics.mse, rr.nmse, rr.max_abs_err, "");
    }
}

static void append_attention_replay_csv(
        const std::string & csv_path,
        const std::string & algorithm,
        const llama_expt::attention_replay_eval_report & report) {
    if (csv_path.empty()) {
        return;
    }

    const bool write_header = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open csv output '" + csv_path + "': " + std::strerror(errno));
    }
    if (write_header) {
        out << "algorithm,record,target,mse,nmse,max_abs_err,kld\n";
    }
    for (const llama_expt::attention_replay_report & rr : report.records) {
        append_csv_row(out, algorithm, rr.kq_record.name, "kq",
                rr.kq_metrics.mse, rr.kq_nmse, rr.max_abs_err_kq, "");
        append_csv_row(out, algorithm, rr.softmax_record.name, "softmax",
                rr.softmax_metrics.mse, rr.softmax_nmse, rr.max_abs_err_softmax, "");
    }
}

static void append_attention_quant_round_csv(
        const std::string & csv_path,
        const llama_expt::attention_replay_nvfp4_outlier_eval_report & report) {
    if (csv_path.empty()) {
        return;
    }

    const bool write_header = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open csv output '" + csv_path + "': " + std::strerror(errno));
    }
    if (write_header) {
        out << "algorithm,record,target,mse,nmse,max_abs_err,kld\n";
    }
    for (const llama_expt::attention_replay_nvfp4_outlier_report & rr : report.records) {
        append_csv_row(out, report.algorithm, rr.kq_record.name, "kq",
                rr.kq_metrics.mse, rr.kq_nmse, rr.max_abs_err_kq, "");
        append_csv_row(out, report.algorithm, rr.softmax_record.name, "softmax",
                rr.softmax_metrics.mse, rr.softmax_nmse, rr.max_abs_err_softmax,
                csv_number(rr.softmax_kld));
    }
}

int main(int argc, char ** argv) {
    std::string manifest_path;
    std::string algorithm = "nvfp4_ref";
    std::string csv_path;
    float global_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--global-scale" && i + 1 < argc) {
            global_scale = std::strtof(argv[++i], nullptr);
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
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
            append_eval_csv(csv_path, algorithm, report);
            std::printf("%s\n", llama_expt::format_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay") {
            const llama_expt::attention_replay_eval_report report =
                llama_expt::evaluate_manifest_attention_replay(manifest_path);
            append_attention_replay_csv(csv_path, algorithm, report);
            std::printf("%s\n", llama_expt::format_attention_replay_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay_nvfp4_outlier") {
            const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
                llama_expt::evaluate_manifest_attention_replay_nvfp4_outlier(manifest_path);
            append_attention_quant_round_csv(csv_path, report);
            std::printf("%s\n", llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report).c_str());
        } else if (algorithm == "attention_replay_fp8_e4m3_e8m0") {
            const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
                llama_expt::evaluate_manifest_attention_replay_fp8_e4m3_e8m0(manifest_path);
            append_attention_quant_round_csv(csv_path, report);
            std::printf("%s\n", llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report).c_str());
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
