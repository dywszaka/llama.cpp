#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "docs/development/nvfp4-kcache-outlier-thresholds/scripts/parse-kcache-outlier-threshold-sweep.py"


def load_parser():
    spec = importlib.util.spec_from_file_location("kcache_outlier_parser", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KcacheOutlierThresholdParserTest(unittest.TestCase):
    def test_parses_current_compact_overflow_log_field(self):
        parser = load_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threshold_12.raw.log"
            path.write_text(
                "\n".join([
                    "llama_kv_cache_unified: NVFP4 K-cache compact outlier sidecar enabled: threshold_profile=env-override layer_capacity_profile=balanced layer_capacities=36",
                    "ggml_cuda_nvfp4_kcache_outlier_extract: target=cache_k_l2 (view) rows=512 threshold=12 stored_max=1024 compact_capacity=1024 compact_used=1482 compact_overflow=0 total_outliers=286 max_row_outliers=2 overflow_rows=0",
                    "Final estimate: PPL = 68.3777 +/- 0.1234",
                ])
                + "\n"
            )

            threshold, summary, layer_rows = parser.parse_log(path, layers=36, head_dim=1024)

        self.assertEqual(threshold, 12)
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["total_outliers"], 286)
        self.assertEqual(summary["max_call_total"], 286)
        self.assertEqual(layer_rows[2]["total_outliers"], 286)
        self.assertEqual(layer_rows[2]["max_row_outliers"], 2)


if __name__ == "__main__":
    unittest.main()
