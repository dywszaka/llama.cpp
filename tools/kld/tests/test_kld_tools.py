#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmallWikitextTests(unittest.TestCase):
    def test_selects_complete_documents_and_records_manifest(self) -> None:
        module = load_module(TOOL_DIR / "scripts" / "prepare-small-wikitext.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "wiki.test.raw"
            output = root / "small.raw"
            manifest = root / "manifest.json"
            source.write_text(
                "\n = A =\n\nalpha one.\nalpha two.\n"
                "\n = = A subsection = =\n\nalpha subsection.\n"
                "\n = B =\n\nbeta one.\nbeta two.\n"
                "\n = C =\n\ngamma one.\ngamma two.\n",
                encoding="utf-8",
            )

            result = module.prepare_small_wikitext(
                source=source,
                output=output,
                manifest_path=manifest,
                sample_count=2,
                start_document=1,
                min_chars=1,
            )

            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["start_document"], 1)
            self.assertEqual(result["selected_titles"], ["B", "C"])
            self.assertIn("= B =", output.read_text(encoding="utf-8"))
            self.assertIn("A subsection", result["available_documents"][0]["text_preview"])
            self.assertNotIn("= A =", output.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["selected_titles"], ["B", "C"])


class KldParserTests(unittest.TestCase):
    def test_parses_kld_metrics_and_diagnostic_histograms(self) -> None:
        module = load_module(TOOL_DIR / "scripts" / "parse-kld-results.py")

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            logs = exp / "logs"
            diagnostics = exp / "diagnostics" / "kld_ubatch_512"
            logs.mkdir(parents=True)
            diagnostics.mkdir(parents=True)
            (logs / "kld_ubatch_512.raw.log").write_text(
                "case=kld_ubatch_512\n"
                "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence\n"
                "   1      10.0 +/- 0.1       0.01 +/- 0.02       0.12 +/- 0.03\n"
                "Mean PPL(Q)                   :  10.682643 +/- 0.084131\n"
                "Mean PPL(base)                :  10.454407 +/- 0.082323\n"
                "Mean PPL(Q)/PPL(base)         :   1.021832 +/- 0.002063\n"
                "Mean    KLD:   0.145256 +/- 0.001261\n"
                "99.9%   KLD:   2.014898\n"
                "RMS dp    : 10.829 +/- 0.064 %\n"
                "Same top p: 84.613 +/- 0.094 %\n",
                encoding="utf-8",
            )
            (diagnostics / "attention-score.hist.json").write_text(
                json.dumps({"schema_version": 1, "sample_count": 4, "histogram": [{"lo": -1, "hi": 1, "count": 7}]}),
                encoding="utf-8",
            )
            (diagnostics / "k.sample-rows.jsonl").write_text(
                json.dumps({"row": 0, "values": [1.0, 2.0]}) + "\n",
                encoding="utf-8",
            )

            rows = module.parse_experiment(exp)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ubatch"], 512)
            self.assertAlmostEqual(rows[0]["mean_kld"], 0.145256)
            self.assertAlmostEqual(rows[0]["kld_p999"], 2.014898)
            self.assertEqual(rows[0]["diagnostic_histogram_files"], 1)
            self.assertEqual(rows[0]["diagnostic_sample_row_files"], 1)


if __name__ == "__main__":
    unittest.main()
