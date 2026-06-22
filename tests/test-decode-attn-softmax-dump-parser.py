import json
import struct
import subprocess
import unittest
from pathlib import Path


class DecodeAttnSoftmaxDumpParserTest(unittest.TestCase):
    def test_parser_summarizes_dump(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp) / "dump"
            dump_dir.mkdir()

            values_in = [1.0, 2.0, 3.0, 4.0]
            values_mask = [0.0, 0.0, 0.0, 0.0]
            values_out = [0.1, 0.2, 0.3, 0.4]
            (dump_dir / "attn_softmax_input.bin").write_bytes(struct.pack("<4f", *values_in))
            (dump_dir / "attn_softmax_mask.bin").write_bytes(struct.pack("<4e", *values_mask))
            (dump_dir / "attn_softmax_output.bin").write_bytes(struct.pack("<4f", *values_out))

            metadata = {
                "schema_version": 1,
                "dump": "first-decode-attn-softmax",
                "softmax": {
                    "scale": 1.0,
                    "max_bias": 0.0,
                },
                "tensors": [
                    {
                        "id": "input",
                        "path": "attn_softmax_input.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [4, 16, 16, 16],
                        "nbytes": 16,
                    },
                    {
                        "id": "mask",
                        "path": "attn_softmax_mask.bin",
                        "dtype": "f16",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [2, 8, 8, 8],
                        "nbytes": 8,
                    },
                    {
                        "id": "output",
                        "path": "attn_softmax_output.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [4, 16, 16, 16],
                        "nbytes": 16,
                    },
                ],
            }
            (dump_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            script = Path(__file__).resolve().parents[1] / "scripts" / "parse-first-decode-attn-softmax-dump.py"
            result = subprocess.run(
                ["python3", str(script), str(dump_dir), "--limit", "2"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["dump"], "first-decode-attn-softmax")
            self.assertEqual(parsed["tensors"][0]["id"], "input")
            self.assertEqual(parsed["tensors"][0]["count"], 4)
            self.assertEqual(parsed["tensors"][0]["min"], 1.0)
            self.assertEqual(parsed["tensors"][0]["max"], 4.0)
            self.assertEqual(parsed["tensors"][0]["preview"], [1.0, 2.0])
            self.assertEqual(parsed["tensors"][1]["id"], "mask")
            self.assertEqual(parsed["tensors"][1]["dtype"], "f16")
            self.assertEqual(parsed["tensors"][2]["id"], "output")
            self.assertAlmostEqual(parsed["tensors"][2]["sum"], 1.0)

    def test_compare_recomputes_softmax_with_mask(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp) / "dump"
            dump_dir.mkdir()

            values_in = [1.0, 2.0, 3.0, 4.0]
            values_mask = [0.0, -1.0, 0.0, -2.0]
            # softmax([1, 1, 3, 2])
            logits = [1.0, 1.0, 3.0, 2.0]
            import math
            max_logit = max(logits)
            denom = sum(math.exp(v - max_logit) for v in logits)
            values_out = [math.exp(v - max_logit) / denom for v in logits]

            (dump_dir / "attn_softmax_input.bin").write_bytes(struct.pack("<4f", *values_in))
            (dump_dir / "attn_softmax_mask.bin").write_bytes(struct.pack("<4f", *values_mask))
            (dump_dir / "attn_softmax_output.bin").write_bytes(struct.pack("<4f", *values_out))

            metadata = {
                "schema_version": 1,
                "dump": "first-decode-attn-softmax",
                "softmax": {
                    "scale": 1.0,
                    "max_bias": 0.0,
                },
                "tensors": [
                    {
                        "id": "input",
                        "path": "attn_softmax_input.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [4, 16, 16, 16],
                        "nbytes": 16,
                    },
                    {
                        "id": "mask",
                        "path": "attn_softmax_mask.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [4, 16, 16, 16],
                        "nbytes": 16,
                    },
                    {
                        "id": "output",
                        "path": "attn_softmax_output.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "strides_bytes": [4, 16, 16, 16],
                        "nbytes": 16,
                    },
                ],
            }
            (dump_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            script = Path(__file__).resolve().parents[1] / "scripts" / "compare-first-decode-attn-softmax-dump.py"
            result = subprocess.run(
                ["python3", str(script), str(dump_dir), "--limit", "2"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["count"], 4)
            self.assertLess(parsed["max_abs_diff"], 1e-6)
            self.assertEqual(parsed["preview"][0]["index"], 0)

    def test_compare_matches_cpu_softmax_with_sinks(self):
        import math
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp) / "dump"
            dump_dir.mkdir()

            values_in = [1.0, 2.0, 3.0]
            sink = 4.0
            row_max = max(max(values_in), sink)
            exps = [math.exp(v - row_max) for v in values_in]
            denom = sum(exps) + math.exp(sink - row_max)
            values_out = [v / denom for v in exps]

            (dump_dir / "attn_softmax_input.bin").write_bytes(struct.pack("<3f", *values_in))
            (dump_dir / "attn_softmax_sinks.bin").write_bytes(struct.pack("<f", sink))
            (dump_dir / "attn_softmax_output.bin").write_bytes(struct.pack("<3f", *values_out))

            metadata = {
                "schema_version": 1,
                "dump": "first-decode-attn-softmax",
                "softmax": {
                    "scale": 1.0,
                    "max_bias": 0.0,
                },
                "tensors": [
                    {
                        "id": "input",
                        "path": "attn_softmax_input.bin",
                        "dtype": "f32",
                        "shape": [3, 1, 1, 1],
                        "strides_bytes": [4, 12, 12, 12],
                        "nbytes": 12,
                    },
                    {
                        "id": "sinks",
                        "path": "attn_softmax_sinks.bin",
                        "dtype": "f32",
                        "shape": [1, 1, 1, 1],
                        "strides_bytes": [4, 4, 4, 4],
                        "nbytes": 4,
                    },
                    {
                        "id": "output",
                        "path": "attn_softmax_output.bin",
                        "dtype": "f32",
                        "shape": [3, 1, 1, 1],
                        "strides_bytes": [4, 12, 12, 12],
                        "nbytes": 12,
                    },
                ],
            }
            (dump_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            script = Path(__file__).resolve().parents[1] / "scripts" / "compare-first-decode-attn-softmax-dump.py"
            result = subprocess.run(
                ["python3", str(script), str(dump_dir), "--limit", "3"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["count"], 3)
            self.assertTrue(parsed["has_sinks"])
            self.assertLess(parsed["max_abs_diff"], 1e-6)


if __name__ == "__main__":
    unittest.main()
