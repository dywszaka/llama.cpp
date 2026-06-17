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
            values_out = [0.1, 0.2, 0.3, 0.4]
            (dump_dir / "attn_softmax_input.bin").write_bytes(struct.pack("<4f", *values_in))
            (dump_dir / "attn_softmax_output.bin").write_bytes(struct.pack("<4f", *values_out))

            metadata = {
                "schema_version": 1,
                "dump": "first-decode-attn-softmax",
                "tensors": [
                    {
                        "id": "input",
                        "path": "attn_softmax_input.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
                        "nbytes": 16,
                    },
                    {
                        "id": "output",
                        "path": "attn_softmax_output.bin",
                        "dtype": "f32",
                        "shape": [4, 1, 1, 1],
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
            self.assertEqual(parsed["tensors"][1]["id"], "output")
            self.assertAlmostEqual(parsed["tensors"][1]["sum"], 1.0)


if __name__ == "__main__":
    unittest.main()
