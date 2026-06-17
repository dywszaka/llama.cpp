#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path


DTYPES = {
    "f32": ("<f", 4),
    "F32": ("<f", 4),
}


def load_values(path: Path, dtype: str):
    if dtype not in DTYPES:
        raise SystemExit(f"unsupported dtype for plaintext summary: {dtype}")
    fmt, size = DTYPES[dtype]
    data = path.read_bytes()
    if len(data) % size != 0:
        raise SystemExit(f"{path} size {len(data)} is not divisible by dtype size {size}")
    return [v[0] for v in struct.iter_unpack(fmt, data)]


def summarize(values, limit):
    finite = [v for v in values if math.isfinite(v)]
    result = {
        "count": len(values),
        "finite_count": len(finite),
        "nan_count": sum(1 for v in values if math.isnan(v)),
        "pos_inf_count": sum(1 for v in values if v == math.inf),
        "neg_inf_count": sum(1 for v in values if v == -math.inf),
        "preview": values[:limit],
    }
    if finite:
        result.update({
            "min": min(finite),
            "max": max(finite),
            "sum": sum(finite),
            "mean": sum(finite) / len(finite),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Parse first-decode attention softmax tensor dump")
    parser.add_argument("dump_dir", type=Path, nargs="?", default=Path("experiments/first-decode-attn-softmax-dump"))
    parser.add_argument("--limit", type=int, default=16, help="number of leading values to include")
    parser.add_argument("--plain", action="store_true", help="include all tensor values")
    args = parser.parse_args()

    metadata_path = args.dump_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = {
        "dump": metadata.get("dump"),
        "directory": str(args.dump_dir),
        "attention_layer": metadata.get("attention_layer"),
        "tensors": [],
    }

    for tensor in metadata["tensors"]:
        tensor_path = args.dump_dir / tensor["path"]
        values = load_values(tensor_path, tensor["dtype"])
        item = {
            "id": tensor["id"],
            "path": tensor["path"],
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "nbytes": tensor["nbytes"],
        }
        item.update(summarize(values, args.limit))
        if args.plain:
            item["values"] = values
        output["tensors"].append(item)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
