#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path


DTYPES = {
    "f32": ("<f", 4),
    "F32": ("<f", 4),
    "f16": ("<e", 2),
    "F16": ("<e", 2),
}


def tensor_by_id(metadata, tensor_id):
    for tensor in metadata["tensors"]:
        if tensor["id"] == tensor_id:
            return tensor
    return None


def load_tensor(dump_dir, meta):
    dtype = meta["dtype"]
    if dtype not in DTYPES:
        raise SystemExit(f"unsupported dtype: {dtype}")
    fmt, size = DTYPES[dtype]
    data = (dump_dir / meta["path"]).read_bytes()
    if len(data) != int(meta["nbytes"]):
        raise SystemExit(f"{meta['path']} size {len(data)} does not match metadata nbytes {meta['nbytes']}")
    return {
        "data": data,
        "fmt": fmt,
        "dtype_size": size,
        "shape": meta["shape"],
        "strides": meta.get("strides_bytes") or default_strides(meta["shape"], size),
    }


def default_strides(shape, dtype_size):
    return [
        dtype_size,
        dtype_size * shape[0],
        dtype_size * shape[0] * shape[1],
        dtype_size * shape[0] * shape[1] * shape[2],
    ]


def tensor_value(tensor, i0, i1, i2, i3):
    strides = tensor["strides"]
    byte_offset = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
    if byte_offset < 0 or byte_offset + tensor["dtype_size"] > len(tensor["data"]):
        raise SystemExit(f"tensor byte offset {byte_offset} is outside data size {len(tensor['data'])}")
    return struct.unpack_from(tensor["fmt"], tensor["data"], byte_offset)[0]


def flat_offset(shape, i0, i1, i2, i3):
    ne0, ne1, ne2, _ = shape
    return ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0


def alibi_slope(head, n_head, max_bias):
    if max_bias <= 0.0:
        return 1.0
    n_head_log2 = 1 << int(math.floor(math.log2(n_head)))
    m0 = math.pow(2.0, -max_bias / n_head_log2)
    m1 = math.pow(2.0, -(max_bias / 2.0) / n_head_log2)
    if head < n_head_log2:
        return math.pow(m0, head + 1)
    return math.pow(m1, 2 * (head - n_head_log2) + 1)


def recompute_softmax(input_tensor, mask_tensor, sinks_tensor, scale, max_bias):
    ne0, ne1, ne2, ne3 = input_tensor["shape"]
    result = [0.0] * (ne0 * ne1 * ne2 * ne3)
    mask_ne2 = mask_tensor["shape"][2] if mask_tensor else 1
    mask_ne3 = mask_tensor["shape"][3] if mask_tensor else 1

    for i3 in range(ne3):
        for i2 in range(ne2):
            slope = alibi_slope(i2, ne2, max_bias)
            sink = tensor_value(sinks_tensor, i2, 0, 0, 0) if sinks_tensor is not None else None
            for i1 in range(ne1):
                logits = []
                for i0 in range(ne0):
                    v = tensor_value(input_tensor, i0, i1, i2, i3) * scale
                    if mask_tensor is not None:
                        mask_i2 = i2 % mask_ne2
                        mask_i3 = i3 % mask_ne3
                        v += slope * tensor_value(mask_tensor, i0, i1, mask_i2, mask_i3)
                    logits.append(v)

                row_max = max(logits)
                if sink is not None:
                    row_max = max(row_max, sink)

                exps = [math.exp(v - row_max) for v in logits]
                denom = sum(exps)
                if sink is not None:
                    denom += math.exp(sink - row_max)
                for i0, ev in enumerate(exps):
                    result[flat_offset(input_tensor["shape"], i0, i1, i2, i3)] = ev / denom
    return result


def flatten_tensor(tensor):
    ne0, ne1, ne2, ne3 = tensor["shape"]
    values = [0.0] * (ne0 * ne1 * ne2 * ne3)
    for i3 in range(ne3):
        for i2 in range(ne2):
            for i1 in range(ne1):
                for i0 in range(ne0):
                    values[flat_offset(tensor["shape"], i0, i1, i2, i3)] = tensor_value(tensor, i0, i1, i2, i3)
    return values


def main():
    parser = argparse.ArgumentParser(description="Recompute first-decode attention softmax dump and compare to exported output")
    parser.add_argument("dump_dir", type=Path, nargs="?", default=Path("experiments/first-decode-attn-softmax-dump"))
    parser.add_argument("--limit", type=int, default=16, help="number of per-element diffs to preview")
    args = parser.parse_args()

    metadata = json.loads((args.dump_dir / "metadata.json").read_text(encoding="utf-8"))
    input_meta = tensor_by_id(metadata, "input")
    mask_meta = tensor_by_id(metadata, "mask")
    sinks_meta = tensor_by_id(metadata, "sinks")
    output_meta = tensor_by_id(metadata, "output")
    if input_meta is None or output_meta is None:
        raise SystemExit("metadata must contain input and output tensors")

    input_tensor = load_tensor(args.dump_dir, input_meta)
    output_tensor = load_tensor(args.dump_dir, output_meta)
    mask_tensor = load_tensor(args.dump_dir, mask_meta) if mask_meta is not None else None
    sinks_tensor = load_tensor(args.dump_dir, sinks_meta) if sinks_meta is not None else None
    if input_tensor["shape"] != output_tensor["shape"]:
        raise SystemExit("input and output shapes differ")

    softmax = metadata.get("softmax", {})
    scale = float(softmax.get("scale", 1.0))
    max_bias = float(softmax.get("max_bias", 0.0))
    recomputed = recompute_softmax(input_tensor, mask_tensor, sinks_tensor, scale, max_bias)
    output_values = flatten_tensor(output_tensor)

    diffs = [abs(a - b) for a, b in zip(recomputed, output_values)]
    sq = [(a - b) * (a - b) for a, b in zip(recomputed, output_values)]
    max_abs = max(diffs) if diffs else 0.0
    max_index = diffs.index(max_abs) if diffs else -1
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    rmse = math.sqrt(sum(sq) / len(sq)) if sq else 0.0

    preview = []
    for i in range(min(args.limit, len(diffs))):
        preview.append({
            "index": i,
            "expected": recomputed[i],
            "actual": output_values[i],
            "abs_diff": diffs[i],
        })

    result = {
        "dump": metadata.get("dump"),
        "count": len(diffs),
        "scale": scale,
        "max_bias": max_bias,
        "has_mask": mask_meta is not None,
        "has_sinks": sinks_meta is not None,
        "max_abs_diff": max_abs,
        "max_abs_diff_index": max_index,
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "preview": preview,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
