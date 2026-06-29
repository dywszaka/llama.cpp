#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw_tensors"
VIZ = RESULTS / "visualization"


def check_raw(subdir: str, expected_size: int) -> None:
    files = sorted((RAW / subdir).glob("layer_*.f32.bin"))
    assert len(files) == 36, f"{subdir}: expected 36 files, got {len(files)}"
    sizes = {path.stat().st_size for path in files}
    assert sizes == {expected_size}, f"{subdir}: unexpected sizes {sizes}"


def check_manifest(path: Path, kinds: tuple[str, str], image_dir: str) -> None:
    manifest = json.loads(path.read_text())
    assert manifest["n_items"] == 72, f"{path}: expected 72 items"
    items = manifest["items"]
    for kind in kinds:
        layers = sorted(item["layer"] for item in items if item["kind"] == kind)
        assert layers == list(range(36)), f"{kind}: missing layers {layers}"
    pngs = sorted((VIZ / image_dir).glob("*.png"))
    assert len(pngs) == 72, f"{image_dir}: expected 72 PNGs, got {len(pngs)}"


def main() -> None:
    check_raw("q_raw_f32", 8_388_608)
    check_raw("kq_raw_f32", 33_554_432)
    check_raw("v_raw_f32", 2_097_152)
    check_raw("vp_raw_f32", 8_388_608)

    check_manifest(VIZ / "q_kq_heatmaps_manifest.json", ("Q", "KQ"), "q_kq_heatmaps")
    check_manifest(VIZ / "v_vp_heatmaps_manifest.json", ("V", "VP"), "v_vp_heatmaps")

    assert Image.open(VIZ / "q_kq_heatmaps/q_layer_00.png").size == (602, 4158)
    assert Image.open(VIZ / "q_kq_heatmaps/kq_layer_00.png").size == (602, 16446)
    assert Image.open(VIZ / "v_vp_heatmaps/v_layer_00.png").size == (602, 1086)
    assert Image.open(VIZ / "v_vp_heatmaps/vp_layer_00.png").size == (602, 4158)

    dist = json.loads((RESULTS / "tensor_distribution.json").read_text())
    counts = {kind: sum(1 for item in dist["tensors"] if item["kind"] == kind)
              for kind in sorted({item["kind"] for item in dist["tensors"]})}
    assert counts == {"KQ": 36, "Q": 36, "V": 36, "VP": 36}, counts

    print("kqv-heatmap outputs validated")


if __name__ == "__main__":
    main()
