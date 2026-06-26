#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


N_LAYERS = 36
N_TOKENS = 512
V_CHANNELS = 1024
VP_CHANNELS = 4096


def flame_palette(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [255, 249, 229],
            [255, 219, 126],
            [247, 140, 45],
            [184, 39, 28],
            [74, 12, 16],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, stops.shape[0], dtype=np.float32)
    flat = values.reshape(-1)
    channels = [np.interp(flat, positions, stops[:, i]) for i in range(3)]
    rgb = np.stack(channels, axis=1).reshape(values.shape[0], values.shape[1], 3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_heatmap(raw_path: Path, image_path: Path, kind: str, layer: int, n_tokens: int, n_channels: int) -> dict:
    data = np.fromfile(raw_path, dtype=np.float32)
    expected = n_tokens * n_channels
    if data.size != expected:
        raise ValueError(f"{raw_path} has {data.size} floats, expected {expected}")

    matrix = data.reshape(n_tokens, n_channels).T
    abs_matrix = np.abs(matrix)
    p99 = float(np.percentile(abs_matrix, 99.0))
    vmax = p99 if p99 > 0.0 else float(abs_matrix.max(initial=0.0))
    if vmax <= 0.0:
        norm = np.zeros_like(abs_matrix, dtype=np.float32)
    else:
        norm = np.clip(abs_matrix / vmax, 0.0, 1.0).astype(np.float32)

    rgb = flame_palette(norm)
    img = Image.fromarray(rgb, mode="RGB")

    left, top, right, bottom = 72, 28, 18, 34
    canvas = Image.new("RGB", (left + n_tokens + right, top + n_channels + bottom), (250, 248, 241))
    canvas.paste(img, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    ink = (42, 34, 28)
    muted = (64, 52, 42)

    draw.text((left, 8), f"{kind} layer {layer:02d}", fill=ink, font=font)
    draw.text((6, top), "channel 1", fill=muted, font=font)
    draw.text((6, top + n_channels - 10), str(n_channels), fill=muted, font=font)
    draw.text((left, top + n_channels + 12), "token 1", fill=muted, font=font)
    draw.text((left + n_tokens - 48, top + n_channels + 12), str(n_tokens), fill=muted, font=font)

    for x in range(0, n_tokens + 1, 128):
        draw.line((left + x, top + n_channels, left + x, top + n_channels + 5), fill=(80, 65, 52))
    y_step = 256 if n_channels <= 1024 else 512
    for y in range(0, n_channels + 1, y_step):
        draw.line((left - 5, top + y, left, top + y), fill=(80, 65, 52))

    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path, optimize=True)

    return {
        "kind": kind,
        "layer": layer,
        "file": image_path.name,
        "source_file": str(raw_path),
        "shape": {"tokens": n_tokens, "channels": n_channels},
        "min": float(matrix.min(initial=0.0)),
        "max": float(matrix.max(initial=0.0)),
        "abs_max": float(abs_matrix.max(initial=0.0)),
        "abs_p99": p99,
        "mean_abs": float(abs_matrix.mean()),
        "std": float(matrix.std()),
    }


def image_card(image_dir_name: str, item: dict) -> str:
    title = f"{item['kind']} layer {item['layer']:02d}"
    file_name = html.escape(item["file"])
    return f"""
        <section class="heatmap">
          <h3>{html.escape(title)}</h3>
          <dl>
            <div><dt>abs p99</dt><dd>{item['abs_p99']:.6g}</dd></div>
            <div><dt>abs max</dt><dd>{item['abs_max']:.6g}</dd></div>
            <div><dt>mean abs</dt><dd>{item['mean_abs']:.6g}</dd></div>
            <div><dt>std</dt><dd>{item['std']:.6g}</dd></div>
          </dl>
          <a href="{image_dir_name}/{file_name}" target="_blank" rel="noopener">
            <img src="{image_dir_name}/{file_name}" alt="{html.escape(title)} heatmap">
          </a>
        </section>"""


def write_html(output: Path, image_dir_name: str, items: list[dict]) -> None:
    by_layer: dict[int, dict[str, dict]] = {}
    for item in items:
        by_layer.setdefault(item["layer"], {})[item["kind"]] = item

    nav = "".join(f'<a href="#layer-{i:02d}">{i:02d}</a>' for i in range(N_LAYERS))
    sections = []
    for layer in range(N_LAYERS):
        v = by_layer[layer]["V"]
        vp = by_layer[layer]["VP"]
        sections.append(f"""
    <section class="layer-card" id="layer-{layer:02d}">
      <h2>Layer {layer:02d}</h2>
      <div class="pair">
        {image_card(image_dir_name, v)}
        {image_card(image_dir_name, vp)}
      </div>
    </section>""")

    output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V and VP Heatmaps - One Chunk</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #271f1a;
      --muted: #76675a;
      --paper: #faf8f1;
      --line: #dfd4c3;
      --accent: #b8271c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 14px 18px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(250, 248, 241, 0.96);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      padding-top: 10px;
    }}
    nav a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 26px;
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      background: #fffaf0;
    }}
    main {{
      width: min(1320px, calc(100vw - 28px));
      margin: 18px auto 42px;
    }}
    .legend {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px 0 18px;
      color: var(--muted);
    }}
    .ramp {{
      width: 320px;
      max-width: 100%;
      height: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #fff9e5, #ffdb7e, #f78c2d, #b8271c, #4a0c10);
    }}
    .layer-card {{
      padding: 18px 0 30px;
      border-top: 1px solid var(--line);
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .pair {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .heatmap h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, auto);
      gap: 8px 12px;
      margin: 0 0 8px;
    }}
    dt {{
      color: var(--muted);
      font-size: 11px;
    }}
    dd {{
      margin: 0;
      font-size: 12px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      background: #fffaf0;
    }}
    a:focus-visible {{
      outline: 2px solid var(--accent);
      outline-offset: 3px;
    }}
    @media (max-width: 900px) {{
      .pair {{ grid-template-columns: 1fr; }}
      .legend {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>V and VP Heatmaps - One Chunk</h1>
    <div class="meta">
      <span>36 layers</span>
      <span>V: 512 tokens x 1024 channels</span>
      <span>VP: 512 tokens x 4096 channels</span>
      <span>color = abs(x), per-layer p99 capped</span>
    </div>
    <nav aria-label="Layer navigation">{nav}</nav>
  </header>
  <main>
    <div class="legend">
      <div>Horizontal axis is token position. Vertical axis is channel. VP is the V @ P attention result before output projection. Lighter cells are smaller absolute values; darker cells are larger absolute values.</div>
      <div class="ramp" aria-label="Light to dark color scale"></div>
    </div>
    {''.join(sections)}
  </main>
</body>
</html>
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    heatmap_dir = args.output_dir / "v_vp_heatmaps"
    items: list[dict] = []
    for layer in range(N_LAYERS):
        items.append(make_heatmap(
            args.raw_dir / "v_raw_f32" / f"layer_{layer:02d}.f32.bin",
            heatmap_dir / f"v_layer_{layer:02d}.png",
            "V",
            layer,
            N_TOKENS,
            V_CHANNELS,
        ))
        items.append(make_heatmap(
            args.raw_dir / "vp_raw_f32" / f"layer_{layer:02d}.f32.bin",
            heatmap_dir / f"vp_layer_{layer:02d}.png",
            "VP",
            layer,
            N_TOKENS,
            VP_CHANNELS,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "v_vp_heatmaps.html"
    manifest_path = args.output_dir / "v_vp_heatmaps_manifest.json"
    write_html(html_path, heatmap_dir.name, items)
    manifest_path.write_text(json.dumps({
        "raw_dir": str(args.raw_dir),
        "html": str(html_path),
        "n_items": len(items),
        "items": items,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
