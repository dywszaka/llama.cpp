#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HEADING_RE = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_wikitext_documents(text: str) -> list[dict[str, Any]]:
    matches = list(HEADING_RE.finditer(text))
    documents: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        documents.append(
            {
                "index": idx,
                "title": match.group(1).strip(),
                "char_start": start,
                "char_end": end,
                "char_count": len(body),
                "text": body,
            }
        )
    return documents


def prepare_small_wikitext(
    *,
    source: Path,
    output: Path,
    manifest_path: Path,
    sample_count: int,
    start_document: int,
    min_chars: int,
) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8", errors="replace")
    documents = [doc for doc in split_wikitext_documents(text) if int(doc["char_count"]) >= min_chars]
    selected = documents[start_document : start_document + sample_count]
    if len(selected) != sample_count:
        raise ValueError(
            f"requested {sample_count} documents from offset {start_document}, "
            f"but only {len(selected)} were available after min_chars={min_chars}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(str(doc["text"]) for doc in selected) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "sample_unit": "complete_wikitext_documents",
        "sample_count": sample_count,
        "start_document": start_document,
        "min_chars": min_chars,
        "selected_titles": [str(doc["title"]) for doc in selected],
        "available_documents": [
            {
                "source_index": int(doc["index"]),
                "title": str(doc["title"]),
                "char_count": int(doc["char_count"]),
                "text_preview": str(doc["text"])[:160].replace("\n", "\\n"),
            }
            for doc in documents[: start_document + sample_count]
        ],
        "selected_documents": [
            {
                "source_index": int(doc["index"]),
                "title": str(doc["title"]),
                "char_start": int(doc["char_start"]),
                "char_end": int(doc["char_end"]),
                "char_count": int(doc["char_count"]),
            }
            for doc in selected
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small complete-document Wikitext sample for KLD smoke runs.")
    parser.add_argument("--source", type=Path, default=Path("data/wikitext/wikitext-2-raw/wiki.test.raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    manifest = prepare_small_wikitext(
        source=args.source,
        output=args.output,
        manifest_path=args.manifest,
        sample_count=args.sample_count,
        start_document=args.start_document,
        min_chars=args.min_chars,
    )
    print(json.dumps({"output": manifest["output"], "sample_count": manifest["sample_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
