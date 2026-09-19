"""Build 40-file golden dataset for Index Engine V3 Phase 2 validation.

Mix:
  JPG/PNG ~20 | normal TIFF ~10 | large TIFF ~5 | hard TIFF ~5
Always include known large sample (real_076 / Çıtır Çiçek style) when present.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_v3"
FILES = GOLDEN / "files"
MANIFEST = GOLDEN / "manifest.json"

TARGET = 40
CANDIDATE_ROOTS = [
    Path(r"F:\karşıdan yüklemeler"),
    Path(r"F:\karsidan yuklemeler"),
]

# Prefer including this known large TIFF (Phase-2 hard case)
MUST_INCLUDE_SUBSTR = ("çıtır çiçek 6 mt", "citir cicek 6 mt", "6 mt.tif")

NORMAL_TIFF_MAX = 5_000_000
LARGE_TIFF_MIN = 5_000_000
LARGE_TIFF_MAX = 80_000_000
# Hard: very large files (decode stress) — still under a hard copy cap
HARD_TIFF_MIN = 40_000_000
HARD_TIFF_MAX = 120_000_000


def _kind(ext: str) -> str:
    e = ext.lower()
    if e in (".jpg", ".jpeg"):
        return "jpg"
    if e == ".png":
        return "png"
    if e in (".tif", ".tiff"):
        return "tiff"
    return ""


def _gather() -> dict[str, list[tuple[Path, int]]]:
    buckets: dict[str, list[tuple[Path, int]]] = {
        "jpg": [],
        "png": [],
        "tiff": [],
    }
    for root in CANDIDATE_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                kind = _kind(Path(name).suffix)
                if not kind:
                    continue
                src = Path(dirpath) / name
                try:
                    sz = src.stat().st_size
                except OSError:
                    continue
                if sz < 100:
                    continue
                buckets[kind].append((src, sz))
        break
    return buckets


def _is_must(path: Path) -> bool:
    low = path.name.lower()
    return any(s in low for s in MUST_INCLUDE_SUBSTR)


def _pick(items: list[tuple[Path, int]], n: int, *, used: set[str]) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for p, sz in items:
        key = str(p).lower()
        if key in used:
            continue
        out.append((p, sz))
        used.add(key)
        if len(out) >= n:
            break
    return out


def main() -> None:
    FILES.mkdir(parents=True, exist_ok=True)
    for p in FILES.iterdir():
        if p.is_file():
            p.unlink()

    buckets = _gather()
    used: set[str] = set()
    selected: list[tuple[str, Path, int, str]] = []  # bucket_label, path, bytes, kind

    # Must-include large/hard TIFF first
    must: list[tuple[Path, int]] = [
        (p, sz) for p, sz in buckets["tiff"] if _is_must(p)
    ]
    must.sort(key=lambda x: x[1], reverse=True)

    hard_pool = sorted(
        [
            (p, sz)
            for p, sz in buckets["tiff"]
            if HARD_TIFF_MIN <= sz <= HARD_TIFF_MAX
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    large_pool = sorted(
        [
            (p, sz)
            for p, sz in buckets["tiff"]
            if LARGE_TIFF_MIN <= sz <= LARGE_TIFF_MAX
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    normal_pool = sorted(
        [(p, sz) for p, sz in buckets["tiff"] if sz < NORMAL_TIFF_MAX],
        key=lambda x: x[1],
        reverse=True,
    )

    # hard ~5 (include must)
    hard_sel = _pick(must + hard_pool, 5, used=used)
    for p, sz in hard_sel:
        selected.append(("hard_tiff", p, sz, "tiff"))

    # large ~5 (not already used)
    large_sel = _pick(large_pool, 5, used=used)
    for p, sz in large_sel:
        selected.append(("large_tiff", p, sz, "tiff"))

    # normal ~10
    normal_sel = _pick(normal_pool, 10, used=used)
    for p, sz in normal_sel:
        selected.append(("normal_tiff", p, sz, "tiff"))

    # JPG/PNG ~20
    jpg_sel = _pick(buckets["jpg"], 12, used=used)
    png_sel = _pick(buckets["png"], 8, used=used)
    for p, sz in jpg_sel:
        selected.append(("jpg", p, sz, "jpg"))
    for p, sz in png_sel:
        selected.append(("png", p, sz, "png"))

    # Pad if short
    if len(selected) < TARGET:
        rest = buckets["jpg"] + buckets["png"] + normal_pool + large_pool
        for p, sz in rest:
            if len(selected) >= TARGET:
                break
            key = str(p).lower()
            if key in used:
                continue
            used.add(key)
            kind = _kind(p.suffix)
            selected.append(("pad", p, sz, kind))

    selected = selected[:TARGET]
    files_meta: list[dict] = []
    for i, (bucket, src, sz, kind) in enumerate(selected):
        ext = src.suffix.lower()
        name = f"real_{i:03d}{ext}"
        dest = FILES / name
        shutil.copy2(src, dest)
        files_meta.append(
            {
                "name": name,
                "kind": kind,
                "bucket": bucket,
                "source": "copied",
                "bytes": dest.stat().st_size,
                "from": str(src),
            }
        )

    by_kind: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    for f in files_meta:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        by_bucket[f["bucket"]] = by_bucket.get(f["bucket"], 0) + 1

    must_names = [f["name"] for f in files_meta if _is_must(Path(f["from"]))]
    manifest = {
        "total": len(files_meta),
        "by_kind": by_kind,
        "by_bucket": by_bucket,
        "must_include": must_names,
        "copied": len(files_meta),
        "synthetic": 0,
        "files": files_meta,
        "root": str(FILES),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "total": manifest["total"],
                "by_kind": by_kind,
                "by_bucket": by_bucket,
                "must_include": must_names,
                "root": str(FILES),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
