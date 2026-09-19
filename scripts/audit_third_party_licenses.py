"""Basit dağıtım öncesi lisans envanteri.

Bu araç patent/lisans uygunluğu GARANTİSİ vermez. requirements/pyproject ve kaynak
metinlerindeki yaygın lisans ifadelerini tarayıp insan incelemesi için raporlar.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "MIT": re.compile(r"MIT License|Permission is hereby granted", re.I),
    "Apache-2.0": re.compile(r"Apache License, Version 2\.0|Apache-2\.0", re.I),
    "BSD": re.compile(r"BSD License|Redistribution and use in source and binary forms", re.I),
    "GPL": re.compile(r"GNU GENERAL PUBLIC LICENSE|GPL-2\.0|GPL-3\.0|AGPL", re.I),
}

def main() -> int:
    hits = {k: [] for k in PATTERNS}
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".py", ".txt", ".md", ".toml", ".cfg"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, rx in PATTERNS.items():
            if rx.search(text):
                hits[name].append(str(p.relative_to(ROOT)))
    print("Vezir üçün üçüncü taraf lisans taraması")
    for name, files in hits.items():
        print(f"{name}: {len(files)} dosya")
        for f in files[:20]:
            print("  -", f)
    print("Not: Bu statik envanterdir; model ağırlıkları, veri kümeleri ve dağıtım lisansları ayrıca doğrulanmalıdır.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
