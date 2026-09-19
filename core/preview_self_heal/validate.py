"""Preview artifact validation — artifact-first; source compare only if suspicious.

Critical rule: solid black/white/monochrome preview alone is NOT broken when
the source is the same (VALID). Only SUSPICIOUS/REPAIR when source is colorful
but preview is solid mono, or artifact is corrupt / unusable / missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Verdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"


@dataclass(frozen=True)
class ValidationResult:
    verdict: Verdict
    reason: str
    artifact_path: str = ""
    mean: float = 0.0
    std: float = 0.0
    white_ratio: float = 0.0
    black_ratio: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict == Verdict.VALID

    @property
    def needs_repair(self) -> bool:
        return self.verdict == Verdict.INVALID


def _sample_stats(
    path: str | Path,
    *,
    max_edge: int = 256,
    min_bytes: int = 32,
    min_edge: int = 8,
) -> tuple[dict[str, float], str]:
    """Decode local image → stats. Never opens NAS unless path is the file."""
    p = Path(path)
    try:
        if not p.is_file():
            return {}, "missing"
        size = int(p.stat().st_size)
        if size < min_bytes:
            return {}, "too_small"
    except OSError as exc:
        return {}, f"stat:{exc}"

    try:
        from PIL import Image
        import numpy as np

        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < min_edge or h < min_edge:
                return {}, "tiny_dims"
            # Half-crop / degenerate: one edge collapsed vs the other
            if min(w, h) <= 2 and max(w, h) >= 64:
                return {}, "half_crop"
            sample = img
            if max(w, h) > max_edge:
                sample = img.copy()
                sample.thumbnail((max_edge, max_edge))
            arr = np.asarray(sample, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std())
            white = float((arr >= 250).mean())
            black = float((arr <= 5).mean())
            return {
                "mean": mean,
                "std": std,
                "white_ratio": white,
                "black_ratio": black,
                "w": float(w),
                "h": float(h),
            }, "ok"
    except Exception as exc:
        return {}, f"decode:{exc}"


def _is_solid_mono(
    stats: dict[str, float],
    *,
    min_std: float = 2.0,
    max_white_ratio: float = 0.992,
    max_black_ratio: float = 0.992,
) -> tuple[bool, str]:
    """Artifact looks solid black/white/near-monochrome — not a verdict alone."""
    mean = float(stats.get("mean") or 0)
    std = float(stats.get("std") or 0)
    white = float(stats.get("white_ratio") or 0)
    black = float(stats.get("black_ratio") or 0)
    if std < min_std and mean >= 250:
        return True, "mono_white"
    if std < min_std and mean <= 5:
        return True, "mono_black"
    if white >= max_white_ratio and std < 8.0:
        return True, "near_white"
    if black >= max_black_ratio and std < 8.0:
        return True, "near_black"
    if std < min_std:
        return True, "mono_flat"
    return False, ""


def _is_colorful(stats: dict[str, float], *, min_std: float = 12.0) -> bool:
    """Source has enough variation that a solid mono preview is suspicious."""
    std = float(stats.get("std") or 0)
    white = float(stats.get("white_ratio") or 0)
    black = float(stats.get("black_ratio") or 0)
    if std >= min_std:
        return True
    # Patterned but low-contrast still counts if not near-solid
    if std >= 4.0 and white < 0.95 and black < 0.95:
        return True
    return False


def validate_artifact_light(
    artifact_path: str | Path,
    *,
    min_bytes: int = 32,
    min_edge: int = 8,
) -> ValidationResult:
    """Artifact-only. Never reads source/NAS.

    Solid mono → SUSPICIOUS (not INVALID). Corrupt/missing/unusable → INVALID.
    """
    ap = str(artifact_path or "").strip()
    if not ap:
        return ValidationResult(Verdict.INVALID, "missing_path")

    stats, reason = _sample_stats(ap, min_bytes=min_bytes, min_edge=min_edge)
    if reason != "ok":
        return ValidationResult(Verdict.INVALID, reason, artifact_path=ap)

    mono, mono_reason = _is_solid_mono(stats)
    base = dict(
        artifact_path=ap,
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        white_ratio=float(stats["white_ratio"]),
        black_ratio=float(stats["black_ratio"]),
    )
    if mono:
        return ValidationResult(Verdict.SUSPICIOUS, mono_reason, **base)
    return ValidationResult(Verdict.VALID, "ok", **base)


def compare_source_vs_preview(
    source_path: str | Path,
    artifact_path: str | Path,
    *,
    preview_stats: dict[str, float] | None = None,
) -> ValidationResult:
    """Heavy path: only for SUSPICIOUS candidates. Compares local samples."""
    ap = str(artifact_path or "").strip()
    sp = str(source_path or "").strip()
    light = validate_artifact_light(ap) if preview_stats is None else None
    if preview_stats is None:
        if light is None:
            return ValidationResult(Verdict.INVALID, "missing_path", artifact_path=ap)
        if light.verdict == Verdict.INVALID:
            return light
        if light.verdict == Verdict.VALID:
            return light
        pstats = {
            "mean": light.mean,
            "std": light.std,
            "white_ratio": light.white_ratio,
            "black_ratio": light.black_ratio,
        }
        mono_reason = light.reason
    else:
        pstats = preview_stats
        mono, mono_reason = _is_solid_mono(pstats)
        if not mono:
            return ValidationResult(
                Verdict.VALID,
                "ok",
                artifact_path=ap,
                mean=float(pstats.get("mean") or 0),
                std=float(pstats.get("std") or 0),
                white_ratio=float(pstats.get("white_ratio") or 0),
                black_ratio=float(pstats.get("black_ratio") or 0),
            )

    if not sp:
        return ValidationResult(
            Verdict.SUSPICIOUS,
            f"{mono_reason}:no_source",
            artifact_path=ap,
            mean=float(pstats.get("mean") or 0),
            std=float(pstats.get("std") or 0),
            white_ratio=float(pstats.get("white_ratio") or 0),
            black_ratio=float(pstats.get("black_ratio") or 0),
        )

    sstats, sreason = _sample_stats(sp, max_edge=128)
    if sreason != "ok":
        # Cannot prove mismatch without source sample — stay suspicious, don't delete.
        return ValidationResult(
            Verdict.SUSPICIOUS,
            f"{mono_reason}:source_{sreason}",
            artifact_path=ap,
            mean=float(pstats.get("mean") or 0),
            std=float(pstats.get("std") or 0),
            white_ratio=float(pstats.get("white_ratio") or 0),
            black_ratio=float(pstats.get("black_ratio") or 0),
        )

    base = dict(
        artifact_path=ap,
        mean=float(pstats.get("mean") or 0),
        std=float(pstats.get("std") or 0),
        white_ratio=float(pstats.get("white_ratio") or 0),
        black_ratio=float(pstats.get("black_ratio") or 0),
    )

    src_mono, src_mono_reason = _is_solid_mono(sstats)
    if src_mono:
        # Same polarity or both flat → VALID (do not repair white-on-white).
        return ValidationResult(
            Verdict.VALID, f"source_matches_mono:{src_mono_reason}", **base
        )

    if _is_colorful(sstats):
        return ValidationResult(
            Verdict.INVALID, f"misrepresenting:{mono_reason}", **base
        )

    return ValidationResult(
        Verdict.SUSPICIOUS, f"{mono_reason}:source_ambiguous", **base
    )


def validate_preview(
    artifact_path: str | Path,
    source_path: str | Path | None = None,
    *,
    allow_source_compare: bool = False,
    compare_path: str | Path | None = None,
) -> ValidationResult:
    """Full policy: light first; source/compare_path only when suspicious + allowed.

    ``compare_path`` may be a local raster already produced (render_path) so
    Mode C avoids a second NAS read of the original.
    """
    light = validate_artifact_light(artifact_path)
    if light.verdict != Verdict.SUSPICIOUS:
        return light
    if not allow_source_compare:
        return light
    peer = compare_path or source_path
    if not peer:
        return light
    return compare_source_vs_preview(
        peer,
        artifact_path,
        preview_stats={
            "mean": light.mean,
            "std": light.std,
            "white_ratio": light.white_ratio,
            "black_ratio": light.black_ratio,
        },
    )


def is_networkish(path: str) -> bool:
    try:
        from core.network_index_throttle import is_network_path

        return bool(is_network_path(path))
    except Exception:
        p = str(path or "").replace("/", "\\")
        return p.startswith("\\\\")
