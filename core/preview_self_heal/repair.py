"""Repair preview/thumb artifacts only — never touch source files."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.logger import setup_logger
from core.preview_self_heal.validate import Verdict, ValidationResult, validate_preview

logger = setup_logger(__name__)

_MAX_RETRIES = 2
_lock = threading.Lock()
_inflight: set[str] = set()
_retry_counts: dict[str, int] = {}
_cooldown_until: dict[str, float] = {}
_COOLDOWN_SEC = 120.0


@dataclass
class RepairOutcome:
    ok: bool
    action: str
    reason: str
    path: str = ""
    format: str = ""
    retry: int = 0
    final: bool = False
    preview_path: str = ""
    elapsed_ms: float = 0.0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "result": "ok" if self.ok else "failed",
            "reason": self.reason,
            "action": self.action,
            "retry": self.retry,
            "final": self.final,
            "preview_path": self.preview_path,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "ts": time.time(),
        }


def _key(path: str) -> str:
    return str(path or "").strip().lower().replace("/", "\\")


def _ext(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".") or "?"


def _unlink_artifact(path: str) -> None:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return
    # Safety: only under known cache segment names
    lowered = str(p).lower().replace("/", "\\")
    if not any(
        t in lowered
        for t in (
            "\\feature_previews\\",
            "\\thumbnails\\",
            "\\previews\\",
            ".v3_cache",
        )
    ):
        logger.warning("preview_self_heal refuse unlink non-cache: %s", p)
        return
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("unlink preview artifact failed: %s", exc)


def _append_log(settings: Any | None, entry: dict[str, Any]) -> None:
    try:
        base = ""
        if settings is not None:
            base = str(getattr(settings, "db_path", "") or "")
        if not base:
            return
        log_dir = Path(base).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "preview_self_heal.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def begin_repair(path: str) -> tuple[bool, int]:
    """Dedup + retry budget. Returns (allowed, retry_index)."""
    k = _key(path)
    if not k:
        return False, 0
    now = time.monotonic()
    with _lock:
        until = float(_cooldown_until.get(k) or 0)
        if until > now:
            return False, int(_retry_counts.get(k) or 0)
        if k in _inflight:
            return False, int(_retry_counts.get(k) or 0)
        n = int(_retry_counts.get(k) or 0)
        if n > _MAX_RETRIES:
            return False, n
        _inflight.add(k)
        return True, n


def end_repair(path: str, *, success: bool, final_fail: bool = False) -> None:
    k = _key(path)
    with _lock:
        _inflight.discard(k)
        if success:
            _retry_counts.pop(k, None)
            _cooldown_until.pop(k, None)
            return
        n = int(_retry_counts.get(k) or 0) + 1
        _retry_counts[k] = n
        if final_fail or n > _MAX_RETRIES:
            _cooldown_until[k] = time.monotonic() + _COOLDOWN_SEC * 5
        else:
            _cooldown_until[k] = time.monotonic() + _COOLDOWN_SEC


def repair_preview_artifact(
    source_path: str,
    *,
    artifact_path: str = "",
    create_fn: Callable[[], Any] | None = None,
    settings: Any | None = None,
    reason: str = "",
    compare_path: str | Path | None = None,
    max_retries: int = _MAX_RETRIES,
) -> RepairOutcome:
    """Delete bad cache artifact (if any) and recreate via existing renderer/Pool.

    ``create_fn`` must return an object with ``.success`` and ``.preview_path``
    (FeaturePreviewResult) — callers wire FeaturePreviewCache.create.
    """
    t0 = time.perf_counter()
    src = str(source_path or "").strip()
    fmt = _ext(src)
    allowed, retry = begin_repair(src or artifact_path)
    if not allowed:
        return RepairOutcome(
            ok=False,
            action="skip_dup_or_limit",
            reason=reason or "dup_or_retry_limit",
            path=src,
            format=fmt,
            retry=retry,
            final=retry > max_retries,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    try:
        if artifact_path:
            _unlink_artifact(artifact_path)
        if create_fn is None:
            out = RepairOutcome(
                ok=False,
                action="no_create_fn",
                reason=reason or "no_create_fn",
                path=src,
                format=fmt,
                retry=retry,
                final=True,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
            end_repair(src or artifact_path, success=False, final_fail=True)
            _append_log(settings, out.to_log_dict())
            return out

        result = create_fn()
        ok = bool(getattr(result, "success", False))
        new_path = str(getattr(result, "preview_path", "") or "")
        err = str(getattr(result, "error", "") or reason or "recreate_failed")

        if ok and new_path:
            # Re-validate lightly; allow compare against local raster if provided
            vr = validate_preview(
                new_path,
                src,
                allow_source_compare=bool(compare_path or src),
                compare_path=compare_path,
            )
            if vr.verdict == Verdict.INVALID:
                _unlink_artifact(new_path)
                final = retry >= max_retries
                out = RepairOutcome(
                    ok=False,
                    action="recreate_invalid",
                    reason=vr.reason,
                    path=src,
                    format=fmt,
                    retry=retry + 1,
                    final=final,
                    preview_path=new_path,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
                end_repair(src, success=False, final_fail=final)
                _append_log(settings, out.to_log_dict())
                return out
            # SUSPICIOUS after repair: accept if source compare says VALID; else soft-ok
            if vr.verdict == Verdict.SUSPICIOUS and compare_path:
                from core.preview_self_heal.validate import compare_source_vs_preview

                vr = compare_source_vs_preview(compare_path, new_path)
                if vr.verdict == Verdict.INVALID:
                    _unlink_artifact(new_path)
                    final = retry >= max_retries
                    out = RepairOutcome(
                        ok=False,
                        action="recreate_misrepresenting",
                        reason=vr.reason,
                        path=src,
                        format=fmt,
                        retry=retry + 1,
                        final=final,
                        preview_path=new_path,
                        elapsed_ms=(time.perf_counter() - t0) * 1000,
                    )
                    end_repair(src, success=False, final_fail=final)
                    _append_log(settings, out.to_log_dict())
                    return out

            out = RepairOutcome(
                ok=True,
                action="recreated",
                reason=vr.reason if isinstance(vr, ValidationResult) else (reason or "ok"),
                path=src,
                format=fmt,
                retry=retry + 1,
                final=True,
                preview_path=new_path,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
            end_repair(src, success=True)
            _append_log(settings, out.to_log_dict())
            return out

        final = retry >= max_retries
        out = RepairOutcome(
            ok=False,
            action="recreate_failed",
            reason=err,
            path=src,
            format=fmt,
            retry=retry + 1,
            final=final,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
        end_repair(src, success=False, final_fail=final)
        _append_log(settings, out.to_log_dict())
        return out
    except Exception as exc:
        final = retry >= max_retries
        out = RepairOutcome(
            ok=False,
            action="exception",
            reason=str(exc)[:300],
            path=src,
            format=fmt,
            retry=retry + 1,
            final=final,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
        end_repair(src or artifact_path, success=False, final_fail=final)
        _append_log(settings, out.to_log_dict())
        return out


def clear_repair_state_for_tests() -> None:
    with _lock:
        _inflight.clear()
        _retry_counts.clear()
        _cooldown_until.clear()
