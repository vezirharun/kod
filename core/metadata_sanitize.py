"""Metadata değerlerini JSON/DB yazımı için güvenli tiplere dönüştür."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime
from fractions import Fraction
from numbers import Rational
from typing import Any


def _is_ifd_rational(value: Any) -> bool:
    try:
        from PIL.TiffImagePlugin import IFDRational

        return isinstance(value, IFDRational)
    except ImportError:
        return type(value).__name__ == "IFDRational"


def _rational_to_json(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        num = getattr(value, "numerator", None)
        den = getattr(value, "denominator", None)
        if num is not None and den is not None:
            return f"{num}/{den}"
        return str(value)


def sanitize_for_json(value: Any) -> Any:
    """Recursive — IFDRational, bytes, tuple ve bilinmeyen tipler."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_ifd_rational(value) or isinstance(value, Rational):
        return _rational_to_json(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return _rational_to_json(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Fraction):
        return _rational_to_json(value)
    if isinstance(value, dict):
        return {
            str(sanitize_for_json(k)): sanitize_for_json(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    return str(value)


def metadata_to_json(metadata: Any, *, default: str = "{}") -> str:
    """Metadata dict → JSON; hata durumunda boş obje döner."""
    try:
        clean = sanitize_for_json(metadata if metadata is not None else {})
        return json.dumps(clean, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(sanitize_for_json(str(metadata)), ensure_ascii=False)
        except Exception:
            return default
