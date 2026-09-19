"""Minimal stubs so Customer Memory tests run with PYTHONPATH=cm_impl."""
from __future__ import annotations

import sys
import types


def _ensure(name: str, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    # register as package attr if parent exists
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child, mod)
    return mod


# core package already exists via cm_impl/core
def _norm_tr(s: str) -> str:
    t = str(s or "").strip().lower()
    for a, b in (
        ("ı", "i"),
        ("İ", "i"),
        ("ğ", "g"),
        ("ü", "u"),
        ("ş", "s"),
        ("ö", "o"),
        ("ç", "c"),
    ):
        t = t.replace(a, b)
    return t


_ensure("core.logger", setup_logger=lambda name: types.SimpleNamespace(
    debug=lambda *a, **k: None,
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
))
_ensure("core.textile_terms", normalize_turkish=_norm_tr)
_ensure(
    "core.utils",
    normalize_path=lambda p: str(p or "").replace("\\", "/"),
    normalize_source_root=lambda p: str(p or "").replace("\\", "/").rstrip("/"),
)
_ensure(
    "core.manual_label_guard",
    is_manual_labeled=lambda *a, **k: False,
    parse_texture_map=lambda x: x if isinstance(x, dict) else {},
)
_ensure(
    "core.search_models",
    SearchResponse=type("SearchResponse", (), {}),
    SearchStats=type("SearchStats", (), {}),
)
_ensure("core.color_index")
_ensure("core.query_attribute_intel", extract_query_attributes=lambda q: types.SimpleNamespace(
    colors=[], motif="", scale="", density=""
), concept_core_text=lambda q: q)
_ensure("core.concept_query_normalize", concept_match_key=lambda s: " ".join(str(s or "").strip().lower().split()), leaf_translation_keys=lambda n: set(), relation_to_concept=lambda *a, **k: False)
