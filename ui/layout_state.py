"""Pencere ve splitter yerleşimini güvenli kaydet/geri yükle."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow, QSplitter

_MIN_SPLITTER_TOTAL = 120

_DEFAULT_CONTENT = [420, 520]
_DEFAULT_QUERY_DETAIL = [900, 420]


def _decode_state(value) -> QByteArray:
    if not value:
        return QByteArray()
    if isinstance(value, QByteArray):
        return value
    if isinstance(value, str):
        try:
            return QByteArray.fromBase64(value.encode("ascii"))
        except Exception:
            return QByteArray()
    return QByteArray()


def _encode_state(value: QByteArray) -> str:
    try:
        return bytes(value.toBase64()).decode("ascii")
    except Exception:
        return ""


def _valid_sizes(sizes: list[int], min_each: int = 80) -> bool:
    if not sizes or sum(sizes) < _MIN_SPLITTER_TOTAL:
        return False
    return all(int(s) >= min_each for s in sizes)


def _set_safe_sizes(
    splitter: QSplitter | None, sizes: list[int], min_each: int = 80
) -> None:
    if not splitter:
        return
    if _valid_sizes(sizes, min_each=min_each):
        splitter.setSizes(sizes)


def reset_layout(window: QMainWindow) -> None:
    """Panel düzenini varsayılanlara döndür."""
    content = getattr(window, "_content_splitter", None)
    query_detail = getattr(window, "_query_detail_splitter", None)
    main = getattr(window, "_main_splitter", None)

    if content:
        content.setSizes(list(_DEFAULT_CONTENT))
    if query_detail:
        query_detail.setSizes(list(_DEFAULT_QUERY_DETAIL))
    if main and main is not query_detail:
        main.setSizes(list(_DEFAULT_QUERY_DETAIL))

    for dock in window.findChildren(QDockWidget):
        name = dock.objectName()
        if name == "SourceDock":
            dock.hide()
        elif name == "InspectorDock":
            dock.hide()
        elif name == "FormatStatusDock":
            dock.hide()
        elif name == "CategoryTreeDock":
            dock.hide()
        elif name == "TeachMeDock":
            dock.hide()
        elif name == "FilterDock":
            dock.hide()
        elif name == "QueryDock":
            dock.show()
        else:
            dock.show()

    ui = getattr(getattr(window, "settings", None), "ui_state", None) or {}
    if isinstance(ui, dict):
        ui.pop("geometry", None)
        mode = getattr(getattr(window, "settings", None), "ui_mode", "simple")
        ui.pop(f"window_state_{mode}", None)
        ui.pop("content_splitter", None)
        ui.pop("query_detail_splitter", None)
        ui.pop("main_splitter", None)
        ui["source_dock_visible"] = False
        ui["inspector_dock_visible"] = False
        ui["filters_visible"] = False
        ui["query_panel_visible"] = True
        try:
            window.settings.ui_state = ui
            window.settings.save()
        except Exception:
            pass


def restore_layout(settings, window: QMainWindow) -> None:
    """Kayıtlı yerleşimi geri yükle; bozuk/çok küçük değerleri güvenli atla."""
    ui = getattr(settings, "ui_state", None) or {}
    if not isinstance(ui, dict):
        ui = {}

    geom = _decode_state(ui.get("geometry"))
    if not geom.isEmpty():
        window.restoreGeometry(geom)

    mode = getattr(settings, "ui_mode", "simple")
    state = _decode_state(ui.get(f"window_state_{mode}") or ui.get("window_state"))
    if not state.isEmpty():
        window.restoreState(state)

    content = getattr(window, "_content_splitter", None)
    query_detail = getattr(window, "_query_detail_splitter", None)
    main = getattr(window, "_main_splitter", None)

    _set_safe_sizes(content, ui.get("content_splitter", []), min_each=160)
    _set_safe_sizes(query_detail, ui.get("query_detail_splitter", []), min_each=240)
    if main is not query_detail:
        _set_safe_sizes(main, ui.get("main_splitter", []), min_each=200)

    if content and not _valid_sizes(content.sizes(), min_each=160):
        content.setSizes(list(_DEFAULT_CONTENT))
    if query_detail and not _valid_sizes(query_detail.sizes(), min_each=240):
        query_detail.setSizes(list(_DEFAULT_QUERY_DETAIL))


def save_layout(settings, window: QMainWindow) -> None:
    """Geometri ve splitter oranlarını settings içine yazar."""
    ui = getattr(settings, "ui_state", None) or {}
    if not isinstance(ui, dict):
        ui = {}

    ui["geometry"] = _encode_state(window.saveGeometry())
    mode = getattr(settings, "ui_mode", "simple")
    ui[f"window_state_{mode}"] = _encode_state(window.saveState())
    ui["window_state"] = ui[f"window_state_{mode}"]

    content = getattr(window, "_content_splitter", None)
    query_detail = getattr(window, "_query_detail_splitter", None)
    main = getattr(window, "_main_splitter", None)

    if content:
        sizes = content.sizes()
        if sizes and sum(sizes) >= _MIN_SPLITTER_TOTAL and int(sizes[0]) >= 80:
            ui["content_splitter"] = sizes
    if query_detail:
        sizes = query_detail.sizes()
        if _valid_sizes(sizes, min_each=80):
            ui["query_detail_splitter"] = sizes
    if main and main is not query_detail:
        sizes = main.sizes()
        if _valid_sizes(sizes, min_each=80):
            ui["main_splitter"] = sizes

    try:
        settings.ui_state = ui
    except Exception:
        pass
