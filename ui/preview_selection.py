"""Arama sonucu önizleme seçimi ve panoya kopyalama.

Arama motoru, ranking, index ve orijinal dosyalara dokunmaz.
Yalnızca kartta kullanılan thumbnail/preview dosya yollarını kullanır.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable


def _abs(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "").strip()))


def preview_path_for_result(result: Any) -> str:
    """Kartın kullandığı önizleme; orijinal desen dosyası değil."""
    thumb = str(getattr(result, "thumbnail_path", "") or "").strip()
    original = str(getattr(result, "path", "") or "").strip()
    if not thumb:
        return ""
    if not os.path.isfile(thumb):
        return ""
    if original and _abs(thumb) == _abs(original):
        return ""
    return os.path.abspath(thumb)


@dataclass
class PreviewCopyPlan:
    paths: list[str] = field(default_factory=list)
    copied_ids: list[int] = field(default_factory=list)
    missing_ids: list[int] = field(default_factory=list)
    skipped_original_ids: list[int] = field(default_factory=list)


def build_copy_plan(results: Iterable[Any], selected_ids: Iterable[int]) -> PreviewCopyPlan:
    wanted = []
    seen_ids: set[int] = set()
    for fid in selected_ids:
        try:
            i = int(fid)
        except (TypeError, ValueError):
            continue
        if i <= 0 or i in seen_ids:
            continue
        seen_ids.add(i)
        wanted.append(i)
    by_id: dict[int, Any] = {}
    order: list[int] = []
    for rec in results or []:
        try:
            fid = int(getattr(rec, "file_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if fid <= 0 or fid in by_id:
            continue
        by_id[fid] = rec
        order.append(fid)
    plan = PreviewCopyPlan()
    seen_paths: set[str] = set()
    for fid in order:
        if fid not in seen_ids:
            continue
        rec = by_id[fid]
        original = str(getattr(rec, "path", "") or "").strip()
        thumb = str(getattr(rec, "thumbnail_path", "") or "").strip()
        if thumb and original and os.path.isfile(thumb) and _abs(thumb) == _abs(original):
            plan.skipped_original_ids.append(fid)
            continue
        path = preview_path_for_result(rec)
        if not path:
            plan.missing_ids.append(fid)
            continue
        key = _abs(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        plan.paths.append(path)
        plan.copied_ids.append(fid)
    for fid in wanted:
        if fid not in by_id and fid not in plan.missing_ids:
            plan.missing_ids.append(fid)
    return plan


def ordered_unique_ids(file_ids: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in file_ids:
        try:
            fid = int(raw)
        except (TypeError, ValueError):
            continue
        if fid <= 0 or fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
    return out


def ids_between(ordered_ids: Iterable[int], start_id: int, end_id: int) -> list[int]:
    ordered = ordered_unique_ids(ordered_ids)
    try:
        start = int(start_id)
        end = int(end_id)
    except (TypeError, ValueError):
        return []
    if end <= 0:
        return []
    if start not in ordered:
        return [end] if end in ordered else []
    if end not in ordered:
        return []
    i = ordered.index(start)
    j = ordered.index(end)
    lo, hi = (i, j) if i <= j else (j, i)
    return ordered[lo : hi + 1]


class PreviewSelectionStore:
    """file_id kümesi — virtual list geri dönüşümünden bağımsız."""

    def __init__(self) -> None:
        self._ids: set[int] = set()
        self._last_id: int = 0

    def clear(self) -> None:
        self._ids.clear()
        self._last_id = 0

    def count(self) -> int:
        return len(self._ids)

    def ids(self) -> set[int]:
        return set(self._ids)

    def is_selected(self, file_id: int) -> bool:
        try:
            fid = int(file_id)
        except (TypeError, ValueError):
            return False
        return fid in self._ids

    def set_selected(self, file_id: int, checked: bool) -> None:
        try:
            fid = int(file_id)
        except (TypeError, ValueError):
            return
        if fid <= 0:
            return
        if checked:
            self._ids.add(fid)
        else:
            self._ids.discard(fid)

    def select_many(self, file_ids: Iterable[int]) -> None:
        for raw in file_ids:
            self.set_selected(raw, True)

    def retain(self, valid_ids: Iterable[int]) -> None:
        allowed = set()
        for raw in valid_ids:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                continue
            if i > 0:
                allowed.add(i)
        self._ids &= allowed

    def toggle(self, file_id: int) -> bool:
        on = not self.is_selected(file_id)
        self.set_selected(file_id, on)
        try:
            fid = int(file_id)
        except (TypeError, ValueError):
            fid = 0
        if fid > 0:
            self._last_id = fid
        return on

    def apply_checkbox(self, file_id: int, checked: bool) -> None:
        self.set_selected(file_id, checked)
        try:
            fid = int(file_id)
        except (TypeError, ValueError):
            return
        if fid > 0:
            self._last_id = fid

    def apply_card_click(
        self,
        file_id: int,
        ordered_ids: Iterable[int],
        *,
        ctrl: bool = False,
        shift: bool = False,
    ) -> None:
        """Windows Explorer: düz tık = tekli, Ctrl = ekle/çıkar, Shift = aralık."""
        try:
            fid = int(file_id)
        except (TypeError, ValueError):
            return
        if fid <= 0:
            return
        ordered = ordered_unique_ids(ordered_ids)
        if shift:
            start = self._last_id if self._last_id in ordered else fid
            self.select_many(ids_between(ordered, start, fid))
            self._last_id = fid
            return
        if ctrl:
            self.toggle(fid)
            return
        self._ids.clear()
        self._ids.add(fid)
        self._last_id = fid


def preview_context_menu_spec(*, this_selected: bool, count: int) -> list[tuple[str, str]]:
    """Sağ tık menü satırları: (action_id, Türkçe etiket)."""
    n = max(0, int(count or 0))
    rows: list[tuple[str, str]] = [
        ("toggle", "Seçimi kaldır" if this_selected else "Deseni seç"),
        ("visible", "Tüm görünenleri seç"),
        ("all", "Tüm sonuçları seç"),
    ]
    if n <= 0:
        rows.append(("copy_original", "Orijinal deseni kopyala"))
        return rows
    rows.extend(
        [
            ("clear", "Seçimi temizle"),
            ("copy", "Seçilen önizlemeleri kopyala"),
            ("copy_original", "Orijinal deseni kopyala"),
            (
                "edit",
                f"Seçilenleri düzenle ({n})" if n > 1 else "Seçileni düzenle",
            ),
            ("count", f"Seçilen: {n}"),
        ]
    )
    return rows


def copy_previews_to_clipboard(plan: PreviewCopyPlan, clipboard) -> str:
    """Windows: dosya listesi (CF_HDROP). Orijinal dosya yok. Yeni cache yazılmaz."""
    if clipboard is None:
        return "Pano kullanılamıyor."
    if not plan.paths:
        if plan.skipped_original_ids and not plan.missing_ids:
            return (
                "Seçilenlerin ayrı önizleme dosyası yok. "
                "Orijinal desen dosyası panoya kopyalanmaz."
            )
        return "Kopyalanacak önizleme bulunamadı. Önce önizlemenin yüklenmesini bekleyin."

    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QImage

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in plan.paths])
    if len(plan.paths) == 1:
        img = QImage(plan.paths[0])
        if not img.isNull():
            mime.setImageData(img)
    clipboard.setMimeData(mime)

    n = len(plan.paths)
    msg = f"{n} önizleme panoya kopyalandı. WhatsApp’ta Ctrl+V ile yapıştırabilirsiniz."
    miss = len(plan.missing_ids)
    skip = len(plan.skipped_original_ids)
    if miss or skip:
        extra = []
        if miss:
            extra.append(f"{miss} sonuçta önizleme yok")
        if skip:
            extra.append(f"{skip} sonuç orijinal dosya olduğu için atlandı")
        msg += " " + "; ".join(extra) + "."
    return msg


def original_path_for_result(result: Any) -> str:
    """Sonuç kartının işaret ettiği gerçek kaynak dosya. Önizleme değil."""
    path = str(getattr(result, "path", "") or "").strip()
    if not path or not os.path.isfile(path):
        return ""
    return os.path.abspath(path)


def build_original_copy_plan(
    results: Iterable[Any], selected_ids: Iterable[int]
) -> PreviewCopyPlan:
    """Seçilen sonuçların orijinal dosyaları. Aynı yol bir kez. Taşıma yok."""
    wanted = ordered_unique_ids(selected_ids)
    by_id: dict[int, Any] = {}
    order: list[int] = []
    for rec in results or []:
        try:
            fid = int(getattr(rec, "file_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if fid <= 0 or fid in by_id:
            continue
        by_id[fid] = rec
        order.append(fid)
    plan = PreviewCopyPlan()
    seen_paths: set[str] = set()
    wanted_set = set(wanted)
    for fid in order:
        if fid not in wanted_set:
            continue
        rec = by_id[fid]
        path = original_path_for_result(rec)
        if not path:
            plan.missing_ids.append(fid)
            continue
        key = _abs(path)
        if key in seen_paths:
            plan.skipped_original_ids.append(fid)
            continue
        seen_paths.add(key)
        plan.paths.append(path)
        plan.copied_ids.append(fid)
    for fid in wanted:
        if fid not in by_id and fid not in plan.missing_ids:
            plan.missing_ids.append(fid)
    return plan


def copy_originals_to_clipboard(plan: PreviewCopyPlan, clipboard) -> str:
    """Orijinal dosyaları Windows panosuna CF_HDROP olarak koyar. Kaynak değişmez."""
    if clipboard is None:
        return "Pano kullanılamıyor."
    if not plan.paths:
        return "Kopyalanacak orijinal desen dosyası yok."

    from PySide6.QtCore import QMimeData, QUrl

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in plan.paths])
    clipboard.setMimeData(mime)

    n = len(plan.paths)
    msg = (
        f"{n} orijinal desen panoya kopyalandı. "
        "Windows'ta istediğiniz klasöre Yapıştır yapabilirsiniz."
    )
    extra = []
    if plan.missing_ids:
        extra.append(f"{len(plan.missing_ids)} dosya bulunamadı")
    if plan.skipped_original_ids:
        extra.append(f"{len(plan.skipped_original_ids)} aynı orijinal bir kez alındı")
    if extra:
        msg += " " + "; ".join(extra) + "."
    return msg
