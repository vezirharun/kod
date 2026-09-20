"""Sonuç detayı — AI çıkarımını düzeltme formu (kategori belleği, index yok)."""

from __future__ import annotations

from PySide6.QtCore import QStringListModel, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.responsive_dialog import apply_responsive_dialog, make_content_scroll_area

from core.canonical_correction import (
    FIELD_BRAND,
    FIELD_CHILD,
    FIELD_COLOR,
    FIELD_FAMILY,
    FIELD_PARENT,
    FIELD_TAG,
    correct_learned_label,
)
from core.category_memory import (
    dynamic_parents,
    register_category,
    register_root_category,
)
from core.category_tree import pattern_fields_for_path
from core.category_selection_sync import (
    canonical_path_from_parts,
    merge_tags_preserve_manual,
    path_segment_tags,
    resolve_parts_against_options,
    split_category_selection,
    sync_selection_state,
)
from core.textile_terms import FAMILY_UI_CHOICES, normalize_turkish
from ui.designer_labels import COLOR_LABELS_TR, brand_badge_text, color_badge_text

# Coalesce repeated option loads (Edit / Teach).
_OPTIONS_CACHE: dict[str, dict] = {}
_OPTIONS_CACHE_TS: dict[str, float] = {}
# Persist until invalidate_edit_options_cache (teach/edit/correct) — no timed rebuild.
_OPTIONS_TTL_SEC = float("inf")


def _norm(text: str) -> str:
    return normalize_turkish(text or "")


def form_values_from_record(result, overlay: dict | None = None) -> dict:
    """Yalnızca bu desenin overlay + kayıtlı metadata'sı. Önceki form state yok."""
    ov = dict(overlay or {})
    dbg = getattr(result, "debug", None) or {} if result is not None else {}
    if not isinstance(dbg, dict):
        dbg = {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    path = str(
        ov.get("category_path")
        or dbg.get("manual_category_path")
        or ""
    ).strip()
    parent = str(ov.get("parent") or "").strip()
    child = str(ov.get("child") or "").strip()
    if path and not parent:
        parent, _, child = path.partition("/")
        parent, child = parent.strip(), child.strip()
    family = str(
        ov.get("pattern_family")
        or (getattr(result, "pattern_family", "") if result is not None else "")
        or ""
    ).strip()
    if family.lower() in {"unknown", "none"}:
        family = ""
    color = str(
        ov.get("color_family")
        or (getattr(result, "color_family", "") if result is not None else "")
        or ""
    ).strip()
    if color.lower() in {"unknown", "none"}:
        color = ""
    brand = str(ov.get("brand") or "").strip()
    if not brand and result is not None:
        brand = brand_badge_text(result)
    if not brand:
        brand = str(dbg.get("brand_name") or tm.get("brand_name") or "").strip()
    if brand.lower() in {"unknown", "none"}:
        brand = ""
    tags = list(ov.get("tags") or [])
    if not tags and result is not None:
        tags = list(dbg.get("user_tags") or [])
    return {
        "parent": parent,
        "child": child,
        "category_path": path,
        "pattern_family": family,
        "color_family": color,
        "brand": brand,
        "tags": [str(t).strip() for t in tags if str(t).strip()],
    }


def load_edit_form_options(db_path: str) -> dict:
    """DB option snapshot for comboboxes — safe off the UI thread.

    Batched: one concept scan + one category_memory scan (not per-parent).
    Per-parent child_categories/dynamic_children loops lock/freeze large DBs.
    """
    import time

    from core.category_memory import _key
    from core.category_tree import CATEGORY_TREE

    key = str(db_path or "")
    now = time.monotonic()
    cached = _OPTIONS_CACHE.get(key)
    ts = float(_OPTIONS_CACHE_TS.get(key) or 0)
    if cached is not None and (now - ts) < _OPTIONS_TTL_SEC:
        return dict(cached)

    parents: list[str] = list(CATEGORY_TREE.keys())
    children_by_parent: dict[str, list[str]] = {
        p: list(kids.keys()) for p, kids in CATEGORY_TREE.items()
    }
    parent_key_to_name: dict[str, str] = {_key(p): p for p in parents}
    kid_keys_by_parent: dict[str, set[str]] = {
        p: {_key(c) for c in kids} for p, kids in children_by_parent.items()
    }

    def _ensure_parent(name: str) -> str:
        n = str(name or "").strip()
        if not n:
            return ""
        k = _key(n)
        if not k:
            return ""
        existing = parent_key_to_name.get(k)
        if existing:
            return existing
        parent_key_to_name[k] = n
        parents.append(n)
        children_by_parent.setdefault(n, [])
        kid_keys_by_parent.setdefault(n, set())
        return n

    def _add_child(parent: str, child: str) -> None:
        p = _ensure_parent(parent) if parent else ""
        c = str(child or "").strip()
        if not p or not c:
            return
        ck = _key(c)
        if not ck or ck in kid_keys_by_parent[p]:
            return
        kid_keys_by_parent[p].add(ck)
        children_by_parent[p].append(c)

    if db_path:
        try:
            for p in dynamic_parents(db_path):
                _ensure_parent(p)
        except Exception:
            pass
        try:
            from core.category_memory import all_dynamic

            for row in all_dynamic(db_path):
                _add_child(str(row.get("parent") or ""), str(row.get("label") or ""))
        except Exception:
            pass
        try:
            from core.concept_registry import concepts

            for row in concepts(db_path):
                status = str(row.get("status") or "active")
                if status in ("inactive", "retired"):
                    continue
                parent = str(row.get("parent") or "").strip()
                can = str(row.get("canonical") or "").strip()
                if parent:
                    _ensure_parent(parent)
                if parent and can:
                    _add_child(parent, can)
        except Exception:
            pass

        brands: list[str] = []
        try:
            from core.brand_aliases import brand_names

            brands = list(brand_names(db_path))
        except Exception:
            brands = []
        # Marka: collapse aliases once (avoid per-row DB in dynamic_children loop).
        marka_name = parent_key_to_name.get(_key("Marka"))
        if marka_name and brands:
            children_by_parent[marka_name] = list(brands)
            kid_keys_by_parent[marka_name] = {_key(b) for b in brands}
    else:
        brands = []

    def _bucket_kids(label: str) -> list[str]:
        if label in children_by_parent:
            return list(children_by_parent.get(label) or [])
        lk = _key(label)
        for p, kids in children_by_parent.items():
            if _key(p) == lk:
                return list(kids or [])
        return []

    family = list(FAMILY_UI_CHOICES)
    for extra in _bucket_kids("Desen ailesi"):
        if not any(extra == fid or extra == lab for lab, fid in family):
            family.append((extra, extra))

    colors = [(lab, ckey) for ckey, lab in COLOR_LABELS_TR.items() if lab]
    for extra in _bucket_kids("Renk"):
        if not any(extra == lab or extra == ckey for lab, ckey in colors):
            colors.append((extra, extra))

    tags = _bucket_kids("Etiket")

    from core.concept_query_normalize import turkish_sort_key

    parents = sorted(parents, key=turkish_sort_key)
    children_by_parent = {
        p: sorted(list(kids or []), key=turkish_sort_key)
        for p, kids in children_by_parent.items()
    }
    family = sorted(family, key=lambda pair: turkish_sort_key(pair[0]))
    colors = sorted(colors, key=lambda pair: turkish_sort_key(pair[0]))
    brands = sorted(list(brands or []), key=turkish_sort_key)
    tags = sorted(list(tags or []), key=turkish_sort_key)

    data = {
        "parents": parents,
        "children_by_parent": children_by_parent,
        "family": family,
        "colors": colors,
        "brands": brands,
        "tags": tags,
    }
    _OPTIONS_CACHE[key] = data
    _OPTIONS_CACHE_TS[key] = now
    return dict(data)


def invalidate_edit_options_cache(db_path: str = "") -> None:
    if db_path:
        _OPTIONS_CACHE.pop(str(db_path), None)
        _OPTIONS_CACHE_TS.pop(str(db_path), None)
    else:
        _OPTIONS_CACHE.clear()
        _OPTIONS_CACHE_TS.clear()
    try:
        from core.semantic_category_autocomplete import invalidate_autocomplete_caches

        invalidate_autocomplete_caches()
    except Exception:
        pass


class TypeaheadCombo(QComboBox):
    """Editable combo: semantic concept/context filter over learned records."""

    def __init__(self, parent=None, *, allow_new: bool = True):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._allow_new = allow_new
        self._choices: list[tuple[str, str]] = []
        self._semantic_catalog = None  # optional richer AutocompleteCandidate list
        self._choices_candidates = None  # indexed candidates from set_choices
        self._parent_hint = ""
        self._field_kind = "category"
        self._ac_model = QStringListModel(self)
        self._ac = QCompleter(self._ac_model, self)
        self._ac.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._ac.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._ac.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(self._ac)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._pending_text = ""
        self._debounce.timeout.connect(self._apply_typed_filter)
        self.lineEdit().textEdited.connect(self._on_typed)

    def set_semantic_catalog(self, catalog) -> None:
        """Optional cross-path catalog (Edit/Teach shared). Preindexes leaf keys."""
        if catalog is None:
            self._semantic_catalog = None
            return
        try:
            from core.semantic_category_autocomplete import ensure_candidates_indexed

            self._semantic_catalog = ensure_candidates_indexed(catalog)
        except Exception:
            self._semantic_catalog = catalog

    def set_parent_hint(self, parent: str) -> None:
        self._parent_hint = str(parent or "").strip()
        self._choices_candidates = None

    def set_field_kind(self, kind: str) -> None:
        self._field_kind = str(kind or "category")
        self._choices_candidates = None

    def set_choices(self, items: list[tuple[str, str]]) -> None:
        """items: (display, data). UI safety-sorts by Turkish alphabet."""
        from core.concept_query_normalize import turkish_sort_key

        raw = [(str(d), str(v)) for d, v in items if str(d).strip()]
        self._choices = sorted(raw, key=lambda pair: turkish_sort_key(pair[0]))
        self._choices_candidates = None
        self.blockSignals(True)
        self.clear()
        for display, data in self._choices:
            self.addItem(display, data)
        # Boş kayıtta ilk öğeyi (ör. son kaydedilen POLO) seçme.
        self.setCurrentIndex(-1)
        if self.lineEdit() is not None:
            self.lineEdit().setText("")
        self.blockSignals(False)
        self._ac_model.setStringList([])

    def set_text(self, text: str, data: str | None = None) -> None:
        text = str(text or "").strip()
        if not text:
            self.setCurrentIndex(-1)
            self.setEditText("")
            return
        idx = self.findText(text, Qt.MatchFlag.MatchFixedString)
        if idx < 0 and data:
            idx = self.findData(data)
        if idx >= 0:
            self.setCurrentIndex(idx)
            return
        self.setEditText(text)

    def current_text(self) -> str:
        return (self.currentText() or "").strip()

    def current_data_or_text(self) -> str:
        data = self.currentData()
        if data and self.currentText() == self.itemText(self.currentIndex()):
            return str(data)
        return self.current_text()

    def is_known(self) -> bool:
        typed = _norm(self.current_text())
        if not typed:
            return True
        if any(_norm(d) == typed or _norm(v) == typed for d, v in self._choices):
            return True
        # Path-qualified display ("Animal Print / Leopard")
        if "/" in typed:
            leaf = typed.rsplit("/", 1)[-1].strip()
            if leaf and any(_norm(d) == _norm(leaf) or _norm(v) == _norm(leaf) for d, v in self._choices):
                return True
        return False

    def filtered_matches(self, query: str) -> list[tuple[str, str]]:
        """Empty → no dump. Typed → semantic concept/context rank over cache."""
        q = " ".join(str(query or "").strip().split())
        if not q:
            return []
        try:
            from core.semantic_category_autocomplete import (
                candidates_from_choices,
                ensure_candidates_indexed,
                filter_choice_tuples,
            )

            if self._semantic_catalog is not None:
                catalog = ensure_candidates_indexed(self._semantic_catalog)
                self._semantic_catalog = catalog
            else:
                if self._choices_candidates is None:
                    self._choices_candidates = candidates_from_choices(
                        self._choices,
                        parent=self._parent_hint,
                        kind=self._field_kind,
                    )
                catalog = ensure_candidates_indexed(self._choices_candidates)
                self._choices_candidates = catalog
            return filter_choice_tuples(
                q,
                self._choices,
                parent=self._parent_hint,
                kind=self._field_kind,
                allow_new=False,
                catalog=catalog,
            )
        except Exception:
            nq = _norm(q)
            return [
                (d, v)
                for d, v in self._choices
                if nq in _norm(d) or nq in _norm(v)
            ]

    def _on_typed(self, text: str) -> None:
        self._pending_text = text
        self._debounce.start()

    def _apply_typed_filter(self) -> None:
        text = self._pending_text
        matches = self.filtered_matches(text)
        labels = [d for d, _ in matches]
        self._ac_model.setStringList(labels)
        popup = self._ac.popup()
        if (text or "").strip() and labels:
            self._ac.setCompletionPrefix("")
            self._ac.complete()
        elif popup is not None:
            popup.hide()
        self._update_new_hint(text, matches)

    def _update_new_hint(self, text: str, matches: list[tuple[str, str]]) -> None:
        hint = getattr(self, "_new_hint", None)
        if hint is None:
            return
        t = (text or "").strip()
        show = bool(self._allow_new and t and not matches and not self.is_known())
        hint.setVisible(show)
        if show:
            hint.setText("Yeni ekle")


class _OptionsWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, db_path: str, parent=None):
        super().__init__(parent)
        self._db_path = str(db_path or "")

    def run(self) -> None:
        try:
            self.finished_ok.emit(load_edit_form_options(self._db_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class ResultMetadataDialog(QDialog):
    """Kaydet = bu görsel, bellek katmanı. Öğret ve Uygula isteğe bağlı."""

    def __init__(
        self,
        parent=None,
        *,
        result=None,
        db_path: str = "",
        overlay: dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Düzenle — metadata")
        self.setMinimumWidth(440)
        self._db_path = str(db_path or "")
        self._result = result
        self._overlay = dict(overlay or {})
        self._teach_apply = False
        self._tags: list[str] = []
        self._options: dict = {}
        self._options_loaded = False
        self._options_worker: _OptionsWorker | None = None
        self._options_req_id = 0
        self._syncing_category = False
        self._build_ui()
        self._prefill()
        apply_responsive_dialog(self, min_width=440, prefer_width=480)
        QTimer.singleShot(0, self._start_options_load)

    def ensure_options_ready(self, *, force: bool = False) -> None:
        """Sync option fill for unit tests (or when worker has not finished)."""
        if force:
            invalidate_edit_options_cache(self._db_path)
            self._options_loaded = False
        if self._options_loaded and self.cmb_parent.count() > 0 and not force:
            return
        self._options_req_id += 1
        self._on_options_loaded(load_edit_form_options(self._db_path), req_id=self._options_req_id)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._body_layout = layout

        self.lbl_intro = QLabel(
            "AI çıkarımını düzeltin. Kayıt yalnızca kategori belleğine yazılır; "
            "desen indeksi değişmez."
        )
        self.lbl_intro.setWordWrap(True)
        layout.addWidget(self.lbl_intro)

        form = QFormLayout()
        self.cmb_parent = TypeaheadCombo()
        self.cmb_child = TypeaheadCombo()
        self.cmb_family = TypeaheadCombo()
        self.cmb_color = TypeaheadCombo()
        self.cmb_brand = TypeaheadCombo()
        self.cmb_parent.currentIndexChanged.connect(self._on_parent_changed)
        self.cmb_parent.lineEdit().editingFinished.connect(self._on_parent_changed)
        self.cmb_child.currentIndexChanged.connect(self._on_child_changed)
        self.cmb_child.lineEdit().editingFinished.connect(self._on_child_changed)

        self.lbl_new_parent = QLabel("")
        self.lbl_new_child = QLabel("")
        self.lbl_new_family = QLabel("")
        self.lbl_new_color = QLabel("")
        self.lbl_new_brand = QLabel("")
        for combo, hint in (
            (self.cmb_parent, self.lbl_new_parent),
            (self.cmb_child, self.lbl_new_child),
            (self.cmb_family, self.lbl_new_family),
            (self.cmb_color, self.lbl_new_color),
            (self.cmb_brand, self.lbl_new_brand),
        ):
            hint.setStyleSheet("color:#1d4ed8;font-weight:600;")
            hint.setVisible(False)
            combo._new_hint = hint
        self.cmb_parent.set_field_kind("category")
        self.cmb_child.set_field_kind("category")
        self.cmb_family.set_field_kind("family")
        self.cmb_color.set_field_kind("other")
        self.cmb_brand.set_field_kind("other")

        tag_row = QWidget()
        tag_l = QHBoxLayout(tag_row)
        tag_l.setContentsMargins(0, 0, 0, 0)
        self.txt_tag = QLineEdit()
        self.txt_tag.setPlaceholderText("Etiket yazın")
        self.btn_add_tag = QPushButton("+ Etiket")
        self.btn_add_tag.clicked.connect(self._add_tag)
        self.txt_tag.returnPressed.connect(self._add_tag)
        tag_l.addWidget(self.txt_tag, stretch=1)
        tag_l.addWidget(self.btn_add_tag)
        self.lbl_tags = QLabel("—")
        self.lbl_tags.setWordWrap(True)

        form.addRow("Ana kategori", self.cmb_parent)
        form.addRow("", self.lbl_new_parent)
        form.addRow("Alt kategori", self.cmb_child)
        form.addRow("", self.lbl_new_child)
        form.addRow("Etiketler", tag_row)
        form.addRow("", self.lbl_tags)
        form.addRow("Desen ailesi", self.cmb_family)
        form.addRow("", self.lbl_new_family)
        form.addRow("Renk", self.cmb_color)
        form.addRow("", self.lbl_new_color)
        form.addRow("Marka", self.cmb_brand)
        form.addRow("", self.lbl_new_brand)
        layout.addLayout(form)

        self.lbl_options_status = QLabel("Listeler hazırlanıyor…")
        self.lbl_options_status.setStyleSheet("color:#64748b;")
        layout.addWidget(self.lbl_options_status)

        self._install_correct_menus()

        scroll = make_content_scroll_area(body)
        root.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox()
        self.btn_save = buttons.addButton(
            "Kaydet", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.btn_teach = buttons.addButton(
            "Öğret ve Uygula", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.btn_cancel = buttons.addButton(
            "İptal", QDialogButtonBox.ButtonRole.RejectRole
        )
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_teach.clicked.connect(self._on_teach)
        self.btn_cancel.clicked.connect(self.reject)
        root.addWidget(buttons)

    def _install_correct_menus(self) -> None:
        pairs = (
            (self.cmb_parent, FIELD_PARENT, "Ana kategori"),
            (self.cmb_child, FIELD_CHILD, "Alt kategori"),
            (self.cmb_family, FIELD_FAMILY, "Desen ailesi"),
            (self.cmb_color, FIELD_COLOR, "Renk"),
            (self.cmb_brand, FIELD_BRAND, "Marka"),
        )
        for combo, field, title in pairs:
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            combo.customContextMenuRequested.connect(
                lambda pos, c=combo, f=field, t=title: self._open_correct_menu(
                    c, f, t, pos
                )
            )
        self.txt_tag.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.txt_tag.customContextMenuRequested.connect(
            lambda pos: self._open_correct_menu(
                self.txt_tag, FIELD_TAG, "Etiket", pos
            )
        )
        self.lbl_tags.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lbl_tags.customContextMenuRequested.connect(
            lambda pos: self._open_correct_menu(
                self.txt_tag, FIELD_TAG, "Etiket", pos
            )
        )

    def _open_correct_menu(self, widget, field: str, title: str, pos) -> None:
        menu = QMenu(self)
        act = menu.addAction("Düzenle / Düzelt")
        chosen = menu.exec(widget.mapToGlobal(pos))
        if chosen is act:
            self._correct_field(field, title)

    def _field_current_text(self, field: str) -> str:
        if field == FIELD_PARENT:
            return self.cmb_parent.current_text()
        if field == FIELD_CHILD:
            return self.cmb_child.current_text()
        if field == FIELD_FAMILY:
            return self.cmb_family.current_text()
        if field == FIELD_COLOR:
            return self.cmb_color.current_text()
        if field == FIELD_BRAND:
            return self.cmb_brand.current_text()
        if field == FIELD_TAG:
            typed = self.txt_tag.text().strip()
            if typed:
                return typed
            if self._tags:
                return str(self._tags[-1])
        return ""

    def _correct_field(self, field: str, title: str) -> None:
        old = self._field_current_text(field).strip()
        if not old:
            QMessageBox.information(
                self,
                title,
                "Önce listeden veya kutudan bir değer seçin / yazın.",
            )
            return
        if field == FIELD_CHILD and not self.cmb_parent.current_text().strip():
            QMessageBox.information(
                self,
                title,
                "Ana ve alt kategoriyi seçin, sonra düzeltin.",
            )
            return
        new, ok = QInputDialog.getText(
            self,
            "Düzenle / Düzelt",
            f"Yeni ad (eski: {old}):",
            text=old,
        )
        if not ok:
            return
        new = " ".join(str(new or "").strip().split())
        if not new:
            return
        parent = self.cmb_parent.current_text().strip()
        stats = correct_learned_label(
            self._db_path,
            field,
            old,
            new,
            parent=parent,
        )
        if not stats.get("ok"):
            reason = str(stats.get("reason") or "bilinmeyen hata")
            if reason == "distinct_concepts":
                msg = (
                    f"“{old}” ve “{new}” farklı kavramlar; "
                    "yanlışlıkla birleştirilmedi (ör. Tiger ≠ Leopard)."
                )
            else:
                msg = f"Düzeltilemedi: {reason}"
            QMessageBox.warning(self, title, msg)
            return
        final = str(stats.get("final") or new)
        invalidate_edit_options_cache(self._db_path)
        self._options_loaded = False
        self._apply_corrected_text(field, final, old)
        self._start_options_load()
        QMessageBox.information(
            self,
            title,
            f"“{old}” → “{final}” olarak güncellendi.\n"
            "Öğrenilmiş kayıtlar düzeltildi.",
        )

    def _apply_corrected_text(self, field: str, final: str, old: str) -> None:
        if field == FIELD_PARENT:
            self.cmb_parent.set_text(final)
        elif field == FIELD_CHILD:
            self.cmb_child.set_text(final)
        elif field == FIELD_FAMILY:
            self.cmb_family.set_text(final)
        elif field == FIELD_COLOR:
            self.cmb_color.set_text(final)
        elif field == FIELD_BRAND:
            self.cmb_brand.set_text(final)
        elif field == FIELD_TAG:
            self._tags = [final if t == old else t for t in self._tags]
            if final not in self._tags:
                self._tags.append(final)
            self.txt_tag.clear()
            self._refresh_tags()

    def _start_options_load(self) -> None:
        """Kick background load. Never skip after invalidate — use req_id to drop stale."""
        self._options_req_id += 1
        req_id = self._options_req_id
        self.lbl_options_status.setText("Listeler hazırlanıyor…")
        self.lbl_options_status.setVisible(True)
        # Previous worker may still be running; its result is ignored via req_id.
        worker = _OptionsWorker(self._db_path, self)
        self._options_worker = worker
        worker.finished_ok.connect(
            lambda data, rid=req_id: self._on_options_loaded(data, req_id=rid)
        )
        worker.failed.connect(
            lambda msg, rid=req_id: self._on_options_failed(msg, req_id=rid)
        )
        worker.start()

    def _on_options_failed(self, _msg: str, req_id: int | None = None) -> None:
        if req_id is not None and req_id != self._options_req_id:
            return
        self.lbl_options_status.setText("Listeler yüklenemedi — yazmaya devam edebilirsiniz.")
        self._options_loaded = True

    def _on_options_loaded(self, data: dict, req_id: int | None = None) -> None:
        if req_id is not None and req_id != self._options_req_id:
            return
        self._options = dict(data or {})
        self._options_loaded = True
        kept = {
            "parent": self.cmb_parent.current_text(),
            "child": self.cmb_child.current_text(),
            "family": self.cmb_family.current_text(),
            "color": self.cmb_color.current_text(),
            "brand": self.cmb_brand.current_text(),
        }
        parents = list(self._options.get("parents") or [])
        self.cmb_parent.set_choices([(p, p) for p in parents])
        family = list(self._options.get("family") or [])
        self.cmb_family.set_choices(family)
        colors = list(self._options.get("colors") or [])
        self.cmb_color.set_choices(colors)
        brands = list(self._options.get("brands") or [])
        self.cmb_brand.set_choices([(b, b) for b in brands])
        try:
            from core.semantic_category_autocomplete import flatten_children_catalog

            catalog = flatten_children_catalog(
                self._options.get("children_by_parent") or {}
            )
            self.cmb_child.set_semantic_catalog(catalog)
            # Parent field: also surface path-qualified concept meanings
            self.cmb_parent.set_semantic_catalog(catalog)
        except Exception:
            self.cmb_child.set_semantic_catalog(None)
            self.cmb_parent.set_semantic_catalog(None)
        if kept["parent"]:
            self.cmb_parent.set_text(kept["parent"])
        self._reload_children()
        if kept["child"]:
            self.cmb_child.set_text(kept["child"])
        if kept["family"]:
            self.cmb_family.set_text(kept["family"])
        if kept["color"]:
            idx = self.cmb_color.findData(kept["color"])
            if idx >= 0:
                self.cmb_color.setCurrentIndex(idx)
            else:
                self.cmb_color.set_text(kept["color"])
        if kept["brand"]:
            self.cmb_brand.set_text(kept["brand"])
        # Merge path-segment tags once options are ready (merge only).
        self._apply_category_selection_sync(source="parent")
        self.lbl_options_status.setVisible(False)

    def _on_parent_changed(self, *_args) -> None:
        """Parent combo changed: sync path-qualified selection, then reload children."""
        if self._syncing_category:
            return
        self._apply_category_selection_sync(source="parent")

    def _on_child_changed(self, *_args) -> None:
        """Child combo changed: merge path segment tags only (do not wipe)."""
        if self._syncing_category:
            return
        self._apply_category_selection_sync(source="child")

    def _apply_category_selection_sync(self, source: str = "parent") -> None:
        """Split path-qualified Ana text into Ana/Alt; merge path tags. Re-entrancy safe."""
        if self._syncing_category:
            return
        self._syncing_category = True
        try:
            parent_text = self.cmb_parent.current_text()
            child_text = self.cmb_child.current_text()
            opts = self._options or {}
            parents = list(opts.get("parents") or [])
            by_parent = opts.get("children_by_parent") or {}

            if source == "child":
                # Child-only: merge path segment tags; do not wipe or re-split parent.
                if "/" in parent_text:
                    parent, child = split_category_selection(parent_text)
                    if not child:
                        child = child_text
                else:
                    parent, child = parent_text, child_text
                parent, child = resolve_parts_against_options(
                    parent, child, parents=parents, children_by_parent=by_parent
                )
                self._tags = merge_tags_preserve_manual(
                    self._tags, path_segment_tags(parent, child)
                )
                self._refresh_tags()
                return

            # source == parent: path-qualified or catalog display → split into Ana/Alt.
            looks_path = "/" in parent_text
            catalog_hit = False
            if not looks_path and parent_text:
                try:
                    catalog = getattr(self.cmb_parent, "_semantic_catalog", None) or []
                    pt = parent_text.strip()
                    for cand in catalog:
                        disp = str(getattr(cand, "display", "") or "").strip()
                        if disp and disp.casefold() == pt.casefold() and "/" in disp:
                            parent_text = disp
                            looks_path = True
                            catalog_hit = True
                            break
                except Exception:
                    pass

            state = sync_selection_state(
                parent_text,
                existing_tags=self._tags,
                parents=parents,
                children_by_parent=by_parent,
                child_text=None if looks_path else child_text,
            )
            new_parent = state["parent"]
            new_child = state["child"]

            self.cmb_parent.blockSignals(True)
            self.cmb_child.blockSignals(True)
            try:
                if new_parent != self.cmb_parent.current_text() or looks_path or catalog_hit:
                    self.cmb_parent.set_text(new_parent)
                self._reload_children()
                if new_child:
                    self.cmb_child.set_text(new_child)
                elif looks_path:
                    self.cmb_child.set_text("")
            finally:
                self.cmb_parent.blockSignals(False)
                self.cmb_child.blockSignals(False)

            self._tags = list(state["tags"])
            self._refresh_tags()
        finally:
            self._syncing_category = False

    def _reload_children(self) -> None:
        parent = self.cmb_parent.current_text()
        children: list[str] = []
        by_parent = self._options.get("children_by_parent") or {}
        if parent and parent in by_parent:
            children = list(by_parent.get(parent) or [])
        elif parent and by_parent:
            # Case / key tolerant lookup without sync DB scan on UI thread.
            from core.category_memory import _key

            pk = _key(parent)
            for p, kids in by_parent.items():
                if _key(p) == pk:
                    children = list(kids or [])
                    break
        kept = self.cmb_child.current_text()
        self.cmb_child.set_parent_hint(parent)
        self.cmb_child.set_choices([(c, c) for c in children])
        # Keep full catalog for cross-context meanings (e.g. Textile / Leopard)
        try:
            from core.semantic_category_autocomplete import flatten_children_catalog

            self.cmb_child.set_semantic_catalog(flatten_children_catalog(by_parent))
        except Exception:
            pass
        if kept:
            self.cmb_child.set_text(kept)

    def _clear_form(self) -> None:
        for combo in (
            self.cmb_parent,
            self.cmb_child,
            self.cmb_family,
            self.cmb_color,
            self.cmb_brand,
        ):
            combo.set_text("")
        self._tags = []
        self.txt_tag.clear()
        self._refresh_tags()

    def _prefill(self) -> None:
        self._clear_form()
        vals = form_values_from_record(self._result, self._overlay)
        parent = vals["parent"]
        child = vals["child"]
        if parent:
            self.cmb_parent.set_text(parent)
        if child:
            self.cmb_child.set_text(child)
        family = vals["pattern_family"]
        if family:
            self.cmb_family.set_text(family, family)
        color = vals["color_family"]
        if color:
            label = color_badge_text(color) or color
            self.cmb_color.set_text(label, color)
        brand = vals["brand"]
        if brand:
            self.cmb_brand.set_text(brand)
        self._tags = list(vals["tags"])
        self._refresh_tags()

    def _add_tag(self) -> None:
        tag = self.txt_tag.text().strip()
        if not tag:
            return
        if tag not in self._tags:
            self._tags.append(tag)
        self.txt_tag.clear()
        self._refresh_tags()

    def _refresh_tags(self) -> None:
        self.lbl_tags.setText(", ".join(self._tags) if self._tags else "—")

    def _remember_new_values(self, overlay: dict) -> None:
        db = self._db_path
        parent = overlay.get("parent") or ""
        child = overlay.get("child") or ""
        if parent and not self.cmb_parent.is_known():
            register_root_category(db, parent)
        if parent and child and not self.cmb_child.is_known():
            register_category(db, parent, child)
        family_txt = self.cmb_family.current_text()
        if family_txt and not self.cmb_family.is_known():
            register_category(db, "Desen ailesi", family_txt)
        color_txt = self.cmb_color.current_text()
        if color_txt and not self.cmb_color.is_known():
            register_category(db, "Renk", color_txt)
        brand = overlay.get("brand") or ""
        if brand and not self.cmb_brand.is_known():
            register_category(db, "Marka", brand)
        for tag in overlay.get("tags") or []:
            register_category(db, "Etiket", str(tag))
        invalidate_edit_options_cache(db)

    def values(self) -> dict:
        parent_raw = self.cmb_parent.current_text()
        child_raw = self.cmb_child.current_text()
        opts = self._options or {}
        if "/" in parent_raw:
            parent, child = split_category_selection(parent_raw)
            # If child combo also has a value and parent split already has child,
            # prefer the split (path-qualified parent is authoritative).
            if not child and child_raw:
                child = child_raw
        else:
            parent, child = parent_raw, child_raw
        parent, child = resolve_parts_against_options(
            parent,
            child,
            parents=list(opts.get("parents") or []),
            children_by_parent=opts.get("children_by_parent") or {},
        )
        path = canonical_path_from_parts(parent, child)
        # Ensure tags include path segments (merge only — never wipe manual).
        self._tags = merge_tags_preserve_manual(
            self._tags, path_segment_tags(parent, child)
        )
        fields = pattern_fields_for_path(path) if path else {}
        family = self.cmb_family.current_data_or_text()
        if not family:
            family = fields.get("pattern_family", "")
        # Alt kategori değişince eski "Leopard / Animal" serbest metni
        # taksonomi anahtarını ezmesin.
        tax_family = str(fields.get("pattern_family") or "").strip()
        if tax_family and (
            not family
            or "/" in str(family)
            or str(family).strip().casefold()
            in {"leopard / animal", "tiger / animal", "animal print"}
        ):
            family = tax_family
        color = self.cmb_color.current_data_or_text()
        if color and color == self.cmb_color.current_text():
            # typed display label → known key
            rev = {lab: key for key, lab in COLOR_LABELS_TR.items() if lab}
            color = rev.get(color, color)
        brand = self.cmb_brand.current_text()
        return {
            "parent": parent,
            "child": child,
            "category_path": path,
            "pattern_family": family,
            "animal_print_type": fields.get("animal_print_type", ""),
            "color_family": color,
            "brand": brand,
            "tags": list(self._tags),
        }

    def teach_apply(self) -> bool:
        return self._teach_apply

    def _on_save(self) -> None:
        overlay = self.values()
        self._remember_new_values(overlay)
        self._teach_apply = False
        self.accept()

    def _on_teach(self) -> None:
        overlay = self.values()
        self._remember_new_values(overlay)
        self._teach_apply = True
        self.accept()
