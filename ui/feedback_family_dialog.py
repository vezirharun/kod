"""Yanlış eşleşme sonrası — görselin doğru grubunu öğretme diyalogu."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.category_tree import (
    category_path,
    child_categories,
    parent_categories,
    resolve_category_query,
)
from core.category_memory import register_category, register_root_category
from core.textile_terms import normalize_turkish
from core.similarity_tiers import SIMILARITY_TIER_CHOICES, resolve_similarity_tier
from core.brand_aliases import brand_suggestions
from core.textile_terms import FAMILY_UI_CHOICES, resolve_user_family_label


class FeedbackFamilyDialog(QDialog):
    """Kullanıcıdan doğru desen ailesini veya benzerlik sınıfını alır."""

    def __init__(
        self,
        parent=None,
        *,
        filename: str = "",
        query_family: str = "",
        learned_shortcuts: list[dict[str, str]] | None = None,
        db_path: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Doğru grup nedir?")
        self.setMinimumWidth(480)
        self.setMinimumHeight(420)
        self.resize(500, 560)
        self._selected_family = ""
        self._custom_text = ""
        self._pattern_subtype = ""
        self._similarity_tier = ""
        self._tier_label = ""
        self._cluster_group = ""
        self._picked_from_shortcut = False
        self._typed_on_accept = False
        self._parent_category = ""
        self._child_category = ""
        self._learned_shortcuts = list(learned_shortcuts or [])
        self._db_path = str(db_path or "")
        self._entity_attributes: list[dict[str, str]] = []
        self._build_ui(filename, query_family)
        from ui.responsive_dialog import apply_responsive_dialog

        apply_responsive_dialog(
            self, min_width=420, min_height=320, prefer_width=500, prefer_height=560
        )

    def _build_ui(self, filename: str, query_family: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll_body = QWidget()
        body = QVBoxLayout(scroll_body)
        body.setSpacing(8)

        intro = (
            "Bu sonuç sorgunuzla eşleşmemeli.\n"
            "Doğru kategoriyi seçin — yalnızca birebir kopyalara yayılır."
        )
        if filename:
            intro += f"\n\nDosya: {filename}"
        if query_family and query_family not in ("", "unknown"):
            intro += f"\n(Sorgu ailesi: {query_family})"
        lbl = QLabel(intro)
        lbl.setWordWrap(True)
        body.addWidget(lbl)

        # Kategori arama: 100+ seçenek içinde kaydırarak aramak yerine
        # "dolce gabbana", "bmw", "yavru kedi", "logo" gibi ifadeyi doğrudan
        # ontoloji/kategori ağacında bulur.
        body.addWidget(QLabel("Kategori ara / doğrudan seç:"))
        search_row = QHBoxLayout()
        self.txt_category_search = QLineEdit()
        self.txt_category_search.setPlaceholderText(
            "örn: Dolce Gabbana, logo, BMW, araba, yavru kedi, masa…"
        )
        self.txt_category_search.returnPressed.connect(self._find_and_select_category)
        search_row.addWidget(self.txt_category_search, stretch=1)
        self.btn_find_category = QPushButton("Bul ve seç")
        self.btn_find_category.clicked.connect(self._find_and_select_category)
        search_row.addWidget(self.btn_find_category)
        body.addLayout(search_row)
        self.lbl_category_search = QLabel("")
        self.lbl_category_search.setWordWrap(True)
        self.lbl_category_search.setStyleSheet("color:#64748b;font-size:11px;")
        body.addWidget(self.lbl_category_search)

        self.brand_candidates = QListWidget()
        self.brand_candidates.setMaximumHeight(150)
        self.brand_candidates.setVisible(False)
        self.brand_candidates.itemClicked.connect(self._pick_brand_candidate)
        body.addWidget(self.brand_candidates)

        self.btn_new_brand = QPushButton("+ Yeni marka olarak ekle")
        self.btn_new_brand.setVisible(False)
        self.btn_new_brand.setMinimumHeight(34)
        self.btn_new_brand.setStyleSheet("font-weight:700;color:#ffffff;background:#2563eb;")
        self.btn_new_brand.clicked.connect(self._add_typed_brand)
        body.addWidget(self.btn_new_brand)

        self.btn_new_category = QPushButton("+ Yeni ana kategori olarak ekle")
        self.btn_new_category.setVisible(False)
        self.btn_new_category.setMinimumHeight(34)
        self.btn_new_category.clicked.connect(self._add_typed_category)
        body.addWidget(self.btn_new_category)

        cat_row = QHBoxLayout()
        self.cmb_parent = QComboBox()
        self.cmb_parent.addItem("— Ana kategori —", "")
        for p in parent_categories(self._db_path):
            self.cmb_parent.addItem(p, p)
        self.cmb_parent.currentIndexChanged.connect(self._on_parent_changed)
        cat_row.addWidget(self.cmb_parent, stretch=1)
        self.cmb_child = QComboBox()
        self.cmb_child.addItem("— Alt kategori —", "")
        self.cmb_child.currentIndexChanged.connect(self._on_child_changed)
        cat_row.addWidget(self.cmb_child, stretch=1)
        body.addLayout(cat_row)

        body.addWidget(QLabel("Nitelik ara / yeni nitelik:"))
        attr_row = QHBoxLayout()
        self.txt_attribute_search = QLineEdit()
        self.txt_attribute_search.setPlaceholderText("örn: boya lekesi, fırça, sıçrama…")
        self.txt_attribute_search.returnPressed.connect(self._add_attribute_from_search)
        attr_row.addWidget(self.txt_attribute_search, stretch=1)
        self.btn_find_attribute = QPushButton("Nitelik ekle")
        self.btn_find_attribute.clicked.connect(self._add_attribute_from_search)
        attr_row.addWidget(self.btn_find_attribute)
        body.addLayout(attr_row)
        self.btn_new_attribute = QPushButton("")
        self.btn_new_attribute.setVisible(False)
        self.btn_new_attribute.setMinimumHeight(34)
        self.btn_new_attribute.setStyleSheet("font-weight:700;color:#ffffff;background:#0f766e;")
        self.btn_new_attribute.clicked.connect(self._add_typed_attribute)
        body.addWidget(self.btn_new_attribute)
        self.lst_attributes = QListWidget()
        self.lst_attributes.setMaximumHeight(90)
        self.lst_attributes.setVisible(False)
        self.lst_attributes.itemClicked.connect(
            lambda item: self._remove_attribute_path(
                str(item.data(Qt.ItemDataRole.UserRole) or "")
            )
        )
        body.addWidget(self.lst_attributes)

        self.lbl_mixed_animal = QLabel("Karışım bileşenleri")
        self.lbl_mixed_animal.setVisible(False)
        body.addWidget(self.lbl_mixed_animal)
        self.lst_mixed_animal = QListWidget()
        self.lst_mixed_animal.setMaximumHeight(120)
        self.lst_mixed_animal.setVisible(False)
        self.lst_mixed_animal.itemChanged.connect(self._on_mixed_animal_item_changed)
        body.addWidget(self.lst_mixed_animal)

        self.chk_exact = QCheckBox("Exact kopyalara uygula (önerilen)")
        self.chk_exact.setChecked(True)
        body.addWidget(self.chk_exact)
        body.addWidget(QLabel("Benzerlik sınıfı (isteğe bağlı):"))
        tier_grid = QGridLayout()
        tier_grid.setHorizontalSpacing(6)
        tier_grid.setVerticalSpacing(6)
        for i, (label, tier_id, cluster, family, subtype) in enumerate(
            SIMILARITY_TIER_CHOICES
        ):
            btn = QPushButton(label)
            btn.setMinimumHeight(32)
            btn.clicked.connect(
                lambda checked=False, tid=tier_id, lbl=label, cl=cluster, fam=family, sub=subtype: (
                    self._pick_tier(tid, lbl, cl, fam, sub)
                ),
            )
            tier_grid.addWidget(btn, i // 2, i % 2)
        body.addLayout(tier_grid)

        if self._learned_shortcuts:
            body.addWidget(QLabel("Kayıtlı kategorileriniz:"))
            learned_grid = QGridLayout()
            learned_grid.setHorizontalSpacing(6)
            learned_grid.setVerticalSpacing(6)
            for i, entry in enumerate(self._learned_shortcuts):
                btn = QPushButton(entry.get("label") or entry.get("tag", ""))
                btn.setMinimumHeight(32)
                btn.setStyleSheet("font-weight:600;")
                btn.clicked.connect(
                    lambda checked=False, e=entry: self._pick_shortcut(e),
                )
                learned_grid.addWidget(btn, i // 2, i % 2)
            body.addLayout(learned_grid)

        body.addWidget(QLabel("Desen ailesi (isteğe bağlı):"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for i, (label, family_id) in enumerate(FAMILY_UI_CHOICES):
            btn = QPushButton(label)
            btn.setMinimumHeight(32)
            btn.clicked.connect(lambda checked=False, f=family_id: self._pick_family(f))
            grid.addWidget(btn, i // 2, i % 2)
        body.addLayout(grid)

        body.addWidget(QLabel("veya yeni sınıf/kategori yazın:"))
        self.txt_custom = QLineEdit()
        self.txt_custom.setPlaceholderText("örn: Benzer Kamuflaj, ekose, ahtapot…")
        self.txt_custom.returnPressed.connect(self._accept_custom)
        self.txt_custom.textChanged.connect(self._on_custom_text_changed)
        body.addWidget(self.txt_custom)

        self.lbl_pick = QLabel("")
        self.lbl_pick.setWordWrap(True)
        self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        body.addWidget(self.lbl_pick)
        body.addStretch(1)

        scroll.setWidget(scroll_body)
        layout.addWidget(scroll, stretch=1)

        self.chk_remember = QCheckBox("Bu kategoriyi kayıtlı gruplara kaydet")
        self.chk_remember.setChecked(True)
        self.chk_remember.setVisible(False)
        self.chk_remember.setStyleSheet("font-weight:600;color:#1d4ed8;")
        layout.addWidget(self.chk_remember)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Öğret ve Uygula")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _find_and_select_category(self) -> None:
        text = self.txt_category_search.text().strip()
        if not text:
            self.lbl_category_search.setText("Bir kategori adı yazın.")
            return
        match = resolve_category_query(text, db_path=self._db_path)
        # Önce kesin eşleşme. Marka için yazım hatası/kısaltma varsa otomatik
        # seçmek yerine adayları kullanıcıya göster.
        suggestions = brand_suggestions(text, limit=8, db_path=self._db_path)
        self.brand_candidates.clear()
        self.brand_candidates.setVisible(bool(suggestions))
        self.btn_new_brand.setVisible(False)
        self.btn_new_category.setVisible(False)
        for name, score in suggestions:
            item = QListWidgetItem(f"{name.title()}  — %{round(score * 100)} eşleşme")
            item.setData(Qt.ItemDataRole.UserRole, name.title())
            self.brand_candidates.addItem(item)
        if suggestions and not match.category_path:
            self.btn_new_brand.setText(f"+ Yazdığım adı yeni marka olarak ekle: {text.title()}")
            self.btn_new_brand.setVisible(True)
            self.lbl_category_search.setText("Marka adayları bulundu. Birini seçin veya yazdığınız adı yeni marka olarak ekleyin.")
            self.lbl_category_search.setStyleSheet("color:#15803d;font-size:11px;font-weight:600;")
            return
        if not match.category_path:
            qn = normalize_turkish(text)
            for p in parent_categories(self._db_path):
                if normalize_turkish(p) == qn:
                    parent_idx = self.cmb_parent.findData(p)
                    if parent_idx >= 0:
                        self.cmb_parent.setCurrentIndex(parent_idx)
                    self.cmb_child.setCurrentIndex(0)
                    self._parent_category = p
                    self._child_category = ""
                    self.brand_candidates.setVisible(False)
                    self.btn_new_brand.setVisible(False)
                    self.btn_new_category.setVisible(False)
                    self.lbl_category_search.setText(f"Bulundu ve seçildi: {p}")
                    self.lbl_category_search.setStyleSheet(
                        "color:#15803d;font-size:11px;font-weight:600;"
                    )
                    self.lbl_pick.setText(f"Kategori: {p}")
                    self._update_remember_visibility()
                    return
            self.btn_new_brand.setText(f"+ Yeni marka olarak ekle: {text.title()}")
            self.btn_new_brand.setVisible(True)
            self.btn_new_category.setText(f"+ Yeni ana kategori olarak ekle: {text}")
            self.btn_new_category.setVisible(True)
            self.lbl_category_search.setText(
                "Marka kaynak listesinde yok. İsterseniz bu adı yeni marka olarak ekleyebilirsiniz."
            )
            self.lbl_category_search.setStyleSheet("color:#b45309;font-size:11px;")
            return

        self.brand_candidates.setVisible(False)
        self.btn_new_brand.setVisible(False)
        parent = match.category_path.split("/", 1)[0]
        child = match.category_path.split("/", 1)[1] if "/" in match.category_path else ""
        parent_idx = self.cmb_parent.findData(parent)
        if parent_idx < 0:
            self.lbl_category_search.setText(f"Kategori ağacında bulunamadı: {match.category_path}")
            self.lbl_category_search.setStyleSheet("color:#b45309;font-size:11px;")
            return
        self.cmb_parent.setCurrentIndex(parent_idx)
        if child:
            child_idx = self.cmb_child.findData(child)
            if child_idx >= 0:
                self.cmb_child.setCurrentIndex(child_idx)
        self.lbl_category_search.setText(f"Bulundu ve seçildi: {match.category_path}")
        self.lbl_category_search.setStyleSheet("color:#15803d;font-size:11px;font-weight:600;")
        self._update_remember_visibility()

    def _pick_brand_candidate(self, item: QListWidgetItem) -> None:
        brand = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not brand:
            return
        self._select_brand(brand)

    def _select_brand(self, brand: str) -> None:
        brand = " ".join(str(brand or "").strip().split())
        if not brand:
            return
        parent_idx = self.cmb_parent.findData("Marka")
        if parent_idx < 0:
            return
        self.cmb_parent.setCurrentIndex(parent_idx)

        # Yeni marka sabit CATEGORY_TREE içinde bulunmasa bile gerçek bir
        # Marka/<isim> yolu oluşturulabilsin. Combo kutusuna çalışma anında
        # ekliyoruz; böylece category_path_value() da doğru yolu döndürür ve
        # öğretme/kaydetme tarafı yalnızca "Marka" olarak kaydetmez.
        child_idx = self.cmb_child.findData(brand)
        if child_idx < 0:
            self.cmb_child.addItem(brand, brand)
            child_idx = self.cmb_child.findData(brand)
        if child_idx >= 0:
            self.cmb_child.setCurrentIndex(child_idx)
        self._parent_category = "Marka"
        self._child_category = brand
        self._custom_text = ""
        self.txt_custom.blockSignals(True)
        self.txt_custom.clear()
        self.txt_custom.blockSignals(False)
        self.brand_candidates.setVisible(False)
        self.btn_new_brand.setVisible(False)
        self.lbl_category_search.setText(f"Seçildi: Marka/{brand}")
        self.lbl_category_search.setStyleSheet("color:#15803d;font-size:11px;font-weight:600;")
        self.lbl_pick.setText(f"Kategori: Marka/{brand}")
        self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        self._update_remember_visibility()

    def _add_typed_brand(self) -> None:
        text = self.txt_category_search.text().strip()
        if text:
            self._select_brand(text)

    def _add_typed_category(self) -> None:
        text = " ".join(self.txt_category_search.text().strip().split())
        if not text:
            return
        label = text[:1].upper() + text[1:] if text else text
        if self._db_path:
            register_root_category(self._db_path, label, aliases=[text, label])
        if self.cmb_parent.findData(label) < 0:
            self.cmb_parent.addItem(label, label)
        idx = self.cmb_parent.findData(label)
        if idx >= 0:
            self.cmb_parent.setCurrentIndex(idx)
        self._parent_category = label
        self._child_category = ""
        self.btn_new_category.setVisible(False)
        self.btn_new_brand.setVisible(False)
        self.lbl_category_search.setText(f"Yeni ana kategori: {label}")
        self.lbl_pick.setText(f"Kategori: {label}")
        self._update_remember_visibility()

    def entity_attributes(self) -> list[dict[str, str]]:
        return list(self._entity_attributes)

    def _refresh_attribute_list(self) -> None:
        self.lst_attributes.blockSignals(True)
        self.lst_attributes.clear()
        for attr in self._entity_attributes:
            item = QListWidgetItem(str(attr.get("path") or attr.get("label") or ""))
            item.setData(Qt.ItemDataRole.UserRole, attr.get("path", ""))
            self.lst_attributes.addItem(item)
        self.lst_attributes.blockSignals(False)
        self.lst_attributes.setVisible(bool(self._entity_attributes))

    def _add_attribute_path(self, path: str, label: str, silent: bool = False) -> None:
        path = str(path or "").strip()
        label = str(label or "").strip()
        if not path:
            return
        if any(a.get("path") == path for a in self._entity_attributes):
            return
        parent, _, child = path.partition("/")
        self._entity_attributes.append({
            "path": path,
            "parent": parent,
            "label": child or label,
            "source": label,
        })
        if not silent:
            self._refresh_attribute_list()

    def _remove_attribute_path(self, path: str) -> None:
        path = str(path or "").strip()
        if not path:
            return
        self._entity_attributes = [
            a for a in self._entity_attributes if a.get("path") != path
        ]
        self._refresh_attribute_list()

    def _add_attribute_from_search(self) -> None:
        text = self.txt_attribute_search.text().strip()
        if not text:
            self.btn_new_attribute.setVisible(False)
            return
        match = resolve_category_query(text, db_path=self._db_path)
        path = match.category_path if match.category_path.startswith("Nitelik/") else ""
        if path:
            self._add_attribute_path(path, text)
            self.btn_new_attribute.setVisible(False)
            return
        self.btn_new_attribute.setText(f"+ {text} yeni nitelik olarak ekle")
        self.btn_new_attribute.setVisible(True)
        # Qt: child isVisible() is false until an ancestor window is shown.
        if not self.isVisible():
            self.show()

    def _add_typed_attribute(self) -> None:
        text = self.txt_attribute_search.text().strip()
        if not text:
            return
        if self._db_path:
            register_category(self._db_path, "Nitelik", text, aliases=[text])
            try:
                from core.concept_registry import learn

                learn(
                    self._db_path,
                    text,
                    parent="Nitelik",
                    concept_type="custom_tag",
                    aliases=[text],
                )
            except Exception:
                pass
        self._add_attribute_path(f"Nitelik/{text}", text)
        self.btn_new_attribute.setVisible(False)
        self.txt_attribute_search.clear()

    def _refresh_mixed_animal_picker(self) -> None:
        show = (
            self._parent_category == "Animal Print"
            and self._child_category == "Mixed Animal Print"
        )
        self.lbl_mixed_animal.setVisible(show)
        self.lst_mixed_animal.setVisible(show)
        if not show:
            return
        self.lst_mixed_animal.blockSignals(True)
        self.lst_mixed_animal.clear()
        for child in child_categories("Animal Print", self._db_path):
            if child == "Mixed Animal Print":
                continue
            item = QListWidgetItem(child)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, child)
            self.lst_mixed_animal.addItem(item)
        self.lst_mixed_animal.blockSignals(False)

    def _on_mixed_animal_item_changed(self, item: QListWidgetItem) -> None:
        child = str(item.data(Qt.ItemDataRole.UserRole) or item.text() or "").strip()
        if not child:
            return
        path = f"Animal Print/{child}"
        if item.checkState() == Qt.CheckState.Checked:
            self._add_attribute_path(path, item.text(), silent=True)
            self._refresh_attribute_list()
        else:
            self._remove_attribute_path(path)

    def _on_parent_changed(self) -> None:
        parent = self.cmb_parent.currentData() or ""
        self._parent_category = parent
        self.cmb_child.blockSignals(True)
        self.cmb_child.clear()
        self.cmb_child.addItem("— Alt kategori —", "")
        for child in child_categories(parent, self._db_path):
            self.cmb_child.addItem(child, child)
        self.cmb_child.blockSignals(False)
        self._child_category = ""
        if parent:
            self.lbl_pick.setText(f"Kategori: {parent}")
            self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
            self._update_remember_visibility()
        self._refresh_mixed_animal_picker()

    def _on_child_changed(self) -> None:
        self._child_category = self.cmb_child.currentData() or ""
        if self._parent_category and self._child_category:
            path = category_path(self._parent_category, self._child_category)
            self.lbl_pick.setText(f"Kategori: {path}")
            self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
            self._update_remember_visibility()
        self._refresh_mixed_animal_picker()

    def _on_custom_text_changed(self, text: str) -> None:
        tag = text.strip()
        if tag:
            self.lbl_pick.setText(f"Yeni etiket: {tag}")
            self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        elif (
            not self._selected_family
            and not self._similarity_tier
            and not self._parent_category
        ):
            self.lbl_pick.clear()
        self._update_remember_visibility()

    def _update_remember_visibility(self) -> None:
        tag = self.txt_custom.text().strip()
        has_combo = bool(self._parent_category)
        show = bool(tag) or (has_combo and not self._picked_from_shortcut)
        self.chk_remember.setVisible(show)
        if show and tag:
            self.chk_remember.setChecked(True)

    def _sync_category_from_combos(self) -> None:
        self._parent_category = self.cmb_parent.currentData() or ""
        self._child_category = self.cmb_child.currentData() or ""

    def propagate_exact(self) -> bool:
        return self.chk_exact.isChecked()

    def parent_category(self) -> str:
        return self._parent_category

    def child_category(self) -> str:
        return self._child_category

    def category_path_value(self) -> str:
        if self._parent_category:
            return category_path(self._parent_category, self._child_category)
        return ""

    def _pick_tier(
        self,
        tier_id: str,
        label: str,
        cluster_group: str,
        pattern_family: str,
        pattern_subtype: str,
    ) -> None:
        self._similarity_tier = tier_id
        self._tier_label = label
        self._cluster_group = cluster_group
        if pattern_family:
            self._selected_family = pattern_family
        if pattern_subtype:
            self._pattern_subtype = pattern_subtype
        self._picked_from_shortcut = False
        self.txt_custom.blockSignals(True)
        self.txt_custom.clear()
        self.txt_custom.blockSignals(False)
        self.lbl_pick.setText(f"Seçildi: {label}")
        self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        self.chk_remember.setVisible(False)

    def _pick_family(self, family_id: str) -> None:
        self._selected_family = family_id
        self._custom_text = ""
        if not self._similarity_tier:
            self._pattern_subtype = ""
        self._picked_from_shortcut = False
        self.txt_custom.blockSignals(True)
        self.txt_custom.clear()
        self.txt_custom.blockSignals(False)
        label = next((l for l, f in FAMILY_UI_CHOICES if f == family_id), family_id)
        extra = f" + {self._tier_label}" if self._tier_label else ""
        self.lbl_pick.setText(f"Seçildi: {label}{extra}")
        self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        self.chk_remember.setVisible(False)

    def _pick_shortcut(self, entry: dict[str, str]) -> None:
        tier_id = entry.get("tier_id", "")
        if tier_id:
            resolved = resolve_similarity_tier(tier_id) or {}
            self._similarity_tier = tier_id
            self._tier_label = entry.get("label") or resolved.get("label", "")
            self._cluster_group = entry.get("cluster_group") or resolved.get(
                "cluster_group", ""
            )
        self._selected_family = entry.get("pattern_family", "")
        raw_tag = (entry.get("tag", "") or "").strip()
        self._custom_text = raw_tag
        self._pattern_subtype = entry.get("pattern_subtype", "")
        self._picked_from_shortcut = True

        # Öğrenilmiş kategori kısayolu "Brand/Dolce Gabbana" gibi bir yol
        # taşıyorsa tekrar açıldığında yalnızca yazı olarak kalmasın; gerçek
        # ana/alt kategori seçimlerini de geri yükle.
        shortcut_path = raw_tag if "/" in raw_tag else ""
        if shortcut_path:
            parent, _, child = shortcut_path.partition("/")
            parent_idx = self.cmb_parent.findData(parent)
            if parent_idx >= 0:
                self.cmb_parent.setCurrentIndex(parent_idx)
                child_idx = self.cmb_child.findData(child)
                if child and child_idx >= 0:
                    self.cmb_child.setCurrentIndex(child_idx)
                self._parent_category = parent
                self._child_category = child
                self._custom_text = ""

        self.txt_custom.blockSignals(True)
        self.txt_custom.clear()
        self.txt_custom.blockSignals(False)
        label = entry.get("label") or entry.get("tag", "")
        self.lbl_pick.setText(f"Seçildi: {label}")
        self.lbl_pick.setStyleSheet("color:#2a6b2a;font-weight:600;")
        self.chk_remember.setVisible(False)

    def _accept_custom(self) -> None:
        self._on_accept()

    def _on_accept(self) -> None:
        text = self.txt_custom.text().strip()
        if text:
            tier = resolve_similarity_tier(text)
            if tier.get("tier_id"):
                self._similarity_tier = tier["tier_id"]
                self._tier_label = tier.get("label", text)
                self._cluster_group = tier.get("cluster_group", "")
                if tier.get("pattern_family"):
                    self._selected_family = tier["pattern_family"]
                if tier.get("pattern_subtype"):
                    self._pattern_subtype = tier["pattern_subtype"]
                self._custom_text = tier.get("tag", text)
            else:
                resolved = resolve_user_family_label(text)
                self._custom_text = resolved.get("tag", text)
                if resolved.get("pattern_family"):
                    self._selected_family = resolved["pattern_family"]
                if resolved.get("pattern_subtype"):
                    self._pattern_subtype = resolved["pattern_subtype"]
            self._typed_on_accept = True
            self._picked_from_shortcut = False
        self._sync_category_from_combos()
        if (
            not self._similarity_tier
            and not self._selected_family
            and not self._custom_text
            and not self._parent_category
            and not self._entity_attributes
        ):
            self.lbl_pick.setText("Lütfen bir sınıf veya grup seçin / yazın.")
            self.lbl_pick.setStyleSheet("color:#b45309;font-weight:600;")
            return
        if self._similarity_tier and not self._tier_label:
            self._tier_label = next(
                (
                    lbl
                    for lbl, tid, *_ in SIMILARITY_TIER_CHOICES
                    if tid == self._similarity_tier
                ),
                self._similarity_tier,
            )
        self.accept()

    def selected_family(self) -> str:
        return self._selected_family

    def brand_query_text(self) -> str:
        return self.txt_category_search.text().strip()

    def custom_tag(self) -> str:
        return self._custom_text

    def pattern_subtype(self) -> str:
        return self._pattern_subtype

    def similarity_tier(self) -> str:
        return self._similarity_tier

    def tier_label(self) -> str:
        return self._tier_label

    def cluster_group(self) -> str:
        return self._cluster_group

    def should_remember_shortcut(self) -> bool:
        if self.chk_remember.isHidden() or not self.chk_remember.isChecked():
            return False
        return bool(self._custom_text.strip() or self._parent_category)

    def shortcut_entry(self) -> dict[str, str]:
        label = self._tier_label or self._custom_text
        path = self.category_path_value()
        if path:
            if self._custom_text:
                label = f"{path} — {self._custom_text}"
            else:
                label = path
        tag = self._custom_text or self._tier_label or path or label
        return {
            "label": label,
            "tag": tag,
            "tier_id": self._similarity_tier,
            "cluster_group": self._cluster_group,
            "pattern_family": self._selected_family,
            "pattern_subtype": self._pattern_subtype,
        }

    def resolved_label(self) -> str:
        if self._tier_label:
            if self._selected_family and self._custom_text:
                return f"{self._tier_label} — {self._custom_text}"
            return self._tier_label
        for entry in self._learned_shortcuts:
            if entry.get("tag") == self._custom_text and self._picked_from_shortcut:
                return entry.get("label") or self._custom_text
        if self._selected_family:
            label = next(
                (l for l, f in FAMILY_UI_CHOICES if f == self._selected_family),
                self._selected_family,
            )
            if self._custom_text and label != self._custom_text:
                return f"{label} ({self._custom_text})"
            return label
        path = self.category_path_value()
        if path:
            return path
        return self._custom_text
