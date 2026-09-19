"""AI & Index ayarları — basit / gelişmiş mod."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.capability_check import (
    AI_FALLBACK_MSG,
    OCR_FALLBACK_MSG,
    probe_ai_dependencies,
    probe_ocr_dependencies,
)
from core.settings import AppSettings


class AiSettingsPanel(QWidget):
    settings_changed = Signal()
    reindex_requested = Signal()
    ai_enable_warning = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._block = False
        self._build_ui()
        self.load_from_settings()
        self.set_advanced_mode(settings.ui_mode == "advanced")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self.grp_toggles = QGroupBox("Özellik Anahtarları")
        tog = QVBoxLayout(self.grp_toggles)
        self.btn_ai_toggle = QPushButton("Yapay zekâ vektörü: KAPALI")
        self.btn_ai_toggle.setCheckable(True)
        self.btn_ai_toggle.setMinimumHeight(40)
        self.btn_ai_toggle.setToolTip(
            "DINOv2 + CLIP vektör üretimini aç/kapat (görsel benzerlik araması)."
        )
        self.btn_ocr_toggle = QPushButton("OCR: KAPALI")
        self.btn_ocr_toggle.setCheckable(True)
        self.btn_ocr_toggle.setMinimumHeight(40)
        self.btn_ocr_toggle.setToolTip(
            "Görsel içindeki yazıların OCR ile okunmasını aç/kapat."
        )
        self.lbl_toggle_hint = QLabel("")
        self.lbl_toggle_hint.setWordWrap(True)
        self.lbl_toggle_hint.setStyleSheet("color:#94a3b8;font-size:10px;")
        tog.addWidget(self.btn_ai_toggle)
        tog.addWidget(self.btn_ocr_toggle)
        tog.addWidget(self.lbl_toggle_hint)
        layout.addWidget(self.grp_toggles)

        self.grp_simple = QGroupBox("Hızlı Ayarlar")
        simple_l = QVBoxLayout(self.grp_simple)
        self.chk_ai_simple = QCheckBox("Yapay zekâ araması açık")
        self.chk_ocr_simple = QCheckBox("OCR Açık")
        self.lbl_ocr_warning = QLabel("")
        self.lbl_ocr_warning.setWordWrap(True)
        self.lbl_ocr_warning.setStyleSheet("color:#94a3b8;font-size:10px;")
        self.lbl_ocr_warning.setVisible(False)
        self.btn_ocr_install = QPushButton("OCR Kurulumu")
        self.btn_ocr_install.setVisible(False)
        self.btn_ocr_install.clicked.connect(self._show_ocr_install_help)
        self.btn_reindex_simple = QPushButton("Yeniden indeksle")
        self.btn_reindex_simple.clicked.connect(self.reindex_requested.emit)
        simple_l.addWidget(self.chk_ai_simple)
        simple_l.addWidget(self.chk_ocr_simple)
        simple_l.addWidget(self.lbl_ocr_warning)
        simple_l.addWidget(self.btn_ocr_install)
        simple_l.addWidget(self.btn_reindex_simple)
        layout.addWidget(self.grp_simple)

        self.grp_advanced = QGroupBox("Yapay Zekâ ve İndeks — Gelişmiş")
        adv = QVBoxLayout(self.grp_advanced)
        self.chk_ai = QCheckBox("Yapay zekâ vektörü kullan (DINOv2 + CLIP)")
        self.chk_index_ai = QCheckBox("İndeks sırasında yapay zekâ hesapla")
        self.chk_ocr = QCheckBox("OCR kullan (dosya adı + görsel yazı)")
        self.chk_auto_patch = QCheckBox("AI Final sonrası PATCH otomatik")
        self.chk_auto_ocr = QCheckBox("AI Final sonrası OCR otomatik (arama sırasında değil)")
        self.chk_fine = QCheckBox("Ayrıntılı yeniden sıralama (ilk K sonuç)")
        self.chk_text_visual = QCheckBox("Metin aramada görsel benzerlik kullan")
        self.chk_live_scores = QCheckBox("Anlık puan güncelle")
        self.chk_ai_live = QCheckBox("Yapay zekâ geldikçe sonuçları yenile")
        self.chk_lock_ranking = QCheckBox("Sonuç sıralamasını kilitle")
        self.chk_format_panel = QCheckBox("Format durum panelini göster")
        self.chk_category_panel = QCheckBox("Sol kategori ağacı filtresini göster")
        for w in (
            self.chk_ai,
            self.chk_index_ai,
            self.chk_ocr,
            self.chk_auto_patch,
            self.chk_auto_ocr,
            self.chk_fine,
            self.chk_text_visual,
            self.chk_live_scores,
            self.chk_ai_live,
            self.chk_lock_ranking,
            self.chk_format_panel,
            self.chk_category_panel,
        ):
            adv.addWidget(w)

        row = QHBoxLayout()
        row.addWidget(QLabel("Ayrıntılı sıralama ilk K:"))
        self.spin_fine_k = QSpinBox()
        self.spin_fine_k.setRange(20, 300)
        self.spin_fine_k.setValue(80)
        row.addWidget(self.spin_fine_k)
        row.addStretch()
        adv.addLayout(row)

        net_row = QHBoxLayout()
        net_row.addWidget(QLabel("Ağ okuma işleyicisi:"))
        self.cmb_net_workers = QComboBox()
        self.cmb_net_workers.addItem("2 (varsayılan)", 2)
        self.cmb_net_workers.addItem("4 (test)", 4)
        self.cmb_net_workers.addItem("6 (test)", 6)
        self.cmb_net_workers.setToolTip(
            "NAS'tan eşzamanlı okuma sayısı. Ağ bant genişliği yeterliyse "
            "artırmak hızlı index'i hızlandırabilir; yavaş ağda 2'de kalın."
        )
        net_row.addWidget(self.cmb_net_workers)
        net_row.addStretch()
        adv.addLayout(net_row)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Çalışma modu:"))
        self.cmb_work_profile = QComboBox()
        self.cmb_work_profile.addItem("Arka Plan (önerilen)", "background")
        self.cmb_work_profile.addItem("Dengeli", "balanced")
        self.cmb_work_profile.addItem("Maksimum Hız", "max_speed")
        self.cmb_work_profile.setToolTip(
            "Arka Plan: Photoshop/Explorer ile birlikte güvenli çalışır.\n"
            "Dengeli: orta hız.\n"
            "Maksimum Hız: PC'yi meşgul edebilir — yalnızca boşta kullanın."
        )
        profile_row.addWidget(self.cmb_work_profile)
        profile_row.addStretch()
        adv.addLayout(profile_row)

        self.btn_reindex = QPushButton("Tüm kaynakları yeniden indeksle")
        self.btn_reindex.clicked.connect(self.reindex_requested.emit)
        adv.addWidget(self.btn_reindex)
        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color:#94a3b8;font-size:11px;")
        adv.addWidget(self.lbl_hint)
        layout.addWidget(self.grp_advanced)
        layout.addStretch()

        self.btn_ai_toggle.toggled.connect(self._on_ai_toggle_button)
        self.btn_ocr_toggle.toggled.connect(self._on_ocr_toggle_button)
        self.chk_ai_simple.toggled.connect(self._on_ai_simple_toggled)
        self.chk_ocr_simple.toggled.connect(self._on_ocr_simple_toggled)
        self.chk_ai.toggled.connect(self._on_ai_toggled)
        self.chk_index_ai.toggled.connect(self._apply_settings)
        self.chk_ocr.toggled.connect(self._on_ocr_toggled)
        for w in (
            self.chk_auto_patch,
            self.chk_auto_ocr,
            self.chk_fine,
            self.chk_text_visual,
            self.chk_live_scores,
            self.chk_ai_live,
            self.chk_lock_ranking,
            self.chk_format_panel,
            self.chk_category_panel,
        ):
            w.toggled.connect(self._apply_settings)
        self.spin_fine_k.valueChanged.connect(self._apply_settings)
        self.cmb_net_workers.currentIndexChanged.connect(self._apply_settings)
        self.cmb_work_profile.currentIndexChanged.connect(self._apply_settings)

    def set_advanced_mode(self, advanced: bool) -> None:
        self.grp_advanced.setVisible(advanced)
        self.grp_simple.setVisible(not advanced)
        self._refresh_ocr_warning()

    @staticmethod
    def _toggle_style(on: bool) -> str:
        if on:
            return (
                "QPushButton{background:#16a34a;color:#fff;font-weight:bold;"
                "border-radius:6px;padding:6px;}"
            )
        return (
            "QPushButton{background:#334155;color:#cbd5e1;font-weight:bold;"
            "border-radius:6px;padding:6px;}"
        )

    def _refresh_toggle_buttons(self) -> None:
        s = self._settings
        ai_on = bool(s.ai_embedding_enabled)
        ocr_on = bool(s.ocr_enabled)
        ocr_ok, _msg = probe_ocr_dependencies()
        self._block = True
        self.btn_ai_toggle.setChecked(ai_on)
        self.btn_ocr_toggle.setChecked(ocr_on)
        self._block = False
        self.btn_ai_toggle.setText(
            f"Yapay zekâ vektörü: {'AÇIK' if ai_on else 'KAPALI'}"
        )
        self.btn_ai_toggle.setStyleSheet(self._toggle_style(ai_on))
        self.btn_ocr_toggle.setEnabled(ocr_ok)
        self.btn_ocr_toggle.setText(
            f"OCR: {'AÇIK' if ocr_on else 'KAPALI'}"
        )
        self.btn_ocr_toggle.setStyleSheet(self._toggle_style(ocr_on))
        if not ocr_ok:
            self.lbl_toggle_hint.setText(
                "OCR kütüphanesi kurulu değil; OCR açılamıyor."
            )
        else:
            self.lbl_toggle_hint.setText("")

    def _on_ai_toggle_button(self, checked: bool) -> None:
        if self._block:
            return
        self._block = True
        self.chk_ai.setChecked(checked)
        self._block = False
        self._on_ai_toggled(checked)
        self._refresh_toggle_buttons()

    def _on_ocr_toggle_button(self, checked: bool) -> None:
        if self._block:
            return
        self._block = True
        self.chk_ocr.setChecked(checked)
        self._block = False
        self._on_ocr_toggled(checked)
        self._refresh_toggle_buttons()

    def _show_ocr_install_help(self) -> None:
        ok, msg = probe_ocr_dependencies()
        if ok:
            QMessageBox.information(self, "OCR", "OCR kütüphanesi kurulu görünüyor.")
            return
        QMessageBox.information(
            self,
            "OCR Kurulumu",
            msg or OCR_FALLBACK_MSG,
        )

    def _refresh_ocr_warning(self) -> None:
        ok, _msg = probe_ocr_dependencies()
        if ok:
            self.lbl_ocr_warning.setVisible(False)
            self.btn_ocr_install.setVisible(False)
            self.chk_ocr.setEnabled(True)
            self.chk_ocr_simple.setEnabled(True)
            return
        self.lbl_ocr_warning.setText(
            "OCR kurulu değil. Yazılı desen araması OCR olmadan sınırlı çalışır."
        )
        self.lbl_ocr_warning.setVisible(True)
        self.btn_ocr_install.setVisible(True)
        self.chk_ocr.setEnabled(False)
        self.chk_ocr_simple.setEnabled(False)
        self._block = True
        self.chk_ocr.setChecked(False)
        self.chk_ocr_simple.setChecked(False)
        self._block = False

    def _enable_index_ai(self) -> None:
        """AI arama açılınca index embedding'i de varsayılan aç."""
        self._block = True
        self.chk_index_ai.setChecked(True)
        self._block = False

    def _update_hint(self) -> None:
        s = self._settings
        parts: list[str] = []
        if s.ai_embedding_enabled:
            parts.append("Yapay zekâ araması: açık")
            parts.append(
                "İndeks yapay zekâsı: açık (yavaş, gece çalıştırın)"
                if not s.index_skip_ai
                else "İndeks yapay zekâsı: kapalı (önce hızlı katman)"
            )
        else:
            parts.append("Yapay zekâ araması: kapalı (özet + doku)")
        if getattr(s, "index_light_first", True):
            parts.append("Hızlı indeks: küçük görsel + özet + doku önce")
        parts.append("OCR: açık" if s.ocr_enabled else "OCR: kapalı")
        if getattr(s, "fine_detail_enabled", True):
            parts.append(f"Ayrıntılı sıralama ilk {getattr(s, 'fine_detail_top_k', 80)}")
        self.lbl_hint.setText(" · ".join(parts))

    def load_from_settings(self) -> None:
        s = self._settings
        self._block = True
        try:
            self.chk_ai.setChecked(bool(s.ai_embedding_enabled))
            self.chk_ai_simple.setChecked(bool(s.ai_embedding_enabled))
            self.chk_index_ai.setChecked(not bool(s.index_skip_ai))
            self.chk_ocr.setChecked(bool(s.ocr_enabled))
            self.chk_ocr_simple.setChecked(bool(s.ocr_enabled))
            self.chk_auto_patch.setChecked(
                bool(getattr(s, "auto_patch_after_ai_final", True))
            )
            self.chk_auto_ocr.setChecked(
                bool(getattr(s, "auto_ocr_after_ai_final", True))
            )
            self.chk_fine.setChecked(bool(getattr(s, "fine_detail_enabled", True)))
            self.chk_text_visual.setChecked(bool(s.search_text_visual))
            self.chk_live_scores.setChecked(
                bool(getattr(s, "live_score_updates", True))
            )
            self.chk_ai_live.setChecked(bool(getattr(s, "ai_live_refresh", True)))
            self.chk_lock_ranking.setChecked(
                bool(getattr(s, "lock_result_ranking", True))
            )
            self.chk_format_panel.setChecked(
                bool(getattr(s, "format_panel_enabled", True))
            )
            self.chk_category_panel.setChecked(
                bool(getattr(s, "category_tree_panel_enabled", True))
            )
            self.spin_fine_k.setValue(int(getattr(s, "fine_detail_top_k", 80) or 80))
            net_workers = int(getattr(s, "max_network_workers", 2) or 2)
            net_idx = self.cmb_net_workers.findData(net_workers)
            if net_idx < 0:
                self.cmb_net_workers.addItem(f"{net_workers}", net_workers)
                net_idx = self.cmb_net_workers.findData(net_workers)
            self.cmb_net_workers.setCurrentIndex(max(0, net_idx))
            wp = str(getattr(s, "work_profile", "background") or "background")
            wp_idx = self.cmb_work_profile.findData(wp)
            if wp_idx < 0:
                wp_idx = self.cmb_work_profile.findData("background")
            self.cmb_work_profile.setCurrentIndex(max(0, wp_idx))
        finally:
            self._block = False
        self._update_hint()
        # Ağır probe'ları bir sonraki event loop'a bırak — sekme geçişi pamuk gibi kalsın
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._refresh_ocr_warning)
        QTimer.singleShot(0, self._refresh_toggle_buttons)

    def _on_ai_simple_toggled(self, checked: bool) -> None:
        if self._block:
            return
        self._block = True
        self.chk_ai.setChecked(checked)
        self._block = False
        self._on_ai_toggled(checked)

    def _on_ocr_simple_toggled(self, checked: bool) -> None:
        if self._block or not checked:
            return
        ok, _msg = probe_ocr_dependencies()
        if not ok:
            self._block = True
            self.chk_ocr_simple.setChecked(False)
            self._block = False
            return
        self._block = True
        self.chk_ocr.setChecked(True)
        self._block = False
        self._apply_settings()

    def _on_ai_toggled(self, checked: bool) -> None:
        if self._block:
            return
        if checked:
            ok, msg = probe_ai_dependencies()
            if not ok:
                QMessageBox.warning(self, "AI", msg or AI_FALLBACK_MSG)
                self._block = True
                self.chk_ai.setChecked(False)
                self.chk_ai_simple.setChecked(False)
                self._block = False
                return
            self._enable_index_ai()
        self._block = True
        self.chk_ai_simple.setChecked(checked)
        self._block = False
        self._apply_settings()

    def _on_ocr_toggled(self, checked: bool) -> None:
        if self._block:
            return
        if checked:
            ok, _msg = probe_ocr_dependencies()
            if not ok:
                self._block = True
                self.chk_ocr.setChecked(False)
                self.chk_ocr_simple.setChecked(False)
                self._block = False
                self._refresh_ocr_warning()
                return
        self._block = True
        self.chk_ocr_simple.setChecked(checked)
        self._block = False
        self._apply_settings()
        self._refresh_ocr_warning()

    def _apply_settings(self) -> None:
        if self._block:
            return
        s = self._settings
        s.ai_embedding_enabled = self.chk_ai.isChecked()
        s.index_skip_ai = not self.chk_index_ai.isChecked()
        s.ocr_enabled = self.chk_ocr.isChecked()
        s.auto_patch_after_ai_final = self.chk_auto_patch.isChecked()
        s.auto_ocr_after_ai_final = self.chk_auto_ocr.isChecked()
        s.fine_detail_enabled = self.chk_fine.isChecked()
        s.search_text_visual = self.chk_text_visual.isChecked()
        s.live_score_updates = self.chk_live_scores.isChecked()
        s.ai_live_refresh = self.chk_ai_live.isChecked()
        s.lock_result_ranking = self.chk_lock_ranking.isChecked()
        s.format_panel_enabled = self.chk_format_panel.isChecked()
        s.category_tree_panel_enabled = self.chk_category_panel.isChecked()
        s.fine_detail_top_k = int(self.spin_fine_k.value())
        net_val = self.cmb_net_workers.currentData()
        if net_val is not None:
            s.max_network_workers = int(net_val)
        wp_val = self.cmb_work_profile.currentData()
        if wp_val is not None:
            s.work_profile = str(wp_val)
        s.save()
        self._update_hint()
        self._refresh_toggle_buttons()
        self.settings_changed.emit()
