"""Yarım kalan index — devam dialogu."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from core.index_queue_manager import SourceQueueSummary
from core.index_session import IndexSessionState, mode_display_label


class ResumeIndexDialog(QDialog):
    def __init__(
        self,
        summaries: list[SourceQueueSummary],
        parent=None,
        *,
        session: IndexSessionState | None = None,
        progress: dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Yarım Kalan İndeksleme")
        self.setMinimumWidth(460)
        self._choice = "later"
        self._session = session
        progress = progress or {}
        layout = QVBoxLayout(self)

        lines: list[str] = ["Önceki oturum bulundu.", ""]
        if session and session.index_mode:
            lines.append(f"Mod: {mode_display_label(session.index_mode)}")
            lines.append(f"Durum: {session.status}")
            lines.append("")

        light_done = int(
            progress.get("light_done", session.light_done if session else 0) or 0
        )
        light_total = int(
            progress.get(
                "light_total",
                progress.get("total_files", session.light_total if session else 0),
            )
            or 0
        )
        heavy_done = int(
            progress.get("heavy_done", session.heavy_done if session else 0) or 0
        )
        heavy_total = int(
            progress.get(
                "heavy_total",
                progress.get("total_files", session.heavy_total if session else 0),
            )
            or 0
        )
        search_ready = int(
            progress.get("search_ready", session.search_ready if session else 0) or 0
        )
        total_files = int(
            progress.get("total_files", session.total_files if session else 0) or 0
        )

        if light_total or heavy_total or total_files:
            lines.append(f"FAST   {light_done:,} / {light_total or total_files:,}")
            lines.append(f"HEAVY  {heavy_done:,} / {heavy_total or total_files:,}")
            if search_ready or total_files:
                lines.append(
                    f"Search Ready  {search_ready:,} / {total_files or heavy_total:,}"
                )
            lines.append("")

        if summaries:
            lines.append("Kaynak özeti:")
            for s in summaries:
                pending = s.processing_files + s.queue_pending
                lines.append(
                    f"• {s.source_name}: işlenen={s.indexed}, "
                    f"bekleyen≈{pending}, hatalı={s.queue_error}"
                )
            lines.append("")

        lines.append("Devam edilsin mi?")
        lbl = QLabel("\n".join(lines))
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        buttons = QDialogButtonBox()
        btn_resume = buttons.addButton(
            "Devam Et", QDialogButtonBox.ButtonRole.AcceptRole
        )
        btn_fresh = buttons.addButton(
            "Sıfırdan Başlat", QDialogButtonBox.ButtonRole.ActionRole
        )
        btn_later = buttons.addButton("Sonra", QDialogButtonBox.ButtonRole.RejectRole)
        btn_stop = buttons.addButton(
            "Bu Oturumda Durdur", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        btn_resume.clicked.connect(self._on_resume)
        btn_fresh.clicked.connect(self._on_fresh)
        btn_later.clicked.connect(self._on_later)
        btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(buttons)
        from ui.responsive_dialog import apply_responsive_dialog

        apply_responsive_dialog(self, min_width=420, prefer_width=480)

    def _on_resume(self) -> None:
        self._choice = "resume"
        self.accept()

    def _on_fresh(self) -> None:
        self._choice = "fresh_start"
        self.accept()

    def _on_later(self) -> None:
        self._choice = "later"
        self.reject()

    def _on_stop(self) -> None:
        self._choice = "stop_session"
        self.reject()

    @property
    def choice(self) -> str:
        return self._choice

    @property
    def resume_index_mode(self) -> str:
        if self._session and self._session.index_mode:
            return self._session.index_mode
        return "standard"
