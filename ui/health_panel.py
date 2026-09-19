"""Sistem Sağlığı paneli — canlı metrikler ve üretim araçları."""

from __future__ import annotations

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.logger import setup_logger
from core.production.benchmark import run_performance_benchmark
from core.production.index_verify import repair_index, verify_index
from core.production.log_analyzer import analyze_logs_24h
from core.production.report import generate_production_report
from core.production.self_healing import run_self_healing
from core.production.stress_test import run_stress_test
from core.production.visual_meaning_test import run_visual_meaning_test

logger = setup_logger(__name__)


class _ProductionWorker(QThread):
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(self, task: str, settings, parent=None):
        super().__init__(parent)
        self.task = task
        self.settings = settings
        self._report: dict | None = None

    def run(self) -> None:
        try:
            if self.task == "verify":
                result = verify_index(self.settings)
            elif self.task == "verify_then_repair":
                report = verify_index(self.settings)
                result = repair_index(self.settings, report)
            elif self.task == "repair":
                report = self._report or verify_index(self.settings)
                result = repair_index(self.settings, report)
            elif self.task == "heal":
                result = run_self_healing(self.settings)
            elif self.task == "benchmark":
                result = run_performance_benchmark(self.settings)
            elif self.task == "stress":
                result = run_stress_test(self.settings, search_count=200)
            elif self.task == "visual_meaning":
                result = run_visual_meaning_test(self.settings)
            elif self.task == "report":
                result = generate_production_report(self.settings)
            elif self.task == "logs":
                result = analyze_logs_24h(self.settings)
            else:
                raise ValueError(f"Bilinmeyen görev: {self.task}")
            self.finished_ok.emit(result)
        except Exception as exc:
            logger.exception("Production task %s failed", self.task)
            self.error.emit(str(exc))


class HealthPanel(QWidget):
    """Canlı sistem sağlığı + doğrulama/benchmark/rapor."""

    status_message = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker: _ProductionWorker | None = None
        self._last_verify: dict | None = None
        self._build_ui()
        self._last_snapshot: dict = {}

    def start_monitoring(self) -> None:
        """Arka plan health monitor snapshot'larını uygula — UI thread'de ağır iş yok."""

    def stop_monitoring(self) -> None:
        pass

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        grp = QGroupBox("Sistem Sağlığı")
        layout.addWidget(grp)
        inner = QVBoxLayout(grp)

        self.lbl_cpu = QLabel("CPU: —")
        self.lbl_ram = QLabel("RAM: —")
        self.lbl_gpu = QLabel("GPU: —")
        self.lbl_disk = QLabel("Disk erişimi: —")
        self.lbl_threads = QLabel("İş parçacığı: — | Sistem tanıtıcısı: —")
        self.lbl_sqlite = QLabel("SQLite: —")
        self.lbl_faiss = QLabel("FAISS: —")
        self.lbl_queues = QLabel("Kuyruklar: —")
        self.lbl_cache = QLabel("Önbellek: —")
        self.lbl_physical_ready = QLabel("Sistem Doğrulaması: —")
        self.lbl_cache_sync = QLabel("Doğrulama Sağlığı: —")
        self.lbl_cache_repair = QLabel("Onarım: —")
        for w in (
            self.lbl_cpu,
            self.lbl_ram,
            self.lbl_gpu,
            self.lbl_disk,
            self.lbl_threads,
            self.lbl_sqlite,
            self.lbl_faiss,
            self.lbl_queues,
            self.lbl_cache,
            self.lbl_physical_ready,
            self.lbl_cache_sync,
            self.lbl_cache_repair,
        ):
            inner.addWidget(w)

        # --- Arşiv Sağlığı (thin projection; counts only) ---
        ah_grp = QGroupBox("Arşiv Sağlığı")
        ah_inner = QVBoxLayout(ah_grp)
        self.lbl_ah_healthy = QLabel("Sağlıklı: —")
        self.lbl_ah_missing = QLabel("Eksik: —")
        self.lbl_ah_repair = QLabel("Onarım gerekli: —")
        self.lbl_ah_broken = QLabel("Bozuk: —")
        self.lbl_ah_failed = QLabel("Başarısız: —")
        for w in (
            self.lbl_ah_healthy,
            self.lbl_ah_missing,
            self.lbl_ah_repair,
            self.lbl_ah_broken,
            self.lbl_ah_failed,
        ):
            ah_inner.addWidget(w)
        inner.addWidget(ah_grp)

        btn_row1 = QHBoxLayout()
        self.btn_verify = QPushButton("İndeksi Doğrula")
        self.btn_verify.clicked.connect(self._on_verify)
        self.btn_repair = QPushButton("Eksikleri Onar")
        self.btn_repair.clicked.connect(self._on_repair)
        self.btn_heal = QPushButton("Self-Heal")
        self.btn_heal.clicked.connect(lambda: self._run_task("heal", "Self-healing…"))
        btn_row1.addWidget(self.btn_verify)
        btn_row1.addWidget(self.btn_repair)
        btn_row1.addWidget(self.btn_heal)
        inner.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_bench = QPushButton("Benchmark")
        self.btn_bench.clicked.connect(lambda: self._run_task("benchmark", "Benchmark…"))
        self.btn_stress = QPushButton("Stres Testi")
        self.btn_stress.clicked.connect(self._on_stress)
        self.btn_report = QPushButton("Üretim Raporu")
        self.btn_report.clicked.connect(lambda: self._run_task("report", "Rapor…"))
        self.btn_logs = QPushButton("Log Analizi (24s)")
        self.btn_logs.clicked.connect(lambda: self._run_task("logs", "Log analizi…"))
        self.btn_meaning = QPushButton("Görsel Anlam Testi")
        self.btn_meaning.clicked.connect(self._on_visual_meaning)
        btn_row2.addWidget(self.btn_bench)
        btn_row2.addWidget(self.btn_stress)
        btn_row2.addWidget(self.btn_report)
        btn_row2.addWidget(self.btn_logs)
        inner.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        btn_row3.addWidget(self.btn_meaning)
        inner.addLayout(btn_row3)

        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setMaximumHeight(360)
        inner.addWidget(self.txt_output)

    def apply_snapshot(self, snap: dict) -> None:
        self._last_snapshot = snap
        cpu = snap.get("system_cpu_percent", snap.get("cpu_percent", 0))
        ram_pct = snap.get("system_ram_percent", 0)
        rss = snap.get("rss_mb", 0)
        gpu = snap.get("gpu_util_percent")
        self.lbl_cpu.setText(f"CPU: {cpu:.0f}% (işlem)")
        self.lbl_ram.setText(f"RAM: {rss:.0f} MB | Sistem %{ram_pct:.0f}")
        self.lbl_gpu.setText(
            f"GPU: {gpu:.0f}%" if gpu is not None else "GPU: — (nvidia-smi yok)"
        )
        dr = snap.get("disk_read_mb_s", 0)
        dw = snap.get("disk_write_mb_s", 0)
        self.lbl_disk.setText(f"Disk I/O: okuma {dr:.1f} MB/s | yazma {dw:.1f} MB/s")
        self.lbl_threads.setText(
            f"İş parçacığı: {snap.get('threads', snap.get('num_threads', '—'))} | "
            f"Sistem tanıtıcısı: {snap.get('handles', snap.get('num_handles', '—'))}"
        )
        sq = snap.get("sqlite", {})
        sq_txt = "OK" if sq.get("ok") else ("KİLİTLİ" if sq.get("locked") else "HATA")
        self.lbl_sqlite.setText(f"SQLite: {sq_txt} | WAL: {sq.get('wal', False)}")
        fa = snap.get("faiss", {})
        self.lbl_faiss.setText(
            f"FAISS: {'aktif' if fa.get('available') else 'pasif'} | "
            f"DINO {fa.get('dino_mb', 0)} MB | CLIP {fa.get('clip_mb', 0)} MB"
        )
        q = snap.get("queues", {})
        self.lbl_queues.setText(
            f"Çalışan: {q.get('INDEX_ACTIVE', q.get('index_active', 0))} | "
            f"Genel AI kalan: {q.get('GENERAL_AI_PENDING', 0)} | "
            f"Light kuyruk: {q.get('LIGHT_PENDING', 0)} | "
            f"OCR: {q.get('ocr_pending', 0)}"
        )
        c = snap.get("cache_mb", {})
        self.lbl_cache.setText(
            f"Cache: {c.get('total', 0)} MB (thumb {c.get('thumbnails', 0)} MB)"
        )
        # Lightweight archive-health counts when db_path is configured
        try:
            if str(getattr(self.settings, "db_path", "") or "").strip():
                self.refresh_archive_health()
        except Exception:
            pass

    def set_reconciliation_status(self, stats: dict) -> None:
        physical = stats.get("physical") or {}
        ready = int(physical.get("physical_ready", stats.get("ready", 0)) or 0)
        missing = int(
            physical.get("physical_missing", stats.get("missing", 0)) or 0
        )
        unverified = int(physical.get("physical_unverified", 0) or 0)
        total = int(physical.get("total", ready + missing + unverified) or 0)
        verified = max(0, total - unverified)
        sync_pct = (100.0 * verified / total) if total else 0.0
        last_at = str(physical.get("last_verified_at", "") or "—")
        self.lbl_physical_ready.setText(
            f"Sistem Doğrulaması: {verified:,} / {total:,} | "
            f"Kullanılabilir: {ready:,}"
        )
        self.lbl_cache_sync.setText(
            f"Doğrulama Sağlığı: Eksik {missing:,} | "
            f"Bekleyen {unverified:,} | %{sync_pct:.1f} | Son: {last_at}"
        )
        repair_state = "duraklatıldı" if stats.get("repair_paused") else "aktif"
        self.lbl_cache_repair.setText(
            f"Repair ({repair_state}): batch {int(stats.get('checked', 0) or 0)} | "
            f"onarılan {int(stats.get('repaired', 0) or 0)} | "
            f"AI eksik {int(stats.get('ai_missing', 0) or 0)} | "
            f"AI requeue {int(stats.get('ai_requeued', 0) or 0)} | "
            f"AI onarılan {int(stats.get('ai_repaired', 0) or 0)} | "
            f"başarısız {int(stats.get('repair_failed', 0) or 0)}"
        )

    def set_archive_health(self, summary: dict) -> None:
        """Update Arşiv Sağlığı labels from summarize_archive_health result."""
        try:
            from core.archive_health import format_archive_health_labels

            labels = format_archive_health_labels(summary or {})
        except Exception:
            labels = {
                "healthy": "Sağlıklı: —",
                "missing": "Eksik: —",
                "repair_needed": "Onarım gerekli: —",
                "broken": "Bozuk: —",
                "failed": "Başarısız: —",
            }
        self.lbl_ah_healthy.setText(labels.get("healthy", "Sağlıklı: —"))
        self.lbl_ah_missing.setText(labels.get("missing", "Eksik: —"))
        self.lbl_ah_repair.setText(labels.get("repair_needed", "Onarım gerekli: —"))
        self.lbl_ah_broken.setText(labels.get("broken", "Bozuk: —"))
        self.lbl_ah_failed.setText(labels.get("failed", "Başarısız: —"))

    def refresh_archive_health(self) -> None:
        """Lightweight counts-only refresh; never blocks on heavy scans."""
        try:
            settings = self.settings
            db_path = str(getattr(settings, "db_path", "") or "").strip()
            if not db_path:
                self.set_archive_health({})
                return
            from core.archive_health import (
                open_job_store_for_settings,
                summarize_archive_health,
            )
            from core.db import Database

            try:
                db = Database(db_path, read_only=True)
            except TypeError:
                db = Database(db_path)
            job_store = open_job_store_for_settings(settings)
            summary = summarize_archive_health(db, job_store=job_store)
            self.set_archive_health(summary)
        except Exception as exc:
            logger.debug("refresh_archive_health failed: %s", exc)
            self.lbl_ah_healthy.setText("Sağlıklı: —")
            self.lbl_ah_missing.setText("Eksik: —")
            self.lbl_ah_repair.setText("Onarım gerekli: —")
            self.lbl_ah_broken.setText("Bozuk: —")
            self.lbl_ah_failed.setText("Başarısız: —")

    def _refresh_metrics(self) -> None:
        if self._last_snapshot:
            self.apply_snapshot(self._last_snapshot)

    def _set_busy(self, busy: bool) -> None:
        for btn in (
            self.btn_verify,
            self.btn_repair,
            self.btn_heal,
            self.btn_bench,
            self.btn_stress,
            self.btn_report,
            self.btn_logs,
            self.btn_meaning,
        ):
            btn.setEnabled(not busy)

    def _run_task(self, task: str, msg: str, *, report: dict | None = None) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._set_busy(True)
        self.status_message.emit(msg)
        self._worker = _ProductionWorker(task, self.settings, self)
        if report is not None:
            self._worker._report = report
        self._worker.finished_ok.connect(self._on_task_done)
        self._worker.error.connect(self._on_task_error)
        self._worker.start()

    def _on_verify(self) -> None:
        self._run_task("verify", "Index doğrulanıyor…")

    def _on_repair(self) -> None:
        if self._last_verify:
            self._run_task("repair", "Eksikler onarılıyor…", report=self._last_verify)
        else:
            self._run_task("verify_then_repair", "Doğrulama ve onarım…")

    def _on_visual_meaning(self) -> None:
        reply = QMessageBox.question(
            self,
            "Görsel Anlam Testi",
            "İndekste 22 kavram aranacak (her biri ilk 20 sonuç). "
            "Kod ve ağırlık değişmez; yalnızca ölçüm raporu üretilir. Devam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_task("visual_meaning", "Görsel anlam testi…")

    def _on_stress(self) -> None:
        reply = QMessageBox.question(
            self,
            "Stres Testi",
            "200 eşzamanlı arama çalıştırılacak. Devam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_task("stress", "Stres testi…")

    def _on_task_done(self, result: dict) -> None:
        self._set_busy(False)
        import json

        if "issues" in result:
            self._last_verify = result
            issues = result.get("issues", {})
            lines = [f"Örneklenen: {result.get('sampled', 0)}"]
            for k, v in issues.items():
                lines.append(f"  {k}: {v}")
            self.txt_output.setPlainText("\n".join(lines))
            self.status_message.emit(
                "Index doğrulama tamam" if result.get("ok") else "Eksikler bulundu"
            )
        elif "html_path" in result:
            self.txt_output.setPlainText(f"Rapor: {result['html_path']}")
            self.status_message.emit("Üretim raporu oluşturuldu")
        elif "search_count" in result:
            self.txt_output.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
            self.status_message.emit(
                "Stres testi OK" if result.get("ok") else "Stres testinde hata"
            )
        elif result.get("report_text"):
            self.txt_output.setPlainText(str(result.get("report_text") or ""))
            self.status_message.emit(
                f"Görsel anlam %{result.get('overall_pct', 0):.0f}"
            )
        else:
            self.txt_output.setPlainText(json.dumps(result, indent=2, ensure_ascii=False)[:4000])
            self.status_message.emit("İşlem tamamlandı")

    def _on_task_error(self, msg: str) -> None:
        self._set_busy(False)
        self.txt_output.setPlainText(f"Hata: {msg}")
        self.status_message.emit("Hata: " + msg[:80])
