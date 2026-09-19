"""Background face indexing service.

Runs independently of the pattern index. It continuously catches up with the
current file index and revisits older IDs so modified files are eventually
re-scanned. It never changes patterns.db or FAISS.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from core.face_identity import FaceIdentityEngine
from core.face_scanner import FaceIndexScanner


class FaceBackgroundIndexer:
    def __init__(self, settings: Any):
        self.settings = settings
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_stats: dict[str, Any] = {}
        self._cursor = 0

    def start(self) -> bool:
        if not bool(getattr(self.settings, "face_index_enabled", False)):
            return False
        if self.thread and self.thread.is_alive():
            return True
        if FaceIdentityEngine.capability().get("face_embedding") == "optional_dependency":
            self.last_stats = {"dependency_missing": True}
            return False
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="VezirFaceIndexer",
            daemon=True,
        )
        self.thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=max(0.1, float(timeout)))
        self.thread = None

    def _run(self) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if bool(getattr(self.settings, "use_gpu", False))
            else ["CPUExecutionProvider"]
        )
        scanner = FaceIndexScanner(
            self.settings.db_path,
            self.settings.face_db_path,
            threshold=float(getattr(self.settings, "face_identity_threshold", 0.62)),
            min_margin=float(getattr(self.settings, "face_identity_min_margin", 0.05)),
            model_name=str(getattr(self.settings, "face_identity_model", "buffalo_l") or "buffalo_l"),
            providers=providers,
            det_size=int(getattr(self.settings, "face_identity_det_size", 640) or 640),
        )
        batch = max(1, min(200, int(getattr(self.settings, "face_identity_scan_batch", 25))))
        interval = max(2.0, float(getattr(self.settings, "face_identity_scan_interval", 10.0)))
        while not self.stop_event.is_set():
            try:
                result = scanner.scan(
                    after_id=self._cursor,
                    limit=batch,
                )
                self.last_stats = dict(result or {})
                last_id = int(result.get("last_id", self._cursor) if result else self._cursor)
                if last_id > self._cursor:
                    self._cursor = last_id
                else:
                    # End of current index. Start another pass so modified files
                    # and files added during the previous pass are discovered.
                    self._cursor = 0
                    try:
                        scanner.reconcile()
                    except Exception:
                        pass
                    self.stop_event.wait(interval)
                    continue
                # Small yield prevents face indexing from starving pattern search.
                self.stop_event.wait(0.05)
            except Exception as exc:
                self.last_stats = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "dependency_missing": False,
                }
                self.stop_event.wait(interval)
