"""StatusWorker deleted-wrapper safety (main_window refresh paths).

Regression for production signature:
  RuntimeError: libshiboken: Internal C++ object (StatusWorker) already deleted.
Root cause was 1 Hz _refresh_status_during_index calling .isRunning() on a
zombie Python wrapper after finished→deleteLater (log spam ~15k× on 2026-09-21).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestStatusWorkerDeletedRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_helpers_safe_on_deleted_status_like_qthread(self):
        from PySide6.QtCore import QThread
        from shiboken6 import delete, isValid

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        t = QThread()
        delete(t)
        self.assertFalse(isValid(t))
        self.assertFalse(qobject_is_alive(t))
        self.assertFalse(qthread_is_running(t))  # must not raise

    def test_qa_signature_zombie_isrunning_raises_without_helper(self):
        """Exact libshiboken signature when calling isRunning on deleted wrapper."""
        from PySide6.QtCore import QThread
        from shiboken6 import delete

        t = QThread()
        t.setObjectName("StatusWorker")
        delete(t)
        with self.assertRaises(RuntimeError) as ctx:
            t.isRunning()
        msg = str(ctx.exception)
        self.assertIn("libshiboken", msg)
        self.assertIn("already deleted", msg)

    def test_refresh_during_index_clears_zombie_and_skips_direct_isrunning(self):
        """Mirrors _refresh_status_during_index guard without constructing MainWindow."""
        from PySide6.QtCore import QThread
        from shiboken6 import delete

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        class Holder:
            def __init__(self):
                self._index_active = True
                self._lane_status_worker = None
                self._spawned = 0

            def _clear_lane_status_worker_ref_if_same(self, worker) -> None:
                if self._lane_status_worker is worker:
                    self._lane_status_worker = None

            def _refresh_status_during_index(self) -> None:
                cur = getattr(self, "_lane_status_worker", None)
                if qthread_is_running(cur):
                    return
                if cur is not None and not qobject_is_alive(cur):
                    self._lane_status_worker = None
                w = QThread()
                self._lane_status_worker = w
                self._spawned += 1
                w.destroyed.connect(
                    lambda *_a, worker=w: self._clear_lane_status_worker_ref_if_same(
                        worker
                    )
                )

        h = Holder()
        zombie = QThread()
        zombie.setObjectName("StatusWorker")
        h._lane_status_worker = zombie
        delete(zombie)
        h._refresh_status_during_index()
        self.assertEqual(h._spawned, 1)
        self.assertIsNotNone(h._lane_status_worker)
        self.assertTrue(qobject_is_alive(h._lane_status_worker))

        running = h._lane_status_worker
        running.start()
        self.assertTrue(qthread_is_running(running))
        h._refresh_status_during_index()
        self.assertEqual(h._spawned, 1)
        running.quit()
        self.assertTrue(running.wait(3000))

    def test_1hz_timer_spam_path_does_not_raise_or_excepthook(self):
        """Simulate index timer 1 Hz hitting zombie lane worker (production spam path)."""
        from PySide6.QtCore import QThread
        from shiboken6 import delete

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        caught = []

        def _hook(exc_type, exc, tb):
            caught.append((exc_type, str(exc)))

        old = sys.excepthook
        sys.excepthook = _hook
        try:
            zombie = QThread()
            zombie.setObjectName("StatusWorker")
            delete(zombie)
            for _ in range(50):
                cur = zombie
                if qthread_is_running(cur):
                    continue
                if cur is not None and not qobject_is_alive(cur):
                    cur = None
                self.assertIsNone(cur)
        finally:
            sys.excepthook = old
        self.assertEqual(caught, [])

    def test_on_status_updated_clears_ref_by_identity_without_isrunning(self):
        """finished_ok must drop refs by identity; no isRunning on possibly-deleted sender."""
        from PySide6.QtCore import QObject, Signal

        from core.qthread_lifecycle import qobject_is_alive

        class FakeWorker(QObject):
            finished_ok = Signal(dict)

            def __init__(self):
                super().__init__()
                self._status_gen = 1

        class Holder(QObject):
            def __init__(self):
                super().__init__()
                self._status_worker = None
                self._lane_status_worker = None
                self._status_generation = 1
                self._status_refresh_queued = False
                self._index_active = False
                self._last_gen = None
                self._last_status = {}

            def _on_status_updated(self, status: dict) -> None:
                sender = self.sender()
                if sender is self._status_worker:
                    self._status_worker = None
                elif sender is self._lane_status_worker:
                    self._lane_status_worker = None
                gen = None
                if qobject_is_alive(sender):
                    try:
                        gen = getattr(sender, "_status_gen", None)
                    except RuntimeError:
                        gen = None
                self._last_gen = gen
                self._last_status = dict(status)

        h = Holder()
        w = FakeWorker()
        h._status_worker = w
        w.finished_ok.connect(h._on_status_updated)
        w.finished_ok.emit({"total": 1})
        self._app.processEvents()
        self.assertIsNone(h._status_worker)
        self.assertEqual(h._last_gen, 1)

    def test_clear_lane_ref_on_destroyed(self):
        from PySide6.QtCore import QCoreApplication, QEvent, QThread
        from shiboken6 import delete

        class Holder:
            def __init__(self):
                self._lane_status_worker = None

            def _clear_lane_status_worker_ref_if_same(self, worker) -> None:
                if self._lane_status_worker is worker:
                    self._lane_status_worker = None

        def _flush():
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self._app.processEvents()

        h = Holder()
        w = QThread()
        h._lane_status_worker = w
        w.destroyed.connect(
            lambda *_a, ww=w: h._clear_lane_status_worker_ref_if_same(ww)
        )
        delete(w)
        _flush()
        self.assertIsNone(h._lane_status_worker)

    def test_main_window_source_uses_qthread_is_running_for_lane(self):
        src = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
        self.assertNotIn("_lane_status_worker.isRunning()", src)
        self.assertNotIn("self._status_worker.isRunning()", src)
        self.assertIn("qthread_is_running(cur)", src)
        self.assertIn("_clear_lane_status_worker_ref_if_same", src)
        self.assertIn("_clear_status_worker_ref_if_same", src)

    def test_refresh_status_has_single_start_no_duplicate_connect(self):
        """Botched hotfix left double finished_ok.connect + start(); must stay single."""
        import re

        src = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
        m = re.search(
            r"def _refresh_status\(self\).*?\n    def _on_index_progress",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertEqual(body.count("w.start()"), 1)
        self.assertEqual(body.count("w.finished_ok.connect(self._on_status_updated)"), 1)
        # hasattr(...) + call → two textual hits; only one call site allowed
        self.assertEqual(body.count("w.arm_delete_later_on_finished()"), 1)


if __name__ == "__main__":
    unittest.main()
