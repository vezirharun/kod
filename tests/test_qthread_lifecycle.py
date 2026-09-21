"""Unit tests for QThread lifecycle stability + shared AI extractor cache."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _ensure_module(name: str, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            # Point at real package dir so sibling modules remain importable
            pkg_dir = ROOT / parent.replace(".", "/")
            pkg.__path__ = [str(pkg_dir)] if pkg_dir.is_dir() else []  # type: ignore[attr-defined]
            pkg.__file__ = str(pkg_dir / "__init__.py")
            sys.modules[parent] = pkg
        else:
            # If a stub parent has empty __path__, repair it
            pkg = sys.modules[parent]
            pkg_dir = ROOT / parent.replace(".", "/")
            if pkg_dir.is_dir() and not getattr(pkg, "__path__", None):
                pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    # Don't overwrite real modules that already exist on disk unless forcing stub
    real = ROOT.joinpath(*parts).with_suffix(".py")
    if real.is_file() and name in ("core.capability_check", "core.search_engine", "core.qthread_lifecycle"):
        # leave real module for later import; still register attrs if needed
        pass
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _FakeSettings:
    use_gpu = False
    ai_embedding_enabled = True


class _FakeQuery:
    def __init__(self):
        self.mode = "text"
        self.text = "test"
        self.image_path = ""
        self.image_paths = []
        self.use_crop = False
        self.crop_rect = None
        self.customer = ""
        self.threshold = 0.5
        self.fast_only = False


class _FakeFeats:
    dino_embedding = [0.1]
    clip_embedding = [0.2]


class FeatureExtractor:
    """Test double — counts AI model inits."""

    init_ai_calls = 0

    def __init__(
        self, use_ai: bool = False, use_gpu: bool = False, fast_hash_only: bool = False
    ):
        self.use_ai = use_ai
        self.use_gpu = use_gpu
        self.fast_hash_only = fast_hash_only
        self.device = "cuda" if use_gpu else "cpu"
        self._dino_model = None
        self._clip_model = None
        self.ai_available = False
        if use_ai:
            self._init_ai_models()

    def _init_ai_models(self) -> None:
        FeatureExtractor.init_ai_calls += 1
        self._dino_model = object()
        self._clip_model = object()
        self.ai_available = True

    def extract_from_array(
        self, arr, include_patches: bool = False, deep_analysis: bool = False
    ):
        return _FakeFeats()


def _install_stubs() -> None:
    class ScanMode:
        QUICK = types.SimpleNamespace(value="quick")

    _ensure_module(
        "core.settings",
        AppSettings=_FakeSettings,
        APP_NAME="test",
        SUPPORTED_EXTENSIONS=set(),
    )
    _ensure_module(
        "core.search_models",
        SearchQuery=_FakeQuery,
        SearchResponse=type("SearchResponse", (), {}),
        SearchStats=type("SearchStats", (), {}),
    )
    _ensure_module(
        "core.sources",
        ScanMode=ScanMode,
        SourceManager=object,
        SearchScope=object,
        SearchFilter=object,
        SEARCH_SCOPE_LABELS={},
    )
    _ensure_module(
        "core.logger",
        setup_logger=lambda name: types.SimpleNamespace(
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            exception=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )
    _ensure_module("core.indexer", Indexer=object)
    _ensure_module("core.scan_scheduler", ScanScheduler=object)
    _ensure_module("core.search_engine", SearchEngine=object)
    _ensure_module("ui.designer_labels", stage_label=lambda s: s)
    _ensure_module("core.feature_extractor", FeatureExtractor=FeatureExtractor)


class TestShouldDefer(unittest.TestCase):
    def test_should_defer_new_worker(self):
        from core.qthread_lifecycle import should_defer_new_worker

        self.assertTrue(should_defer_new_worker(True))
        self.assertFalse(should_defer_new_worker(False))


class TestWorkerObjectNameAndArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stubs()
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_object_names_and_arm(self):
        from ui.worker_threads import (
            BackgroundTask,
            CacheReconciliationWorker,
            IndexWorker,
            PurgeMissingWorker,
            PurgeSourceWorker,
            QuickIndexWorker,
            SearchWorker,
            SourceFileCountWorker,
            StatusWorker,
        )

        settings = _FakeSettings()
        query = _FakeQuery()
        workers = [
            (StatusWorker(settings), "StatusWorker"),
            (SearchWorker(settings, query), "SearchWorker"),
            (IndexWorker(settings), "IndexWorker"),
            (QuickIndexWorker(settings, "/tmp"), "QuickIndexWorker"),
            (BackgroundTask(lambda: None), "BackgroundTask"),
            (SourceFileCountWorker("/tmp", source_id=1), "SourceFileCountWorker"),
            (CacheReconciliationWorker(settings), "CacheReconciliationWorker"),
            (PurgeMissingWorker(settings), "PurgeMissingWorker"),
            (PurgeSourceWorker(settings, 1, "x"), "PurgeSourceWorker"),
        ]
        for w, name in workers:
            self.assertEqual(w.objectName(), name, msg=type(w).__name__)
            w.arm_delete_later_on_finished()
            w.arm_delete_later_on_finished()
            self.assertTrue(getattr(w, "_delete_later_armed", False))
            w.deleteLater()


class TestSharedAiExtractor(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        FeatureExtractor.init_ai_calls = 0
        import core.capability_check as cc

        cc._AI_FE = None
        cc._AI_FE_KEY = None

    def test_try_load_twice_does_not_double_init(self):
        import core.capability_check as cc

        settings = _FakeSettings()
        with mock.patch.object(
            cc, "diagnose_ai_stack", return_value={"ok": True, "message": ""}
        ):
            ok1, msg1 = cc.try_load_ai_extractor(settings)
            self.assertTrue(ok1, msg1)
            first = FeatureExtractor.init_ai_calls
            self.assertGreaterEqual(first, 1)

            ok2, msg2 = cc.try_load_ai_extractor(settings)
            self.assertTrue(ok2, msg2)
            self.assertIn("cached", msg2.lower())
            self.assertEqual(
                FeatureExtractor.init_ai_calls,
                first,
                "second try_load must not re-init AI models",
            )
            shared = cc.get_shared_ai_extractor(settings)
            self.assertIsNotNone(shared)
            self.assertTrue(shared.ai_available)

    def test_ensure_ai_loaded_uses_shared(self):
        import core.capability_check as cc

        settings = _FakeSettings()
        with mock.patch.object(
            cc, "diagnose_ai_stack", return_value={"ok": True, "message": ""}
        ):
            ok, _ = cc.try_load_ai_extractor(settings)
            self.assertTrue(ok)
            shared = cc.get_shared_ai_extractor(settings)
            self.assertIsNotNone(shared)

            class Eng:
                def __init__(self):
                    self.settings = settings
                    self.extractor = FeatureExtractor(use_ai=False)
                    self._ai_load_error = ""

                def ensure_ai_loaded(self) -> bool:
                    if self.extractor.use_ai and (
                        getattr(self.extractor, "_dino_model", None) is not None
                        or getattr(self.extractor, "_clip_model", None) is not None
                    ):
                        return True
                    ok, msg = cc.try_load_ai_extractor(self.settings)
                    if not ok:
                        self._ai_load_error = msg
                        self.extractor = FeatureExtractor(use_ai=False)
                        return False
                    ext = cc.get_shared_ai_extractor(self.settings)
                    if ext is None:
                        return False
                    self.extractor = ext
                    return bool(self.extractor.ai_available)

            before = FeatureExtractor.init_ai_calls
            eng = Eng()
            self.assertTrue(eng.ensure_ai_loaded())
            self.assertIs(eng.extractor, shared)
            self.assertEqual(FeatureExtractor.init_ai_calls, before)


class TestSearchSingleFlight(unittest.TestCase):
    def test_single_flight_decision(self):
        from core.qthread_lifecycle import should_defer_new_worker

        def decide(running: bool, new_q: str):
            if should_defer_new_worker(running):
                return ("defer", new_q)
            return ("launch", new_q)

        self.assertEqual(decide(True, "q2"), ("defer", "q2"))
        self.assertEqual(decide(False, "q1"), ("launch", "q1"))


class TestPanelObjectNames(unittest.TestCase):
    def test_production_worker_object_name_in_source(self):
        src = (ROOT / "ui/health_panel.py").read_text(encoding="utf-8")
        self.assertIn('setObjectName("_ProductionWorker")', src)

    def test_options_worker_object_name_in_source(self):
        src = (ROOT / "ui/result_metadata_dialog.py").read_text(encoding="utf-8")
        self.assertIn('setObjectName("_OptionsWorker")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestQObjectAliveHelpers(unittest.TestCase):
    """Safe guards must not raise libshiboken RuntimeError on deleted wrappers."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_none_is_not_alive_and_not_running(self):
        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        self.assertFalse(qobject_is_alive(None))
        self.assertFalse(qthread_is_running(None))

    def test_living_qthread_reports_running_state(self):
        from PySide6.QtCore import QThread

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        t = QThread()
        self.assertTrue(qobject_is_alive(t))
        self.assertFalse(qthread_is_running(t))
        t.start()
        self.assertTrue(qthread_is_running(t))
        t.quit()
        self.assertTrue(t.wait(3000))
        self.assertTrue(qobject_is_alive(t))
        self.assertFalse(qthread_is_running(t))
        t.deleteLater()
        self._app.processEvents()

    def test_deleted_wrapper_safe_helpers_return_false(self):
        from PySide6.QtCore import QObject
        from shiboken6 import isValid, delete

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        obj = QObject()
        self.assertTrue(qobject_is_alive(obj))
        delete(obj)
        self.assertFalse(isValid(obj))
        # Critical: must NOT raise RuntimeError
        self.assertFalse(qobject_is_alive(obj))
        self.assertFalse(qthread_is_running(obj))

    def test_deleted_qthread_isrunning_safe(self):
        from PySide6.QtCore import QThread
        from shiboken6 import delete, isValid

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        t = QThread()
        delete(t)
        self.assertFalse(isValid(t))
        self.assertFalse(qobject_is_alive(t))
        self.assertFalse(qthread_is_running(t))


class TestSearchWorkerRefClearOnDestroyed(unittest.TestCase):
    """Python _search_worker becomes None when C++ object is destroyed."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_clear_on_destroyed_clears_same_instance_only(self):
        from PySide6.QtCore import QCoreApplication, QEvent, QObject, QThread
        from shiboken6 import delete

        class Holder:
            def __init__(self):
                self._search_worker = None

            def _clear_search_worker_ref_if_same(self, worker) -> None:
                if self._search_worker is worker:
                    self._search_worker = None

        def _flush_deletes():
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self._app.processEvents()

        h = Holder()
        w = QThread()
        other = QThread()
        h._search_worker = w
        w.destroyed.connect(lambda *_a, ww=w: h._clear_search_worker_ref_if_same(ww))
        delete(w)  # immediate C++ destroy → destroyed signal
        _flush_deletes()
        self.assertIsNone(h._search_worker)

        h._search_worker = other
        # Destroying a different worker must not clear current ref
        gone = QObject()
        gone.destroyed.connect(
            lambda *_a, ww=gone: h._clear_search_worker_ref_if_same(ww)
        )
        delete(gone)
        _flush_deletes()
        self.assertIs(h._search_worker, other)
        delete(other)
        _flush_deletes()

    def test_finished_arm_delete_later_clears_ref(self):
        """Mirrors SearchWorker path: finished → deleteLater → destroyed → clear."""
        from PySide6.QtCore import QThread

        class Holder:
            def __init__(self):
                self._search_worker = None

            def _clear_search_worker_ref_if_same(self, worker) -> None:
                if self._search_worker is worker:
                    self._search_worker = None

            def _is_search_worker_running(self) -> bool:
                from core.qthread_lifecycle import qthread_is_running

                return qthread_is_running(self._search_worker)

        h = Holder()
        w = QThread()
        h._search_worker = w
        w.destroyed.connect(lambda *_a, ww=w: h._clear_search_worker_ref_if_same(ww))
        w.finished.connect(w.deleteLater)
        self.assertFalse(h._is_search_worker_running())
        w.start()
        self.assertTrue(h._is_search_worker_running())
        w.quit()
        self.assertTrue(w.wait(3000))
        # Process deleteLater + destroyed
        for _ in range(20):
            self._app.processEvents()
            if h._search_worker is None:
                break
        self.assertIsNone(h._search_worker)
        self.assertFalse(h._is_search_worker_running())


class TestSearchSingleFlightHotfix(unittest.TestCase):
    def test_new_search_after_finished(self):
        from core.qthread_lifecycle import qthread_is_running, should_defer_new_worker

        class FakeWorker:
            def __init__(self, running=False):
                self._running = running

            def isRunning(self):
                return self._running

        running = FakeWorker(True)
        self.assertTrue(should_defer_new_worker(qthread_is_running(running)))

        done = FakeWorker(False)
        self.assertFalse(should_defer_new_worker(qthread_is_running(done)))

        # After clear
        self.assertFalse(should_defer_new_worker(qthread_is_running(None)))

    def test_cancel_then_new_search_logic(self):
        from core.qthread_lifecycle import qthread_is_running, should_defer_new_worker

        state = {"worker": None, "pending": None, "launched": []}

        class FakeWorker:
            def __init__(self, name):
                self.name = name
                self._running = True
                self.stop_requested = False

            def isRunning(self):
                return self._running

            def request_stop(self):
                self.stop_requested = True
                self._running = False

        def start(name):
            cur = state["worker"]
            if cur is not None and should_defer_new_worker(qthread_is_running(cur)):
                state["pending"] = name
                cur.request_stop()
                return "defer"
            state["worker"] = FakeWorker(name)
            state["launched"].append(name)
            return "launch"

        def on_finished():
            state["worker"] = None
            pending = state["pending"]
            if pending is not None:
                state["pending"] = None
                start(pending)

        self.assertEqual(start("a"), "launch")
        self.assertEqual(start("b"), "defer")
        self.assertTrue(state["worker"].stop_requested)
        on_finished()
        self.assertEqual(state["launched"], ["a", "b"])
        self.assertEqual(state["worker"].name, "b")

        # Rapid cancel → new
        state["worker"].request_stop()
        state["pending"] = None  # cancel clears pending
        on_finished()
        self.assertIsNone(state["worker"])
        self.assertEqual(start("c"), "launch")
        self.assertEqual(state["worker"].name, "c")
