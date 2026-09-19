"""Unit tests for Archive Health thin projection (no PySide required)."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from core.archive_health import (
    HealthState,
    PRIORITY,
    assess_file_health,
    format_archive_health_labels,
    merge_states,
    summarize_archive_health,
)
from core.index_v3.types import (
    AI_FINAL_REQUIRED,
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
)


def _report(
    file_id: int = 1,
    *,
    preview: ArtifactStatus = ArtifactStatus.READY,
    thumb: ArtifactStatus = ArtifactStatus.READY,
    ai_ready: bool = True,
) -> FileArtifactReport:
    status: dict[Artifact, ArtifactStatus] = {
        Artifact.THUMBNAIL: thumb,
        Artifact.PREVIEW: preview,
    }
    for a in AI_FINAL_REQUIRED:
        status[a] = ArtifactStatus.READY if ai_ready else ArtifactStatus.MISSING
    return FileArtifactReport(
        file_id=file_id, source_id=1, path="/x.png", status=status
    )


class _FakeDb:
    def __init__(self, rows: dict[int, dict]):
        self._rows = rows

    def get_file_by_id(self, file_id: int):
        return self._rows.get(int(file_id))


def _install_assess_mock(reports: dict[int, FileArtifactReport]):
    mod = ModuleType("core.index_v3.artifact_state")

    def assess_file(db, file_id, *, require_disk=False):
        rid = int(file_id)
        if rid not in reports:
            return FileArtifactReport(file_id=rid, source_id=0, path="", status={})
        return reports[rid]

    mod.assess_file = assess_file
    prev = sys.modules.get("core.index_v3.artifact_state")
    sys.modules["core.index_v3.artifact_state"] = mod
    return prev


def _restore_assess(prev):
    if prev is None:
        sys.modules.pop("core.index_v3.artifact_state", None)
    else:
        sys.modules["core.index_v3.artifact_state"] = prev


class TestMergeStates(unittest.TestCase):
    def test_priority_order(self):
        self.assertEqual(
            merge_states(HealthState.HEALTHY, HealthState.MISSING),
            HealthState.MISSING,
        )
        self.assertEqual(
            merge_states(HealthState.MISSING, HealthState.STALE),
            HealthState.STALE,
        )
        self.assertEqual(
            merge_states(HealthState.STALE, HealthState.REPAIR_PENDING),
            HealthState.REPAIR_PENDING,
        )
        self.assertEqual(
            merge_states(HealthState.REPAIR_PENDING, HealthState.BROKEN),
            HealthState.BROKEN,
        )
        self.assertEqual(
            merge_states(HealthState.BROKEN, HealthState.FAILED),
            HealthState.FAILED,
        )

    def test_worst_first_tuple_matches(self):
        self.assertEqual(PRIORITY[0], HealthState.FAILED)
        self.assertEqual(PRIORITY[-1], HealthState.HEALTHY)

    def test_empty_is_unknown(self):
        self.assertEqual(merge_states(), HealthState.UNKNOWN)


class TestAssessFileHealth(unittest.TestCase):
    def _run(self, row, report, job_store=None):
        db = _FakeDb({1: row})
        prev = _install_assess_mock({1: report})
        try:
            return assess_file_health(db, 1, job_store=job_store)
        finally:
            _restore_assess(prev)

    def test_healthy_file_synthetic(self):
        row = {
            "id": 1,
            "status": "active",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "preview_ok",
        }
        out = self._run(row, _report(preview=ArtifactStatus.READY, ai_ready=True))
        self.assertEqual(out["state"], HealthState.HEALTHY)
        self.assertEqual(out["preview"], HealthState.HEALTHY)
        self.assertEqual(out["source"], HealthState.HEALTHY)
        self.assertEqual(out["physical"], HealthState.HEALTHY)

    def test_missing_source(self):
        row = {
            "id": 1,
            "status": "missing",
            "feature_preview_path": "",
            "physical_preview_ready": 0,
            "preview_status": "",
        }
        out = self._run(row, _report(preview=ArtifactStatus.MISSING, ai_ready=False))
        self.assertEqual(out["source"], HealthState.MISSING)
        self.assertEqual(out["state"], HealthState.MISSING)
        self.assertTrue(any("files.status=missing" in r for r in out["reasons"]))

    def test_missing_preview(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "",
            "physical_preview_ready": 0,
            "preview_status": "",
        }
        out = self._run(row, _report(preview=ArtifactStatus.MISSING, ai_ready=False))
        self.assertEqual(out["preview"], HealthState.MISSING)
        self.assertEqual(out["state"], HealthState.MISSING)
        self.assertTrue(any("artifact.PREVIEW=MISSING" in r for r in out["reasons"]))

    def test_stale_preview_path_without_physical(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 0,
            "preview_status": "",
        }
        out = self._run(row, _report(preview=ArtifactStatus.MISSING, ai_ready=False))
        self.assertEqual(out["physical"], HealthState.STALE)
        self.assertEqual(out["state"], HealthState.STALE)
        self.assertTrue(
            any("physical_preview_ready=0" in r for r in out["reasons"])
        )

    def test_broken_preview_invalid(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "preview_invalid:misrepresenting",
        }
        out = self._run(row, _report(preview=ArtifactStatus.READY, ai_ready=True))
        self.assertEqual(out["preview"], HealthState.BROKEN)
        self.assertEqual(out["state"], HealthState.BROKEN)
        self.assertTrue(
            any(r.startswith("preview_status=preview_invalid") for r in out["reasons"])
        )

    def test_repair_pending_mock_job_store(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "preview_ok",
        }
        js = SimpleNamespace(file_job_states=lambda fid: ["pending"])
        out = self._run(
            row, _report(preview=ArtifactStatus.READY, ai_ready=True), job_store=js
        )
        self.assertEqual(out["state"], HealthState.REPAIR_PENDING)
        self.assertTrue(any("jobstore.state=pending" in r for r in out["reasons"]))

    def test_failed_permanent(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "preview_ok",
        }
        js = SimpleNamespace(file_job_states=lambda fid: ["failed_permanent"])
        out = self._run(
            row, _report(preview=ArtifactStatus.READY, ai_ready=True), job_store=js
        )
        self.assertEqual(out["state"], HealthState.FAILED)
        self.assertTrue(any("failed_permanent" in r for r in out["reasons"]))

    def test_unknown_default_no_row(self):
        db = _FakeDb({})
        prev = _install_assess_mock({})
        try:
            out = assess_file_health(db, 99)
        finally:
            _restore_assess(prev)
        self.assertEqual(out["state"], HealthState.UNKNOWN)
        self.assertTrue(any("file_row_missing" in r for r in out["reasons"]))

    def test_physical_artifact_invalid(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/gone.webp",
            "physical_preview_ready": 1,
            "preview_status": "",
        }
        out = self._run(
            row, _report(preview=ArtifactStatus.INVALID, ai_ready=False)
        )
        self.assertEqual(out["preview"], HealthState.BROKEN)
        self.assertEqual(out["state"], HealthState.BROKEN)
        self.assertTrue(any("PREVIEW=INVALID" in r for r in out["reasons"]))

    def test_preview_suspicious_alone_not_broken(self):
        """Solid black/white: preview_suspicious alone must NOT force BROKEN."""
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "preview_suspicious",
        }
        out = self._run(row, _report(preview=ArtifactStatus.READY, ai_ready=True))
        self.assertNotEqual(out["state"], HealthState.BROKEN)
        self.assertEqual(out["state"], HealthState.HEALTHY)

    def test_multiple_issues_deterministic_worst(self):
        row = {
            "id": 1,
            "status": "missing",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 0,
            "preview_status": "preview_invalid:corrupt",
        }
        js = SimpleNamespace(file_job_states=lambda fid: ["pending"])
        out = self._run(
            row,
            _report(preview=ArtifactStatus.MISSING, ai_ready=False),
            job_store=js,
        )
        self.assertEqual(out["state"], HealthState.BROKEN)

    def test_legacy_empty_preview_status(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 1,
            "preview_status": "",
        }
        out = self._run(row, _report(preview=ArtifactStatus.READY, ai_ready=True))
        self.assertEqual(out["state"], HealthState.HEALTHY)
        self.assertFalse(any(r.startswith("preview_status=") for r in out["reasons"]))

    def test_preview_missing_physical_status_stale(self):
        row = {
            "id": 1,
            "status": "ok",
            "feature_preview_path": "/cache/1_fp.webp",
            "physical_preview_ready": 0,
            "preview_status": "preview_missing_physical",
        }
        out = self._run(row, _report(preview=ArtifactStatus.MISSING, ai_ready=False))
        self.assertEqual(out["state"], HealthState.STALE)


class _ConnCM:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
        self._conn.close()


class _SqliteDb:
    def __init__(self, path: str):
        self._path = path

    def connect(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return _ConnCM(conn)


def _init_minimal_schema(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            source_id INTEGER DEFAULT 1,
            status TEXT DEFAULT 'ok',
            path TEXT DEFAULT '',
            thumbnail_path TEXT DEFAULT '',
            feature_preview_path TEXT DEFAULT '',
            physical_thumbnail_ready INTEGER DEFAULT 0,
            physical_preview_ready INTEGER DEFAULT 0,
            preview_status TEXT DEFAULT '',
            format_metadata TEXT DEFAULT '',
            width INTEGER DEFAULT 0,
            ocr_processed INTEGER DEFAULT 0,
            ocr_error TEXT DEFAULT '',
            patch_error TEXT DEFAULT '',
            light_status TEXT DEFAULT '',
            heavy_status TEXT DEFAULT ''
        );
        CREATE TABLE features (
            file_id INTEGER PRIMARY KEY,
            phash TEXT DEFAULT '',
            dino_embedding BLOB,
            clip_embedding BLOB,
            texture_features TEXT DEFAULT '',
            texture_map TEXT,
            patch_embeddings TEXT DEFAULT ''
        );
        """
    )
    conn.execute(
        "INSERT INTO files (id, source_id, status, feature_preview_path, "
        "physical_preview_ready, thumbnail_path, physical_thumbnail_ready, "
        "preview_status, width) VALUES (1,1,'ok','/p1.webp',1,'/t1.webp',1,'preview_ok',100)"
    )
    conn.execute(
        "INSERT INTO features (file_id, phash, texture_map) VALUES (1, 'abc', NULL)"
    )
    conn.execute(
        "INSERT INTO files (id, source_id, status, preview_status) "
        "VALUES (2,1,'ok','')"
    )
    conn.execute("INSERT INTO features (file_id, texture_map) VALUES (2, NULL)")
    conn.execute(
        "INSERT INTO files (id, source_id, status, feature_preview_path, "
        "physical_preview_ready, preview_status) "
        "VALUES (3,1,'ok','/p3.webp',0,'')"
    )
    conn.execute("INSERT INTO features (file_id, texture_map) VALUES (3, NULL)")
    conn.execute(
        "INSERT INTO files (id, source_id, status, feature_preview_path, "
        "physical_preview_ready, preview_status) "
        "VALUES (4,1,'ok','/p4.webp',1,'preview_invalid:corrupt')"
    )
    conn.execute("INSERT INTO features (file_id, texture_map) VALUES (4, NULL)")
    conn.execute(
        "INSERT INTO files (id, source_id, status, light_status, "
        "feature_preview_path, physical_preview_ready) "
        "VALUES (5,1,'ok','failed','/p5.webp',1)"
    )
    conn.execute("INSERT INTO features (file_id, texture_map) VALUES (5, NULL)")
    conn.commit()
    conn.close()


class TestSummarizeArchiveHealth(unittest.TestCase):
    def test_summarize_buckets_with_sqlite(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "patterns.db")
            _init_minimal_schema(db_path)
            db = _SqliteDb(db_path)
            js = SimpleNamespace(
                archive_health_counts=lambda source_ids=None: {
                    "pending_files": 2,
                    "pending_light_preview_repair_files": 2,
                    "failed_permanent_files": 1,
                    "claimed_files": 0,
                }
            )
            summary = summarize_archive_health(db, job_store=js, source_ids=None)
            self.assertGreaterEqual(summary["total"], 5)
            self.assertGreaterEqual(summary["healthy"], 1)
            self.assertGreaterEqual(summary["missing"], 1)
            self.assertGreaterEqual(summary["stale"], 1)
            self.assertEqual(summary["repair_pending"], 2)
            self.assertEqual(
                summary["repair_needed"],
                summary["stale"] + summary["repair_pending"],
            )
            self.assertGreaterEqual(summary["broken"], 1)
            self.assertGreaterEqual(summary["failed"], 1)
            self.assertIn("ssot", summary)
            self.assertIn("jobstore", summary)

            labels = format_archive_health_labels(summary)
            self.assertIn("Sağlıklı:", labels["healthy"])
            self.assertIn("Eksik:", labels["missing"])
            self.assertIn("Onarım gerekli:", labels["repair_needed"])
            self.assertIn("Bozuk:", labels["broken"])
            self.assertIn("Başarısız:", labels["failed"])


class TestFormatLabels(unittest.TestCase):
    def test_empty_summary(self):
        labels = format_archive_health_labels({})
        self.assertEqual(labels["healthy"], "Sağlıklı: 0")
        self.assertEqual(labels["broken"], "Bozuk: 0")


if __name__ == "__main__":
    unittest.main()
