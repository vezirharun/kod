"""Müşteri Hafızası — customer_key as context/prior/evidence only."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

# conftest stubs load via pytest; for unittest ensure path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "cm_impl"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import conftest  # noqa: F401  — install stubs

from core.concept_registry import (
    _conn,
    _norm_customer_key,
    add_example,
    concepts,
    customer_example_file_ids,
    ensure,
    example_rows_for_concept,
    learn,
    upsert,
)
from core.customer_discovery import (
    CustomerRegistry,
    normalize_customer_key,
    path_under_any_prefix,
)
from core.learned_concept_search import collect_learned_hits, example_file_ids
from core.teach_me import resolve_active_customer_key


class TestCustomerMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "patterns.db")
        ensure(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    # --- discovery / scope still works ---
    def test_customer_discovery_scope_still_works(self):
        reg = CustomerRegistry()
        triples = [
            (r"\\server\imalat2\Ünal Tekstil\koleksiyon\a.tif", r"\\server\imalat2", 1),
            (r"\\server\imalat2\Ünal Tekstil\Numuneler\b.tif", r"\\server\imalat2", 1),
            (r"\\server\imalat2\Leydi Tekstil\c.tif", r"\\server\imalat2", 1),
        ]
        reg.discover_from_paths(triples)
        names = {normalize_customer_key(n) for n in reg.names()}
        self.assertIn(normalize_customer_key("Ünal Tekstil"), names)
        prefixes = reg.scope_prefixes_for("Ünal Tekstil")
        self.assertTrue(prefixes)
        self.assertTrue(
            path_under_any_prefix(
                r"\\server\imalat2\Ünal Tekstil\koleksiyon\a.tif", prefixes
            )
        )

    def test_norm_customer_key_empty_is_global(self):
        self.assertEqual(_norm_customer_key(""), "")
        self.assertEqual(_norm_customer_key(None), "")
        self.assertTrue(_norm_customer_key("Ünal Tekstil"))

    def test_resolve_active_customer_key_explicit(self):
        ck = resolve_active_customer_key("Ünal Tekstil")
        self.assertTrue(ck)
        self.assertEqual(ck, normalize_customer_key("Ünal Tekstil"))

    # --- customer context read with customer_key ---
    def test_customer_context_read_with_customer_key(self):
        cid = learn(
            self.db_path,
            "Leopar",
            file_id=10,
            file_path="/unal/red_leo.tif",
            customer_key="Ünal Tekstil",
        )
        self.assertGreater(cid, 0)
        rows = example_rows_for_concept(
            self.db_path, cid, customer_key="Ünal Tekstil"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file_id"], 10)
        self.assertEqual(
            rows[0]["customer_key"], _norm_customer_key("Ünal Tekstil")
        )

    # --- same concept keeps global canonical ---
    def test_same_concept_keeps_global_canonical(self):
        cid1 = learn(
            self.db_path, "Leopar", file_id=1, customer_key="Ünal Tekstil"
        )
        cid2 = learn(self.db_path, "Leopar", file_id=2, customer_key="")
        self.assertEqual(cid1, cid2)
        rows = concepts(self.db_path)
        leopars = [r for r in rows if r["canonical"] == "Leopar"]
        self.assertEqual(len(leopars), 1)
        self.assertEqual(leopars[0]["id"], cid1)

    # --- customer memory does not overwrite global concept ---
    def test_customer_memory_does_not_overwrite_global_concept(self):
        learn(self.db_path, "Leopar", file_id=1, customer_key="")
        learn(
            self.db_path,
            "Leopar",
            file_id=2,
            file_path="/unal/x.tif",
            customer_key="Ünal Tekstil",
        )
        # Teaching under customer must NOT rename / fork canonical
        cans = [r["canonical"] for r in concepts(self.db_path)]
        self.assertEqual(cans.count("Leopar"), 1)
        self.assertNotIn("Ünal Tekstil Leopar", cans)
        self.assertNotIn("unal tekstil", " ".join(cans).lower())

    # --- user teaching beats customer context where modeled ---
    def test_user_teaching_beats_customer_context(self):
        cid = upsert(self.db_path, "Leopar", concept_type="visual_concept")
        # Global user teach
        add_example(
            self.db_path, cid, file_id=100, role="positive", source="user", customer_key=""
        )
        # Customer soft examples
        add_example(
            self.db_path,
            cid,
            file_id=200,
            role="positive",
            source="user",
            customer_key="Ünal Tekstil",
        )
        # Without customer: only global user teaching
        global_ids = example_file_ids(self.db_path, cid, user_only=True, customer_key="")
        self.assertEqual(global_ids, [100])
        # With customer: customer first, then global fallback (user still present)
        scoped = example_file_ids(
            self.db_path, cid, user_only=True, customer_key="Ünal Tekstil"
        )
        self.assertEqual(scoped[0], 200)
        self.assertIn(100, scoped)
        # Soft-only customer ids helper excludes global
        only_cust = customer_example_file_ids(
            self.db_path, cid, customer_key="Ünal Tekstil"
        )
        self.assertEqual(only_cust, [200])

    # --- without customer, global search path unchanged ---
    def test_without_customer_global_search_path_unchanged(self):
        learn(self.db_path, "Leopar", file_id=11, customer_key="")
        learn(
            self.db_path, "Leopar", file_id=22, customer_key="Leydi Tekstil"
        )
        pack = collect_learned_hits(self.db_path, "leopar", customer_key="")
        self.assertEqual(pack.get("canonical"), "Leopar")
        self.assertIn(11, pack.get("exact_ids") or [])
        self.assertNotIn(22, pack.get("exact_ids") or [])

    # --- with customer, soft preference applies ---
    def test_with_customer_soft_preference_applies(self):
        learn(self.db_path, "Leopar", file_id=11, customer_key="")
        learn(
            self.db_path,
            "Leopar",
            file_id=33,
            customer_key="Ünal Tekstil",
        )
        pack = collect_learned_hits(
            self.db_path, "leopar", customer_key="Ünal Tekstil"
        )
        exact = pack.get("exact_ids") or []
        self.assertTrue(exact)
        # Customer example preferred (listed first)
        self.assertEqual(exact[0], 33)
        self.assertIn(11, exact)
        self.assertEqual(pack.get("customer_key"), _norm_customer_key("Ünal Tekstil"))

    # --- wrong customer does not use another customer's examples ---
    def test_wrong_customer_does_not_use_other_customer_examples(self):
        learn(
            self.db_path, "Leopar", file_id=44, customer_key="Ünal Tekstil"
        )
        learn(
            self.db_path, "Leopar", file_id=55, customer_key="Leydi Tekstil"
        )
        learn(self.db_path, "Leopar", file_id=66, customer_key="")
        ids_unal = example_file_ids(
            self.db_path, concepts(self.db_path)[0]["id"], customer_key="Ünal Tekstil"
        )
        self.assertIn(44, ids_unal)
        self.assertIn(66, ids_unal)
        self.assertNotIn(55, ids_unal)
        ids_leydi = example_file_ids(
            self.db_path, concepts(self.db_path)[0]["id"], customer_key="Leydi Tekstil"
        )
        self.assertIn(55, ids_leydi)
        self.assertNotIn(44, ids_leydi)

    # --- no duplicate rows for same (concept, file, customer_key) ---
    def test_no_duplicate_rows_same_concept_file_customer(self):
        cid = learn(
            self.db_path,
            "Leopar",
            file_id=7,
            file_path="/a.tif",
            customer_key="Ünal Tekstil",
        )
        learn(
            self.db_path,
            "Leopar",
            file_id=7,
            file_path="/a.tif",
            customer_key="Ünal Tekstil",
        )
        # Global row for same file is allowed (different customer_key)
        learn(
            self.db_path,
            "Leopar",
            file_id=7,
            file_path="/a.tif",
            customer_key="",
        )
        c = _conn(self.db_path)
        n_cust = c.execute(
            """SELECT COUNT(*) n FROM concept_examples
               WHERE concept_id=? AND file_id=? AND IFNULL(customer_key,'')=?""",
            (cid, 7, _norm_customer_key("Ünal Tekstil")),
        ).fetchone()["n"]
        n_glob = c.execute(
            """SELECT COUNT(*) n FROM concept_examples
               WHERE concept_id=? AND file_id=? AND IFNULL(customer_key,'')=''""",
            (cid, 7),
        ).fetchone()["n"]
        c.close()
        self.assertEqual(int(n_cust), 1)
        self.assertEqual(int(n_glob), 1)

    # --- restart/persistence via DB schema ---
    def test_restart_persistence_via_db_schema(self):
        learn(
            self.db_path,
            "Leopar",
            file_id=9,
            file_path="/p.tif",
            customer_key="Ünal Tekstil",
        )
        # Re-open connection (simulates restart)
        ensure(self.db_path)
        c = _conn(self.db_path)
        cols = {str(r[1]) for r in c.execute("PRAGMA table_info(concept_examples)")}
        self.assertIn("customer_key", cols)
        row = c.execute(
            """SELECT customer_key, file_id FROM concept_examples
               WHERE file_id=9"""
        ).fetchone()
        c.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["customer_key"], _norm_customer_key("Ünal Tekstil"))
        self.assertEqual(int(row["file_id"]), 9)

    def test_migration_from_legacy_table_without_customer_key(self):
        """Legacy UNIQUE(concept_id,file_id,role,file_path) → rebuild with scope."""
        legacy = str(Path(self.tmp.name) / "legacy.db")
        c = sqlite3.connect(legacy)
        c.execute(
            """CREATE TABLE concept_registry(
                id INTEGER PRIMARY KEY AUTOINCREMENT, canonical TEXT NOT NULL UNIQUE,
                concept_type TEXT DEFAULT 'unknown', parent TEXT DEFAULT '',
                aliases TEXT DEFAULT '[]', confidence REAL DEFAULT 0.5,
                status TEXT DEFAULT 'learned', source TEXT DEFAULT 'user',
                created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')"""
        )
        c.execute(
            """CREATE TABLE concept_examples(
                id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id INTEGER NOT NULL,
                file_id INTEGER DEFAULT 0, file_path TEXT DEFAULT '',
                role TEXT NOT NULL, source TEXT DEFAULT 'user',
                created_at TEXT DEFAULT '',
                UNIQUE(concept_id,file_id,role,file_path))"""
        )
        c.execute(
            "INSERT INTO concept_registry(canonical,aliases) VALUES('Leopar','[]')"
        )
        c.execute(
            """INSERT INTO concept_examples(concept_id,file_id,file_path,role,source,created_at)
               VALUES(1,1,'/g.tif','positive','user','t')"""
        )
        c.commit()
        c.close()
        # Opening via _conn runs migration
        ensure(legacy)
        learn(legacy, "Leopar", file_id=1, file_path="/g.tif", customer_key="Ünal Tekstil")
        learn(legacy, "Leopar", file_id=1, file_path="/g.tif", customer_key="")
        c = _conn(legacy)
        cols = {str(r[1]) for r in c.execute("PRAGMA table_info(concept_examples)")}
        self.assertIn("customer_key", cols)
        n = c.execute("SELECT COUNT(*) n FROM concept_examples WHERE file_id=1").fetchone()["n"]
        c.close()
        # global + customer scoped rows both present
        self.assertGreaterEqual(int(n), 2)

    def test_search_memory_scoped_query_key(self):
        from core.search_memory import scoped_query_key

        a = scoped_query_key("kirmizi leopar", "Ünal Tekstil")
        b = scoped_query_key("kirmizi leopar", "Leydi Tekstil")
        c = scoped_query_key("kirmizi leopar", "")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.endswith("cust:" + normalize_customer_key("Ünal Tekstil")) or "cust:" in a)


if __name__ == "__main__":
    unittest.main()
