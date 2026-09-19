"""Müşteri / Klasör akıllı arama — keşif, fuzzy, scope filtresi testleri."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.customer_discovery import (
    CustomerRegistry,
    discover_customers_from_db,
    first_folder_under_root,
    normalize_customer_key,
    path_under_any_prefix,
    score_customer_query,
)
from core.settings import AppSettings
from core.utils import normalize_path


class TestCustomerDiscovery(unittest.TestCase):
    def _sample_triples(self):
        root_s = r"\\server\imalat2"
        root_n = r"\\nas\desenler"
        root_d = r"D:\Depo"
        triples = []
        for name, n in (
            ("Ünal Tekstil", 12),
            ("Leydi Tekstil", 8),
            ("Elif Minder", 5),
            ("Modapo ESMODA", 4),
            ("2A TEKSTİL", 6),
            ("Habib Bey", 7),
            ("Nella", 3),
        ):
            for i in range(n):
                triples.append(
                    (
                        rf"{root_s}\{name}\koleksiyon\f{i}.tif",
                        root_s,
                        1,
                    )
                )
        # Alt klasörler
        triples.append((rf"{root_s}\Ünal Tekstil\Yeni Koleksiyon\a.tif", root_s, 1))
        triples.append((rf"{root_s}\Ünal Tekstil\Numuneler\b.tif", root_s, 1))
        # Aynı müşteri farklı kaynak
        triples.append((rf"{root_n}\Ünal Tekstil\x.tif", root_n, 2))
        triples.append((rf"{root_d}\Ünal Tekstil\y.tif", root_d, 3))
        # Junk
        triples.append((rf"{root_s}\14\z.tif", root_s, 1))
        triples.append((rf"{root_s}\1\z.tif", root_s, 1))
        return triples

    def test_01_unal_auto_discover(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        names = {normalize_customer_key(n) for n in reg.names()}
        self.assertIn(normalize_customer_key("Ünal Tekstil"), names)

    def test_02_leydi_auto_discover(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        self.assertTrue(any("leydi" in normalize_customer_key(n) for n in reg.names()))

    def test_03_new_folder_appears(self):
        reg = CustomerRegistry()
        base = self._sample_triples()
        reg.discover_from_paths(base)
        before = reg.discovered_count
        base2 = list(base) + [
            (r"\\server\imalat2\Yeni Musteri ABC\f.tif", r"\\server\imalat2", 1)
        ]
        reg.discover_from_paths(base2)
        self.assertGreater(reg.discovered_count, before)
        self.assertTrue(
            any("yeni musteri abc" == normalize_customer_key(n) for n in reg.names())
        )

    def test_04_case_insensitive(self):
        self.assertGreaterEqual(score_customer_query("UNAL TEKSTIL", "Ünal Tekstil"), 0.88)
        self.assertGreaterEqual(score_customer_query("unal tekstil", "Ünal Tekstil"), 0.88)

    def test_05_turkish_chars(self):
        self.assertEqual(
            normalize_customer_key("Ünal Tekstil"),
            normalize_customer_key("unal tekstil"),
        )
        self.assertGreaterEqual(score_customer_query("ünal tekstil", "Ünal Tekstil"), 0.9)

    def test_06_simple_typo(self):
        sc = score_customer_query("unal teksitl", "Ünal Tekstil")
        self.assertGreaterEqual(sc, 0.72)

    def test_07_fuzzy_suggest(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        sug = reg.suggest("unal teksitl")
        labels = [s.label for s in sug]
        self.assertTrue(any("Ünal" in L or "unal" in L.casefold() for L in labels))
        # Orijinal ad UI'da
        top = sug[0]
        self.assertIn("Tekstil", top.label)

    def test_08_no_false_customer_match(self):
        # "unal" → Nella olmamalı
        sc_nella = score_customer_query("unal", "Nella")
        sc_unal = score_customer_query("unal", "Ünal Tekstil")
        self.assertLess(sc_nella, 0.72)
        self.assertGreaterEqual(sc_unal, 0.72)
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        labels = [s.label for s in reg.suggest("unal")]
        self.assertFalse(any(L == "Nella" for L in labels))

    def test_09_scope_filter_leopar_path(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        path_ok = r"\\server\imalat2\Ünal Tekstil\5000_leopard.tif"
        path_other = r"\\server\imalat2\Leydi Tekstil\leopard.tif"
        self.assertTrue(reg.path_matches(path_ok, "Ünal Tekstil"))
        self.assertFalse(reg.path_matches(path_other, "Ünal Tekstil"))

    def test_10_scope_filter_ekose(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        p = r"\\server\imalat2\Leydi Tekstil\eski\ekose_01.tif"
        self.assertTrue(reg.path_matches(p, "Leydi Tekstil"))
        self.assertFalse(reg.path_matches(p, "Ünal Tekstil"))

    def test_11_no_customer_means_no_scope_restriction(self):
        # Seçim yok → scope_prefixes boş; arama motoru cust boşken filtrelemez
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        self.assertEqual(reg.scope_prefixes_for(""), [])

    def test_12_merge_same_customer_across_sources(self):
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        entry = reg.get("Ünal Tekstil")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertGreaterEqual(len(entry.roots), 2)
        # Path bilgisi korunmalı
        joined = " | ".join(entry.roots)
        self.assertTrue("imalat2" in joined.casefold() or "unal" in joined.casefold())

    def test_13_subdir_keeps_customer_root(self):
        root = r"\\server\imalat2"
        path = r"\\server\imalat2\Ünal Tekstil\eski\dosya.tif"
        self.assertEqual(first_folder_under_root(path, root), "Ünal Tekstil")
        reg = CustomerRegistry()
        reg.discover_from_paths(self._sample_triples())
        self.assertTrue(reg.path_matches(path, "Ünal Tekstil"))

    def test_14_search_intelligence_untouched_import(self):
        # Mevcut intelligence zinciri import edilebilmeli (regresyon dumanı)
        import core.search_intelligence_chain  # noqa: F401
        from core.search_engine import SearchEngine  # noqa: F401

        # customer_discovery motor bağımlılığı içermemeli
        src = Path(__file__).resolve().parents[1] / "core" / "customer_discovery.py"
        text = src.read_text(encoding="utf-8")
        for banned_import in (
            "from core.faiss",
            "import faiss",
            "FeatureExtractor",
            "OpenCLIP",
            "JobStore",
            "pattern_dna",
        ):
            self.assertNotIn(banned_import, text)


class TestCustomerScopeEngine(unittest.TestCase):
    def test_indexed_files_path_filter(self):
        """SearchEngine müşteri filtresi path prefix kullanır (boş customer kolonu)."""
        from core.db import Database
        from core.search_engine import SearchEngine

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cache = base / "cache"
            data = base / "data"
            cache.mkdir()
            data.mkdir()
            db_path = str(data / "patterns.db")
            settings = AppSettings(
                db_path=db_path,
                cache_dir=str(cache),
                faiss_dino_path=str(data / "faiss_dino.index"),
                faiss_clip_path=str(data / "faiss_clip.index"),
                face_db_path=str(data / "face_index.db"),
                face_index_enabled=False,
                ai_embedding_enabled=False,
                ocr_enabled=False,
            )
            db = Database(db_path)
            with db.connect() as conn:
                cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(sources)").fetchall()
                }
                # Kaynak satırı — şema alanlarına uyumlu (is_active)
                if "is_active" in cols:
                    conn.execute(
                        "INSERT INTO sources (id, name, root_path, source_type, is_active) "
                        "VALUES (1, 's', ?, 'server_share', 1)",
                        (r"\\server\imalat2",),
                    )
                else:
                    conn.execute(
                        "INSERT INTO sources (id, name, root_path, source_type) "
                        "VALUES (1, 's', ?, 'server_share')",
                        (r"\\server\imalat2",),
                    )
                try:
                    conn.execute(
                        "INSERT INTO files (path, filename, source_id, status, customer) "
                        "VALUES (?, ?, 1, 'indexed', '')",
                        (r"\\server\imalat2\Ünal Tekstil\leopar.tif", "leopar.tif"),
                    )
                    conn.execute(
                        "INSERT INTO files (path, filename, source_id, status, customer) "
                        "VALUES (?, ?, 1, 'indexed', '')",
                        (r"\\server\imalat2\Leydi Tekstil\ekose.tif", "ekose.tif"),
                    )
                    conn.commit()
                except Exception as exc:
                    self.skipTest(f"files schema uyumsuz: {exc}")

            from core.customer_discovery import set_customer_registry

            reg = discover_customers_from_db(db)
            set_customer_registry(reg)
            eng = SearchEngine(settings, load_ai=False)
            eng._customer_registry = reg
            settings.customer_filter = "Ünal Tekstil"
            files = eng._indexed_files(customer="Ünal Tekstil", lightweight=True)
            paths = [normalize_path(f.get("path", "")) for f in files]
            self.assertEqual(len(files), 1)
            self.assertTrue(
                any(
                    normalize_customer_key("Ünal Tekstil") in normalize_customer_key(p)
                    for p in paths
                )
            )
            self.assertFalse(
                any(
                    normalize_customer_key("Leydi Tekstil") in normalize_customer_key(p)
                    for p in paths
                )
            )
            # Müşteri yok → her ikisi
            settings.customer_filter = ""
            all_f = eng._indexed_files(customer="", lightweight=True)
            self.assertGreaterEqual(len(all_f), 2)


if __name__ == "__main__":
    unittest.main()
