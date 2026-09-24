"""Kalıcı kullanıcı kategori hafızası.

Marka hafızasındaki yaklaşımın genel kategori karşılığıdır:
- exact alias -> otomatik çözüm
- fuzzy öneri -> kullanıcı seçimi
- yeni alt kategori -> SQLite'a kalıcı kayıt
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable

from core.textile_terms import normalize_turkish


def _write_db_path(db_path: str) -> str:
    """Pattern Index'e yazma: production patterns.db → search_memory.db."""
    if not db_path:
        return db_path
    from pathlib import Path

    if Path(db_path).name.lower() == "patterns.db":
        try:
            from core.search_memory import memory_db_path

            return memory_db_path(db_path)
        except Exception:
            return db_path
    return db_path


def _read_db_paths(db_path: str) -> list[str]:
    paths: list[str] = []
    if not db_path:
        return paths
    try:
        from core.search_memory import memory_db_path

        mem = memory_db_path(db_path)
        if mem:
            paths.append(mem)
    except Exception:
        pass
    if db_path not in paths:
        paths.append(db_path)
    return paths


def _conn(db_path: str):
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS category_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent TEXT NOT NULL,
            label TEXT NOT NULL,
            aliases TEXT DEFAULT '[]',
            source TEXT DEFAULT 'user',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT '',
            UNIQUE(parent, label)
        )
    """)
    # Ana kategori de kullanıcı tarafından öğrenilebilir. Ayrı tablo kullanmak
    # mevcut category_memory kayıtlarını ve eski veritabanlarını bozmadan
    # kök düğüm eklememizi sağlar.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS category_memory_roots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            aliases TEXT DEFAULT '[]',
            source TEXT DEFAULT 'user',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category_memory_root_label ON category_memory_roots(label)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_category_memory_parent ON category_memory(parent)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_category_memory_label ON category_memory(label)"
    )
    return conn


def _read_conn(db_path: str):
    """Read-only: never CREATE/migrate Pattern Index sqlite files."""
    from pathlib import Path

    p = Path(db_path)
    if not p.is_file():
        return None
    try:
        uri = f"file:{p.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _key(value: str) -> str:
    try:
        from core.concept_query_normalize import concept_match_key

        return concept_match_key(value)
    except Exception:
        return normalize_turkish(str(value or "")).strip().replace("ı", "i")


def _prefer_label(candidates: Iterable[str]) -> str:
    """Aynı anahtarlı etiketlerden görünen tek adı seç (Title Case / uzun olan)."""
    labs = [" ".join(str(x or "").strip().split()) for x in candidates if str(x or "").strip()]
    if not labs:
        return ""
    # Statik ağaçta varsa onu tercih et
    try:
        from core.category_tree import CATEGORY_TREE

        by_k = {_key(p): p for p in CATEGORY_TREE.keys()}
        for lab in labs:
            hit = by_k.get(_key(lab))
            if hit:
                return hit
    except Exception:
        pass
    return sorted(labs, key=lambda s: (-len(s), s.swapcase(), s))[0]


def register_root_category(
    db_path: str,
    label: str,
    aliases: Iterable[str] = (),
    *,
    source: str = "user",
) -> bool:
    """Kullanıcı tarafından yeni ana kategori oluşturur.

    Aynı anahtar (animal/Animal) varsa yeni satır açılmaz; mevcut köke alias eklenir.
    """
    label = " ".join(str(label or "").strip().split())
    if not db_path or not label:
        return False
    db_path = _write_db_path(db_path)
    vals = [label, *[str(x).strip() for x in aliases if str(x).strip()]]
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _conn(db_path)
        # Case-insensitive çakışma → mevcut köke birleştir
        rows = conn.execute("SELECT id, label, aliases FROM category_memory_roots").fetchall()
        target = None
        for r in rows:
            if _key(r["label"]) == _key(label):
                target = r
                break
        if target is not None:
            old_aliases = []
            try:
                old_aliases = json.loads(target["aliases"] or "[]")
            except Exception:
                old_aliases = []
            merged = list(dict.fromkeys([*[str(x) for x in old_aliases], *vals, target["label"]]))
            # Görünen ad: tercih edilen tek etiket
            display = _prefer_label([target["label"], label])
            conn.execute(
                """UPDATE category_memory_roots
                   SET label=?, aliases=?, updated_at=? WHERE id=?""",
                (
                    display,
                    json.dumps(merged, ensure_ascii=False),
                    now,
                    int(target["id"]),
                ),
            )
            if display != target["label"]:
                conn.execute(
                    "UPDATE category_memory SET parent=? WHERE parent=?",
                    (display, target["label"]),
                )
        else:
            aliases_json = json.dumps(list(dict.fromkeys(vals)), ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO category_memory_roots(label,aliases,source,created_at,updated_at)
                VALUES(?,?,?,?,?)
                """,
                (label, aliases_json, source, now, now),
            )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def dynamic_parents(db_path: str) -> list[str]:
    if not db_path:
        return []
    out: list[str] = []
    seen: set[str] = set()
    buckets: dict[str, list[str]] = {}
    for path in _read_db_paths(db_path):
        conn = _read_conn(path)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                "SELECT label FROM category_memory_roots ORDER BY label COLLATE NOCASE"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            lab = str(r["label"])
            buckets.setdefault(_key(lab), []).append(lab)
    for k, labs in buckets.items():
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(_prefer_label(labs))
    return out


def dynamic_root_rows(db_path: str) -> list[dict]:
    if not db_path:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for path in _read_db_paths(db_path):
        conn = _read_conn(path)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                "SELECT label,aliases FROM category_memory_roots ORDER BY label COLLATE NOCASE"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            lab = str(r["label"])
            k = _key(lab)
            if not k or k in seen:
                continue
            seen.add(k)
            try:
                aliases = json.loads(r["aliases"] or "[]")
            except Exception:
                aliases = []
            out.append({"label": lab, "aliases": aliases})
    return out


def rename_root_category(db_path: str, old_label: str, new_label: str) -> dict:
    """Ana kategori adını düzelt; eski yazımı alias olarak tutar, case çiftlerini birleştirir.

    Index/DNA üretmez. category_memory (+ mümkünse files path) günceller.
    """
    old_label = " ".join(str(old_label or "").strip().split())
    new_label = " ".join(str(new_label or "").strip().split())
    stats = {
        "ok": False,
        "renamed": 0,
        "merged": 0,
        "children_retargeted": 0,
        "files_updated": 0,
        "reason": "",
    }
    if not db_path or not old_label or not new_label:
        stats["reason"] = "empty"
        return stats
    if _key(old_label) == _key(new_label) and old_label == new_label:
        stats["reason"] = "unchanged"
        stats["ok"] = True
        return stats
    write = _write_db_path(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _conn(write)
        rows = conn.execute("SELECT id, label, aliases FROM category_memory_roots").fetchall()
        old_row = None
        new_row = None
        for r in rows:
            if _key(r["label"]) == _key(old_label) or r["label"] == old_label:
                old_row = r
            if _key(r["label"]) == _key(new_label):
                new_row = r
        if old_row is None:
            # Yoksa yeni kök olarak kaydet (düzeltme = oluştur)
            register_root_category(db_path, new_label, aliases=[old_label])
            stats["ok"] = True
            stats["renamed"] = 1
            stats["reason"] = "created"
            conn.close()
            return stats

        old_aliases = []
        try:
            old_aliases = json.loads(old_row["aliases"] or "[]")
        except Exception:
            old_aliases = []

        if new_row is not None and int(new_row["id"]) != int(old_row["id"]):
            # Hedef zaten var → old'u new'e merge et
            new_aliases = []
            try:
                new_aliases = json.loads(new_row["aliases"] or "[]")
            except Exception:
                new_aliases = []
            merged = list(
                dict.fromkeys(
                    [
                        *new_aliases,
                        *old_aliases,
                        old_row["label"],
                        old_label,
                        new_row["label"],
                        new_label,
                    ]
                )
            )
            display = _prefer_label([new_row["label"], new_label])
            conn.execute(
                "UPDATE category_memory_roots SET label=?, aliases=?, updated_at=? WHERE id=?",
                (display, json.dumps(merged, ensure_ascii=False), now, int(new_row["id"])),
            )
            cur = conn.execute(
                "UPDATE category_memory SET parent=? WHERE parent=?",
                (display, old_row["label"]),
            )
            stats["children_retargeted"] = int(cur.rowcount or 0)
            conn.execute("DELETE FROM category_memory_roots WHERE id=?", (int(old_row["id"]),))
            stats["merged"] = 1
            stats["renamed"] = 1
            final = display
        else:
            merged = list(
                dict.fromkeys([*old_aliases, old_row["label"], old_label, new_label])
            )
            conn.execute(
                "UPDATE category_memory_roots SET label=?, aliases=?, updated_at=? WHERE id=?",
                (new_label, json.dumps(merged, ensure_ascii=False), now, int(old_row["id"])),
            )
            cur = conn.execute(
                "UPDATE category_memory SET parent=? WHERE parent=?",
                (new_label, old_row["label"]),
            )
            stats["children_retargeted"] = int(cur.rowcount or 0)
            # Aynı anahtarlı başka kök kalmışsa temizle
            for r in conn.execute("SELECT id, label FROM category_memory_roots").fetchall():
                if int(r["id"]) == int(old_row["id"]):
                    continue
                if _key(r["label"]) == _key(new_label):
                    conn.execute("DELETE FROM category_memory_roots WHERE id=?", (int(r["id"]),))
                    stats["merged"] += 1
            stats["renamed"] = 1
            final = new_label

        conn.commit()
        conn.close()
        stats["ok"] = True
        stats["final"] = final
        # Dosya kategori yollarını güncelle (best-effort; index üretmez)
        stats["files_updated"] = _retarget_file_category_paths(
            db_path, old_row["label"], final
        )
        if old_label != old_row["label"]:
            stats["files_updated"] += _retarget_file_category_paths(
                db_path, old_label, final
            )
        return stats
    except Exception as exc:
        stats["reason"] = str(exc)
        return stats


def rename_child_category(
    db_path: str,
    parent: str,
    old_label: str,
    new_label: str,
) -> dict:
    """Alt kategori adını düzelt (aynı parent altında)."""
    parent = " ".join(str(parent or "").strip().split())
    old_label = " ".join(str(old_label or "").strip().split())
    new_label = " ".join(str(new_label or "").strip().split())
    stats = {"ok": False, "renamed": 0, "merged": 0, "reason": ""}
    if not db_path or not parent or not old_label or not new_label:
        stats["reason"] = "empty"
        return stats
    write = _write_db_path(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _conn(write)
        # parent case variants
        parents = [
            r["parent"]
            for r in conn.execute("SELECT DISTINCT parent FROM category_memory").fetchall()
            if _key(r["parent"]) == _key(parent)
        ] or [parent]
        old_row = None
        for p in parents:
            old_row = conn.execute(
                "SELECT id, parent, label, aliases FROM category_memory WHERE parent=? AND label=?",
                (p, old_label),
            ).fetchone()
            if old_row:
                break
            # key match
            for r in conn.execute(
                "SELECT id, parent, label, aliases FROM category_memory WHERE parent=?",
                (p,),
            ).fetchall():
                if _key(r["label"]) == _key(old_label):
                    old_row = r
                    break
            if old_row:
                break
        if not old_row:
            register_category(db_path, parent, new_label, aliases=[old_label])
            stats["ok"] = True
            stats["renamed"] = 1
            stats["reason"] = "created"
            conn.close()
            return stats
        # conflict?
        conflict = conn.execute(
            "SELECT id, aliases FROM category_memory WHERE parent=? AND label=?",
            (old_row["parent"], new_label),
        ).fetchone()
        old_aliases = []
        try:
            old_aliases = json.loads(old_row["aliases"] or "[]")
        except Exception:
            old_aliases = []
        if conflict and int(conflict["id"]) != int(old_row["id"]):
            new_aliases = []
            try:
                new_aliases = json.loads(conflict["aliases"] or "[]")
            except Exception:
                new_aliases = []
            merged = list(
                dict.fromkeys([*new_aliases, *old_aliases, old_row["label"], old_label, new_label])
            )
            conn.execute(
                "UPDATE category_memory SET aliases=?, updated_at=? WHERE id=?",
                (json.dumps(merged, ensure_ascii=False), now, int(conflict["id"])),
            )
            conn.execute("DELETE FROM category_memory WHERE id=?", (int(old_row["id"]),))
            stats["merged"] = 1
        else:
            merged = list(dict.fromkeys([*old_aliases, old_row["label"], old_label, new_label]))
            conn.execute(
                "UPDATE category_memory SET label=?, aliases=?, updated_at=? WHERE id=?",
                (new_label, json.dumps(merged, ensure_ascii=False), now, int(old_row["id"])),
            )
        stats["renamed"] = 1
        conn.commit()
        conn.close()
        path_old = f"{old_row['parent']}/{old_row['label']}"
        path_new = f"{old_row['parent']}/{new_label}"
        _retarget_file_category_paths(db_path, path_old, path_new)
        stats["ok"] = True
        return stats
    except Exception as exc:
        stats["reason"] = str(exc)
        return stats


def _retarget_file_category_paths(db_path: str, old: str, new: str) -> int:
    """Best-effort path rewrite on patterns.db files table."""
    if not db_path or not old or not new or old == new:
        return 0
    n = 0
    try:
        from pathlib import Path

        p = Path(db_path)
        candidates = []
        if p.name.lower() == "search_memory.db":
            candidates = [p.with_name("patterns.db")]
        elif p.name.lower() == "patterns.db":
            candidates = [p]
        else:
            candidates = [p, p.with_name("patterns.db")]
        for cand in candidates:
            if not cand.is_file():
                continue
            conn = sqlite3.connect(str(cand), timeout=10)
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
                for col in ("manual_category_path", "category_path"):
                    if col not in cols:
                        continue
                    cur = conn.execute(
                        f"UPDATE files SET {col}=? WHERE {col}=?",
                        (new, old),
                    )
                    n += int(cur.rowcount or 0)
                    if "/" not in old:
                        rows = conn.execute(
                            f"SELECT id, {col} FROM files WHERE {col} LIKE ?",
                            (old + "/%",),
                        ).fetchall()
                        for rid, path in rows:
                            s = str(path or "")
                            if s.startswith(old + "/"):
                                conn.execute(
                                    f"UPDATE files SET {col}=? WHERE id=?",
                                    (new + s[len(old) :], rid),
                                )
                                n += 1
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()
    except Exception:
        return n
    return n


def dedupe_category_roots(db_path: str) -> dict:
    """Aynı anahtarlı (animal/Animal) kökleri tek satıra indir."""
    write = _write_db_path(db_path)
    stats = {"merged": 0, "kept": 0}
    if not write:
        return stats
    conn = _conn(write)
    rows = conn.execute("SELECT id, label, aliases FROM category_memory_roots").fetchall()
    by_key: dict[str, list] = {}
    for r in rows:
        by_key.setdefault(_key(r["label"]), []).append(r)
    now = datetime.now(timezone.utc).isoformat()
    for k, group in by_key.items():
        if not k or len(group) < 2:
            stats["kept"] += len(group)
            continue
        keep_lab = _prefer_label([r["label"] for r in group])
        keep = next((r for r in group if r["label"] == keep_lab), None)
        if keep is None:
            keep = group[0]
            keep_lab = str(keep["label"])
        aliases: list[str] = []
        for r in group:
            try:
                aliases.extend(json.loads(r["aliases"] or "[]"))
            except Exception:
                pass
            aliases.append(r["label"])
            if int(r["id"]) != int(keep["id"]):
                conn.execute(
                    "UPDATE category_memory SET parent=? WHERE parent=?",
                    (keep_lab, r["label"]),
                )
                conn.execute("DELETE FROM category_memory_roots WHERE id=?", (int(r["id"]),))
                stats["merged"] += 1
        conn.execute(
            "UPDATE category_memory_roots SET label=?, aliases=?, updated_at=? WHERE id=?",
            (
                keep_lab,
                json.dumps(list(dict.fromkeys(aliases)), ensure_ascii=False),
                now,
                int(keep["id"]),
            ),
        )
        stats["kept"] += 1
    conn.commit()
    conn.close()
    return stats


def register_category(
    db_path: str,
    parent: str,
    label: str,
    aliases: Iterable[str] = (),
    *,
    source: str = "user",
) -> bool:
    parent = " ".join(str(parent or "").strip().split())
    label = " ".join(str(label or "").strip().split())
    if not db_path or not parent or not label:
        return False
    db_path = _write_db_path(db_path)
    if parent.casefold() == "marka":
        try:
            from core.brand_aliases import canonical_brand_display, normalize_brand_key

            label = canonical_brand_display(label, db_path) or label
        except Exception:
            pass
    vals = [label, *[str(x).strip() for x in aliases if str(x).strip()]]
    aliases_json = json.dumps(list(dict.fromkeys(vals)), ensure_ascii=False)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _conn(db_path)
        conn.execute("""
            INSERT INTO category_memory(parent,label,aliases,source,created_at,updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(parent,label) DO UPDATE SET
                aliases=excluded.aliases,
                updated_at=excluded.updated_at
        """, (parent, label, aliases_json, source, now, now))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def all_dynamic_children_grouped(db_path: str) -> dict[str, list[str]]:
    """One-pass load of category_memory children grouped by parent key.

    Marka brand collapsing is left to callers (same as ``child_categories``).
    """
    if not db_path:
        return {}
    grouped: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for path in _read_db_paths(db_path):
        conn = _read_conn(path)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                "SELECT parent, label FROM category_memory ORDER BY label COLLATE NOCASE"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            pk = _key(str(r["parent"] or ""))
            if not pk:
                continue
            lab = str(r["label"] or "")
            lk = _key(lab)
            if not lab or not lk:
                continue
            bucket = seen.setdefault(pk, set())
            if lk in bucket:
                continue
            bucket.add(lk)
            grouped.setdefault(pk, []).append(lab)
    return grouped


def dynamic_children(db_path: str, parent: str) -> list[str]:
    if not db_path or not parent:
        return []
    out = list(all_dynamic_children_grouped(db_path).get(_key(parent), []))
    if parent.casefold() == "marka":
        try:
            from core.brand_aliases import canonical_brand_display, normalize_brand_key

            collapsed: list[str] = []
            seen: set[str] = set()
            for lab in out:
                key = normalize_brand_key(lab)
                if not key or key in seen:
                    continue
                seen.add(key)
                collapsed.append(canonical_brand_display(lab, db_path) or lab)
            return collapsed
        except Exception:
            pass
    return out


def all_dynamic(db_path: str) -> list[dict]:
    if not db_path:
        return []
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in _read_db_paths(db_path):
        conn = _read_conn(path)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                "SELECT parent,label,aliases FROM category_memory ORDER BY parent,label"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            key = (str(r["parent"]), str(r["label"]))
            if key in seen:
                continue
            seen.add(key)
            try:
                aliases = json.loads(r["aliases"] or "[]")
            except Exception:
                aliases = []
            out.append({"parent": r["parent"], "label": r["label"], "aliases": aliases})
    return out


def resolve_dynamic_category(db_path: str, query: str) -> str:
    q = _key(query)
    if not q:
        return ""
    # Önce dinamik ana kategoriler. Alias birden fazla köke işaret ediyorsa
    # sessizce seçim yapma; UI explicit seçim/ekleme akışına geçsin.
    root_hits = []
    for row in dynamic_root_rows(db_path):
        for value in [row["label"], *row.get("aliases", [])]:
            if _key(value) == q:
                root_hits.append(str(row["label"]))
    if len(set(root_hits)) == 1:
        return root_hits[0]
    # Sonra dinamik alt kategoriler.
    out: list[tuple[str, str, float]] = []
    for row in dynamic_root_rows(db_path):
        candidates = [row["label"], *row.get("aliases", [])]
        score = max(
            SequenceMatcher(None, q, _key(v)).ratio()
            for v in candidates if _key(v)
        )
        compact_q = q.replace(" ", "")
        if compact_q and any(compact_q in _key(v).replace(" ", "") for v in candidates):
            score = max(score, 0.96)
        if score >= (0.54 if len(q) <= 5 else 0.60):
            out.append((row["label"], row["label"], score))

    for row in all_dynamic(db_path):
        candidates = [row["label"], *row.get("aliases", [])]
        for value in candidates:
            if _key(value) == q:
                return f'{row["parent"]}/{row["label"]}'
    return ""


def category_suggestions(
    db_path: str,
    query: str,
    *,
    limit: int = 12,
) -> list[tuple[str, str, float]]:
    """(kategori_yolu, görünen_ad, eşleşme) döndürür."""
    q = _key(query)
    if len(q) < 2:
        return []
    out: list[tuple[str, str, float]] = []
    # Öğrenilmiş ana kategoriler de statik ağacın birinci sınıf adayıdır.
    for row in dynamic_root_rows(db_path):
        candidates = [row["label"], *row.get("aliases", [])]
        vals = [_key(v) for v in candidates if _key(v)]
        if not vals:
            continue
        score = max(SequenceMatcher(None, q, v).ratio() for v in vals)
        compact_q = q.replace(" ", "")
        if compact_q and any(compact_q in v.replace(" ", "") for v in vals):
            score = max(score, 0.96)
        if score >= (0.54 if len(q) <= 5 else 0.60):
            out.append((row["label"], row["label"], score))
    try:
        from core.category_tree import CATEGORY_TREE
        for parent, children in CATEGORY_TREE.items():
            for label, aliases in children.items():
                candidates = [label, *aliases]
                score = max(
                    SequenceMatcher(None, q, _key(v)).ratio()
                    for v in candidates if _key(v)
                )
                compact_q = q.replace(" ", "")
                if compact_q and any(
                    compact_q in _key(v).replace(" ", "") for v in candidates
                ):
                    score = max(score, 0.96)
                if score >= (0.54 if len(q) <= 5 else 0.60):
                    out.append((f"{parent}/{label}", label, score))
    except Exception:
        pass

    for row in all_dynamic(db_path):
        candidates = [row["label"], *row.get("aliases", [])]
        score = max(
            SequenceMatcher(None, q, _key(v)).ratio()
            for v in candidates if _key(v)
        )
        compact_q = q.replace(" ", "")
        if compact_q and any(compact_q in _key(v).replace(" ", "") for v in candidates):
            score = max(score, 0.96)
        if score >= (0.54 if len(q) <= 5 else 0.60):
            out.append((f'{row["parent"]}/{row["label"]}', row["label"], score))

    # Aynı yol tekrar edebilir; en yüksek skoru tut.
    best: dict[str, tuple[str, float]] = {}
    for path, label, score in out:
        if path not in best or score > best[path][1]:
            best[path] = (label, score)
    return [
        (path, label, score)
        for path, (label, score) in sorted(
            best.items(), key=lambda x: (-x[1][1], x[0])
        )[:max(1, int(limit))]
    ]
