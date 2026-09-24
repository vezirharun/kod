"""Vezir Concept Brain - açık uçlu kavram hafızası.

Kategori listesinden bağımsız, kullanıcı geri bildirimlerini kavram bazında
kalıcılaştırır. Mevcut arama/indeks motorunu değiştirmez; yalnızca güvenli bir
öğrenme kaydı ve örnek/prototip deposu sağlar.
"""
from __future__ import annotations
import json, sqlite3, hashlib
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable

def _now(): return datetime.now(timezone.utc).isoformat()
def _norm(s):
    try:
        from core.concept_query_normalize import concept_match_key

        return concept_match_key(s)
    except Exception:
        return " ".join(str(s or "").strip().lower().split())

def _norm_customer_key(customer_key):
    """Empty string = global scope. Never merges into concept canonical."""
    raw=" ".join(str(customer_key or "").strip().split())
    if not raw:
        return ""
    try:
        from core.customer_discovery import normalize_customer_key
        return normalize_customer_key(raw) or raw.casefold()
    except Exception:
        return raw.casefold()

def _migrate_concept_examples_customer_unique(c):
    """Ensure UNIQUE includes customer_key (rebuild when legacy UNIQUE blocks scoped rows)."""
    try:
        cols=[str(r[1]) for r in c.execute("PRAGMA table_info(concept_examples)")]
    except Exception:
        return
    if "customer_key" not in cols:
        return
    need_rebuild=False
    try:
        # Legacy table UNIQUE(concept_id,file_id,role,file_path) omits customer_key.
        for idx in c.execute("PRAGMA index_list(concept_examples)"):
            # idx: seq, name, unique, origin, partial
            if not idx[2]:
                continue
            info=list(c.execute(f"PRAGMA index_info({idx[1]})"))
            names=[]
            for ii in info:
                cid=int(ii[1])
                if 0<=cid<len(cols):
                    names.append(cols[cid])
            if names and "customer_key" not in names and set(names)>= {"concept_id","file_id","role","file_path"}:
                need_rebuild=True
                break
    except Exception:
        need_rebuild=False
    if not need_rebuild:
        # Fresh DBs already have customer_key in CREATE UNIQUE — nothing to do.
        # Also cover ALTER-only DBs with no unique index listing customer_key:
        try:
            c.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_examples_scope
                   ON concept_examples(concept_id,file_id,role,file_path,IFNULL(customer_key,''))"""
            )
        except sqlite3.OperationalError:
            # Duplicate rows under old unique shape → must rebuild
            need_rebuild=True
        except Exception:
            pass
    if not need_rebuild:
        try:
            c.execute("UPDATE concept_examples SET customer_key='' WHERE customer_key IS NULL")
        except Exception:
            pass
        return
    # Rebuild with scoped UNIQUE
    c.execute("""CREATE TABLE IF NOT EXISTS concept_examples__cm(
        id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id INTEGER NOT NULL,
        file_id INTEGER DEFAULT 0, file_path TEXT DEFAULT '',
        role TEXT NOT NULL, source TEXT DEFAULT 'user',
        created_at TEXT DEFAULT '',
        embedding BLOB,
        embedding_dim INTEGER DEFAULT 0,
        embedding_backend TEXT DEFAULT '',
        customer_key TEXT DEFAULT '',
        UNIQUE(concept_id,file_id,role,file_path,customer_key))""")
    colset=set(cols)
    sel_cols=["concept_id","file_id","file_path","role","source","created_at"]
    for opt in ("embedding","embedding_dim","embedding_backend","customer_key"):
        if opt in colset:
            sel_cols.append(opt)
    # Deduplicate by scoped key while preferring user source
    rows=c.execute(
        f"SELECT id, {', '.join(sel_cols)} FROM concept_examples ORDER BY "
        "CASE WHEN IFNULL(source,'user')='user' THEN 0 ELSE 1 END, id ASC"
    ).fetchall()
    seen=set()
    for r in rows:
        d={k: r[k] for k in sel_cols}
        if "customer_key" not in d:
            d["customer_key"]=""
        ck=str(d.get("customer_key") or "")
        d["customer_key"]=ck
        key=(int(d["concept_id"]), int(d.get("file_id") or 0), str(d.get("role") or ""),
             str(d.get("file_path") or ""), ck)
        if key in seen:
            continue
        seen.add(key)
        use_cols=list(sel_cols) if "customer_key" in sel_cols else list(sel_cols)+["customer_key"]
        placeholders=",".join("?"*len(use_cols))
        c.execute(
            f"INSERT OR IGNORE INTO concept_examples__cm({', '.join(use_cols)}) VALUES({placeholders})",
            tuple(d.get(k,"") for k in use_cols),
        )
    c.execute("DROP TABLE concept_examples")
    c.execute("ALTER TABLE concept_examples__cm RENAME TO concept_examples")
    c.execute("CREATE INDEX IF NOT EXISTS idx_concept_examples ON concept_examples(concept_id,role)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_concept_examples_customer ON concept_examples(concept_id,customer_key,role)")

def _write_path(db_path):
    """INDEX_FROZEN: persist concepts in search_memory.db, not patterns.db."""
    try:
        from core.index_freeze import INDEX_FROZEN
        if INDEX_FROZEN:
            from core.search_memory import memory_db_path
            return memory_db_path(db_path)
    except Exception:
        pass
    return db_path
def _conn(db_path):
    c=sqlite3.connect(str(_write_path(db_path)), timeout=10); c.row_factory=sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=DELETE")
    except Exception:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS concept_registry(
        id INTEGER PRIMARY KEY AUTOINCREMENT, canonical TEXT NOT NULL UNIQUE,
        concept_type TEXT DEFAULT 'unknown', parent TEXT DEFAULT '',
        aliases TEXT DEFAULT '[]', confidence REAL DEFAULT 0.5,
        status TEXT DEFAULT 'learned', source TEXT DEFAULT 'user',
        created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS concept_examples(
        id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id INTEGER NOT NULL,
        file_id INTEGER DEFAULT 0, file_path TEXT DEFAULT '',
        role TEXT NOT NULL, source TEXT DEFAULT 'user',
        created_at TEXT DEFAULT '',
        customer_key TEXT DEFAULT '',
        UNIQUE(concept_id,file_id,role,file_path,customer_key))""")
    c.execute("""CREATE TABLE IF NOT EXISTS concept_feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id INTEGER NOT NULL,
        file_id INTEGER DEFAULT 0, role TEXT NOT NULL,
        note TEXT DEFAULT '', created_at TEXT DEFAULT '')""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_concept_canonical ON concept_registry(canonical)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_concept_examples ON concept_examples(concept_id,role)")
    ex_cols={str(r[1]) for r in c.execute("PRAGMA table_info(concept_examples)")}
    for name, spec in (
        ("embedding","BLOB"),
        ("embedding_dim","INTEGER DEFAULT 0"),
        ("embedding_backend","TEXT DEFAULT ''"),
        ("customer_key","TEXT DEFAULT ''"),
    ):
        if name not in ex_cols:
            c.execute(f"ALTER TABLE concept_examples ADD COLUMN {name} {spec}")
    _migrate_concept_examples_customer_unique(c)
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_concept_examples_customer ON concept_examples(concept_id,customer_key,role)")
    except Exception:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS teach_me_dismissed(
        file_id INTEGER PRIMARY KEY,
        created_at TEXT DEFAULT '',
        note TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS autonomous_clusters(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suggested_name TEXT DEFAULT '',
        file_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        reason TEXT DEFAULT '',
        created_at TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS autonomous_review(
        file_id INTEGER PRIMARY KEY,
        lane TEXT NOT NULL,
        suggested TEXT DEFAULT '',
        confidence REAL DEFAULT 0,
        rivals_json TEXT DEFAULT '[]',
        reason TEXT DEFAULT '',
        cluster_id INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT '')""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_auto_review_status ON autonomous_review(status,lane)")
    c.execute("""CREATE TABLE IF NOT EXISTS learning_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT '',
        file_id INTEGER DEFAULT 0,
        concept_id INTEGER DEFAULT 0,
        action TEXT NOT NULL,
        confidence REAL DEFAULT 0,
        source TEXT DEFAULT 'auto',
        anchor_id INTEGER DEFAULT 0,
        rival_concept_id INTEGER DEFAULT 0,
        reason TEXT DEFAULT '',
        UNIQUE(file_id, concept_id, action, source))""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_learning_events_file ON learning_events(file_id)")
    return c

def ensure(db_path):
    c=_conn(db_path); c.commit(); c.close()

def _row_by_canonical(c, canonical):
    n=_norm(canonical)
    row=c.execute("SELECT id,aliases,confidence,canonical FROM concept_registry WHERE canonical=?",(canonical,)).fetchone()
    if row: return row
    for r in c.execute("SELECT id,aliases,confidence,canonical FROM concept_registry"):
        if _norm(r["canonical"])==n: return r
    return None

def upsert(db_path, canonical, *, concept_type="unknown", parent="", aliases=(), confidence=None, source="user"):
    canonical=" ".join(str(canonical or "").strip().split())
    if not db_path or not canonical: return 0
    aliases=list(dict.fromkeys([canonical,*[str(x).strip() for x in aliases if str(x).strip()]]))
    c=_conn(db_path); now=_now()
    row=_row_by_canonical(c, canonical)
    if row:
        old=json.loads(row["aliases"] or "[]")
        merged=list(dict.fromkeys(old+aliases))
        conf=float(row["confidence"] if confidence is None else confidence)
        c.execute("""UPDATE concept_registry SET concept_type=COALESCE(NULLIF(?,'unknown'),concept_type),
            parent=COALESCE(NULLIF(? ,''),parent), aliases=?, confidence=?, updated_at=? WHERE id=?""",
            (concept_type,parent,json.dumps(merged,ensure_ascii=False),conf,now,row["id"]))
        cid=int(row["id"])
    else:
        c.execute("""INSERT INTO concept_registry(canonical,concept_type,parent,aliases,confidence,status,source,created_at,updated_at)
            VALUES(?,?,?,?,?,'active',?,?,?)""",
            (canonical,concept_type,parent,json.dumps(aliases,ensure_ascii=False),
             0.5 if confidence is None else float(confidence),source,now,now))
        cid=int(c.execute("SELECT last_insert_rowid()").fetchone()[0])
    c.commit(); c.close(); return cid

def add_example(db_path, concept_id, *, file_id=0, file_path="", role="positive", source="user",
                embedding=None, embedding_backend="", customer_key=""):
    if not db_path or not concept_id: return False
    role="negative" if role=="negative" else "positive"
    src=_norm_source(source)
    ck=_norm_customer_key(customer_key)
    blob=bytes(embedding) if embedding else None
    dim=0
    if blob:
        dim=max(0, len(blob)//4)
    c=_conn(db_path)
    fid=int(file_id or 0)
    fpath=str(file_path or "")
    # user her zaman kazanır; auto/candidate mevcut user satırını ezemez.
    # Scope: aynı customer_key ('' = global) içinde bak.
    existing=c.execute(
        """SELECT id, source, file_path FROM concept_examples
           WHERE concept_id=? AND file_id=? AND role=?
             AND IFNULL(customer_key,'')=?
           ORDER BY CASE WHEN IFNULL(source,'user')='user' THEN 0 ELSE 1 END, id ASC""",
        (int(concept_id), fid, role, ck),
    ).fetchone()
    if existing:
        old_src=_norm_source(existing["source"])
        if old_src=="user" and src!="user":
            c.close()
            return False
        if src=="user" and old_src!="user":
            c.execute(
                "UPDATE concept_examples SET source='user' WHERE id=?",
                (int(existing["id"]),),
            )
    # Aynı file+scope için user satırı varsa (farklı path ile) auto/candidate yazma.
    if src!="user" and fid>0:
        user_row=c.execute(
            """SELECT id FROM concept_examples
               WHERE concept_id=? AND file_id=? AND role=?
                 AND IFNULL(customer_key,'')=?
                 AND IFNULL(source,'user')='user'""",
            (int(concept_id), fid, role, ck),
        ).fetchone()
        if user_row:
            c.close()
            return False
    c.execute("""INSERT OR IGNORE INTO concept_examples
        (concept_id,file_id,file_path,role,source,created_at,embedding,embedding_dim,embedding_backend,customer_key)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (int(concept_id),fid,fpath,role,src,_now(),
         blob, dim, str(embedding_backend or ""), ck))
    if blob:
        # Embedding yalnızca boşsa dolar; user satırını auto ile yeniden yazmaz.
        if src=="user":
            c.execute("""UPDATE concept_examples SET embedding=?, embedding_dim=?, embedding_backend=?
                WHERE concept_id=? AND file_id=? AND role=? AND file_path=?
                  AND IFNULL(customer_key,'')=?
                  AND (embedding IS NULL OR length(embedding)=0)""",
                (blob, dim, str(embedding_backend or ""), int(concept_id), fid, role, fpath, ck))
        else:
            c.execute("""UPDATE concept_examples SET embedding=?, embedding_dim=?, embedding_backend=?
                WHERE concept_id=? AND file_id=? AND role=? AND file_path=?
                  AND IFNULL(customer_key,'')=?
                  AND IFNULL(source,'user')!='user'
                  AND (embedding IS NULL OR length(embedding)=0)""",
                (blob, dim, str(embedding_backend or ""), int(concept_id), fid, role, fpath, ck))
    pos=c.execute("SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='positive'",(concept_id,)).fetchone()["n"]
    neg=c.execute("SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='negative'",(concept_id,)).fetchone()["n"]
    total=pos+neg
    conf=min(0.98, 0.5 + (0.08*pos) - (0.05*neg)) if total else 0.5
    c.execute("UPDATE concept_registry SET confidence=?,updated_at=? WHERE id=?",(conf,_now(),int(concept_id)))
    c.commit(); c.close(); return True

def _norm_source(source):
    s=str(source or "user").strip().lower()
    if s in ("auto","autonomous"):
        return "auto"
    if s in ("candidate","review","pending"):
        return "candidate"
    return "user"

def record_learning_event(
    db_path,
    *,
    file_id=0,
    concept_id=0,
    action="",
    confidence=0.0,
    source="auto",
    anchor_id=0,
    rival_concept_id=0,
    reason="",
):
    """Append-only learning memory. Duplicate (file,concept,action,source) ignored."""
    act=str(action or "").strip()
    if not db_path or not act:
        return 0
    c=_conn(db_path)
    cur=c.execute(
        """INSERT OR IGNORE INTO learning_events(
            created_at,file_id,concept_id,action,confidence,source,anchor_id,rival_concept_id,reason)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            _now(),
            int(file_id or 0),
            int(concept_id or 0),
            act,
            float(confidence or 0),
            _norm_source(source),
            int(anchor_id or 0),
            int(rival_concept_id or 0),
            str(reason or "")[:500],
        ),
    )
    eid=int(cur.lastrowid or 0)
    c.commit()
    c.close()
    return eid

def list_learning_events(db_path, *, file_id=0, concept_id=0, limit=50):
    c=_conn(db_path)
    try:
        if int(file_id or 0)>0:
            rows=c.execute(
                "SELECT * FROM learning_events WHERE file_id=? ORDER BY id DESC LIMIT ?",
                (int(file_id), int(limit)),
            ).fetchall()
        elif int(concept_id or 0)>0:
            rows=c.execute(
                "SELECT * FROM learning_events WHERE concept_id=? ORDER BY id DESC LIMIT ?",
                (int(concept_id), int(limit)),
            ).fetchall()
        else:
            rows=c.execute(
                "SELECT * FROM learning_events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    return [dict(r) for r in rows]

def user_positive_file_ids(db_path):
    """Yalnız user verified positive file_id'ler — auto/candidate dahil değil."""
    c=_conn(db_path)
    try:
        rows=c.execute(
            """SELECT DISTINCT file_id FROM concept_examples
               WHERE file_id>0 AND role='positive'
                 AND IFNULL(source,'user') NOT IN ('auto','autonomous','candidate')"""
        ).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    return {int(r["file_id"]) for r in rows}

def learn(db_path, label, *, file_id=0, file_path="", parent="", concept_type="attribute", role="positive", aliases=(),
          embedding=None, embedding_backend="", source="user", customer_key=""):
    src=_norm_source(source)
    parent_name=" ".join(str(parent or "").strip().split())
    if parent_name and src=="user":
        ensure_parent_concept(db_path, parent_name, source="user")
    # Canonical identity is always global — customer_key stamps examples only.
    cid=upsert(db_path,label,concept_type=concept_type,parent=parent_name,aliases=aliases,source=src)
    if cid: add_example(db_path,cid,file_id=file_id,file_path=file_path,role=role,source=src,
                        embedding=embedding, embedding_backend=embedding_backend,
                        customer_key=customer_key)
    return cid

def ensure_parent_concept(db_path, parent_label, *, aliases=(), source="user"):
    """Parent shell concept — child örneklerini çalmaz; eşanlamlıları birleştirir."""
    name=" ".join(str(parent_label or "").strip().split())
    if not db_path or not name:
        return 0
    extra=list(aliases or ())
    try:
        from core.textile_terms import parent_synonym_keys

        for t in parent_synonym_keys(name):
            t=" ".join(str(t or "").strip().split())
            if t and _norm(t)!=_norm(name):
                extra.append(t)
    except Exception:
        pass
    return upsert(
        db_path,
        name,
        concept_type="parent_group",
        aliases=extra,
        source=source,
    )

def children_of_parent(db_path, parent_label, *, user_verified_only=True):
    """parent alanına bağlı child concept satırları (canonical, id, source)."""
    want=_norm(parent_label)
    if not db_path or len(want)<2:
        return []
    keys={want}
    try:
        from core.textile_terms import parent_synonym_keys

        keys|=parent_synonym_keys(parent_label)
    except Exception:
        pass
    rows=[]
    for r in concepts(db_path):
        status=str(r.get("status") or "active")
        if status in ("inactive","retired"):
            continue
        p=_norm(r.get("parent") or "")
        if not p or p not in keys:
            continue
        # Child kendisi parent grubu ise atla
        if _norm(r.get("canonical") or "") in keys:
            continue
        rows.append(r)
    if not user_verified_only:
        return rows
    # En az bir user positive'i olan child'lar
    out=[]
    c=_conn(db_path)
    try:
        for r in rows:
            cid=int(r["id"])
            n=c.execute(
                """SELECT COUNT(*) n FROM concept_examples
                   WHERE concept_id=? AND role='positive' AND file_id>0
                     AND IFNULL(source,'user') NOT IN ('auto','autonomous','candidate')""",
                (cid,),
            ).fetchone()["n"]
            if int(n or 0)>0:
                out.append(r)
    except Exception:
        out=rows
    c.close()
    return out

def _refresh_confidence(c, concept_id):
    pos=c.execute("SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='positive'",(concept_id,)).fetchone()["n"]
    neg=c.execute("SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='negative'",(concept_id,)).fetchone()["n"]
    total=pos+neg
    conf=min(0.98, 0.5 + (0.08*pos) - (0.05*neg)) if total else 0.5
    c.execute("UPDATE concept_registry SET confidence=?,updated_at=? WHERE id=?",(conf,_now(),int(concept_id)))
    return conf

def positives_for_file(db_path, file_id):
    """file_id → positive concept rows (id, canonical, source). Bulk silmez."""
    fid=int(file_id or 0)
    if not db_path or fid<=0: return []
    c=_conn(db_path)
    try:
        rows=c.execute("""SELECT e.concept_id, r.canonical, e.source, e.file_path,
            e.embedding, e.embedding_dim, e.embedding_backend
            FROM concept_examples e JOIN concept_registry r ON r.id=e.concept_id
            WHERE e.file_id=? AND e.role='positive'
              AND IFNULL(r.status,'active') NOT IN ('inactive','retired')""",
            (fid,)).fetchall()
    except Exception:
        rows=[]
    c.close()
    out=[]
    for r in rows:
        out.append({
            "concept_id": int(r["concept_id"]),
            "canonical": str(r["canonical"] or ""),
            "source": str(r["source"] or ""),
            "file_path": str(r["file_path"] or ""),
            "embedding": bytes(r["embedding"]) if r["embedding"] else None,
            "embedding_dim": int(r["embedding_dim"] or 0),
            "embedding_backend": str(r["embedding_backend"] or ""),
        })
    return out

def demote_positive(db_path, concept_id, *, file_id=0, file_path="", keep_as_negative=True):
    """Tek dosyanın positive üyeliğini kaldır; isteğe bağlı negative boundary yaz.

    Diğer concept örneklerine dokunmaz. Toplu silme yapmaz.
    """
    cid=int(concept_id or 0)
    fid=int(file_id or 0)
    if not db_path or cid<=0 or fid<=0: return False
    c=_conn(db_path)
    rows=c.execute(
        """SELECT id, file_path, embedding, embedding_dim, embedding_backend FROM concept_examples
           WHERE concept_id=? AND file_id=? AND role='positive'""",
        (cid, fid),
    ).fetchall()
    if not rows and str(file_path or "").strip():
        rows=c.execute(
            """SELECT id, file_path, embedding, embedding_dim, embedding_backend FROM concept_examples
               WHERE concept_id=? AND file_path=? AND role='positive'""",
            (cid, str(file_path)),
        ).fetchall()
    if not rows:
        c.close()
        return False
    blob=None
    backend=""
    path_kept=str(file_path or "")
    for r in rows:
        if r["embedding"] and not blob:
            blob=bytes(r["embedding"])
            backend=str(r["embedding_backend"] or "")
        if not path_kept:
            path_kept=str(r["file_path"] or "")
        c.execute("DELETE FROM concept_examples WHERE id=?", (int(r["id"]),))
    _refresh_confidence(c, cid)
    c.commit()
    c.close()
    if keep_as_negative:
        add_example(
            db_path, cid, file_id=fid, file_path=path_kept, role="negative",
            source="user", embedding=blob, embedding_backend=backend or ("clip" if blob else ""),
        )
    return True

def demote_file_rivals(db_path, file_id, *, keep_canonical="", keep_concept_id=0, keep_as_negative=True):
    """Dosyayı keep edilen concept dışındaki tüm positive concept'lerden demote et."""
    fid=int(file_id or 0)
    if not db_path or fid<=0: return 0
    keep_id=int(keep_concept_id or 0)
    keep_n=_norm(keep_canonical)
    n=0
    for row in positives_for_file(db_path, fid):
        cid=int(row["concept_id"])
        if keep_id and cid==keep_id:
            continue
        if keep_n and _norm(row.get("canonical") or "")==keep_n:
            continue
        if demote_positive(
            db_path, cid, file_id=fid, file_path=str(row.get("file_path") or ""),
            keep_as_negative=keep_as_negative,
        ):
            n+=1
    return n

def find(db_path, query, limit=8):
    q=_norm(query)
    if not db_path or len(q)<2: return []
    from pathlib import Path
    path=str(_write_path(db_path))
    try:
        from core.index_freeze import INDEX_FROZEN, in_search_session, process_search_active
        search_ro=INDEX_FROZEN and (in_search_session() or process_search_active())
    except Exception:
        search_ro=False
    if search_ro:
        if not Path(path).is_file():
            return []
        uri=Path(path).resolve().as_posix()
        c=sqlite3.connect(f"file:{uri}?mode=ro", uri=True, timeout=10)
        c.row_factory=sqlite3.Row
    else:
        c=_conn(db_path)
    try:
        rows=c.execute("SELECT * FROM concept_registry").fetchall()
    except sqlite3.OperationalError:
        rows=[]
    finally:
        c.close()
    out=[]
    for r in rows:
        vals=[r["canonical"]]+json.loads(r["aliases"] or "[]")
        score=max(SequenceMatcher(None,q,_norm(v)).ratio() for v in vals if _norm(v))
        if q in _norm(r["canonical"]): score=max(score,.96)
        if score>=.58: out.append((r,score))
    out.sort(key=lambda x:-x[1])
    return [{"id":int(r["id"]),"canonical":r["canonical"],"concept_type":r["concept_type"],
             "parent":r["parent"],"confidence":float(r["confidence"]),"score":s} for r,s in out[:limit]]

def concepts(db_path):
    c=_conn(db_path); rows=c.execute("SELECT * FROM concept_registry ORDER BY canonical").fetchall(); c.close()
    return [dict(r) for r in rows]

def concepts_readonly(db_path):
    """SELECT concept rows without CREATE/migrate — safe for UI category tree.

    Falls back to ``concepts()`` (migrating writer) if the RO path is unavailable.
    """
    path = _write_path(db_path)
    if not path:
        return []
    try:
        from pathlib import Path as _P

        if not _P(path).is_file():
            return []
        uri = f"file:{path}?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute(
                "SELECT canonical, parent, status, concept_type, confidence "
                "FROM concept_registry ORDER BY canonical"
            ).fetchall()
        finally:
            c.close()
        return [dict(r) for r in rows]
    except Exception:
        try:
            return concepts(db_path)
        except Exception:
            return []

def set_status(db_path, concept_id, status):
    st=str(status or "active").strip().lower()
    if st in ("learned","active"): st="active"
    if st not in ("active","inactive"): st="active"
    if not db_path or not concept_id: return False
    c=_conn(db_path)
    c.execute("UPDATE concept_registry SET status=?, updated_at=? WHERE id=?",(st,_now(),int(concept_id)))
    c.commit(); c.close(); return True

def example_count(db_path, concept_id, role="positive"):
    c=_conn(db_path)
    n=c.execute("SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role=?",
                (int(concept_id), role)).fetchone()["n"]
    c.close(); return int(n or 0)

def taught_file_ids(db_path):
    """User + auto positive file ids. Candidate review'da kalır — skip listesine girmez."""
    c=_conn(db_path)
    rows=c.execute(
        """SELECT DISTINCT file_id FROM concept_examples
           WHERE file_id>0 AND role='positive'
             AND IFNULL(source,'user') NOT IN ('candidate')"""
    ).fetchall()
    c.close(); return {int(r["file_id"]) for r in rows}

def pending_review_file_ids(db_path):
    c=_conn(db_path)
    try:
        rows=c.execute(
            "SELECT file_id FROM autonomous_review WHERE status='pending' AND file_id>0"
        ).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    return {int(r["file_id"]) for r in rows}

def dismiss_file(db_path, file_id, note=""):
    if not db_path or int(file_id or 0)<=0: return False
    c=_conn(db_path)
    c.execute("INSERT OR REPLACE INTO teach_me_dismissed(file_id,created_at,note) VALUES(?,?,?)",
              (int(file_id),_now(),str(note or "")))
    c.commit(); c.close(); return True

def dismissed_file_ids(db_path):
    c=_conn(db_path)
    try:
        rows=c.execute("SELECT file_id FROM teach_me_dismissed").fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close(); return {int(r["file_id"]) for r in rows}

def positive_example_vectors(db_path):
    """Active concepts → CLIP blobs for later comparison. No model training."""
    c=_conn(db_path)
    try:
        rows=c.execute("""SELECT e.concept_id, r.canonical, r.status, e.file_id, e.embedding, e.embedding_dim
            FROM concept_examples e JOIN concept_registry r ON r.id=e.concept_id
            WHERE e.role='positive' AND e.embedding IS NOT NULL AND length(e.embedding)>0
              AND IFNULL(r.status,'active') NOT IN ('inactive','retired')
              AND IFNULL(e.source,'user') NOT IN ('auto','autonomous','candidate')""").fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    out=[]
    for r in rows:
        out.append({"concept_id":int(r["concept_id"]),"canonical":r["canonical"],
                    "file_id":int(r["file_id"] or 0),"embedding":bytes(r["embedding"]),
                    "embedding_dim":int(r["embedding_dim"] or 0)})
    return out

def negative_example_vectors(db_path, concept_id=None):
    """Boundary negatives with CLIP blobs. Optional filter by concept_id."""
    c=_conn(db_path)
    try:
        if concept_id:
            rows=c.execute("""SELECT e.concept_id, r.canonical, e.file_id, e.embedding, e.embedding_dim
                FROM concept_examples e JOIN concept_registry r ON r.id=e.concept_id
                WHERE e.role='negative' AND e.concept_id=?
                  AND e.embedding IS NOT NULL AND length(e.embedding)>0
                  AND IFNULL(r.status,'active') NOT IN ('inactive','retired')""",
                (int(concept_id),)).fetchall()
        else:
            rows=c.execute("""SELECT e.concept_id, r.canonical, e.file_id, e.embedding, e.embedding_dim
                FROM concept_examples e JOIN concept_registry r ON r.id=e.concept_id
                WHERE e.role='negative' AND e.embedding IS NOT NULL AND length(e.embedding)>0
                  AND IFNULL(r.status,'active') NOT IN ('inactive','retired')""").fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    out=[]
    for r in rows:
        out.append({"concept_id":int(r["concept_id"]),"canonical":r["canonical"],
                    "file_id":int(r["file_id"] or 0),"embedding":bytes(r["embedding"]),
                    "embedding_dim":int(r["embedding_dim"] or 0)})
    return out

def negative_file_ids(db_path, concept_id):
    """file_ids marked negative for this concept (boundary members)."""
    cid=int(concept_id or 0)
    if not db_path or cid<=0: return set()
    c=_conn(db_path)
    try:
        rows=c.execute(
            "SELECT DISTINCT file_id FROM concept_examples WHERE concept_id=? AND role='negative' AND file_id>0",
            (cid,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    return {int(r["file_id"]) for r in rows if int(r["file_id"] or 0)>0}

def example_rows_for_concept(db_path, concept_id, *, role="positive", user_only=False, customer_key=""):
    """Return example rows; when customer_key set, prefer that scope then fall back to global.

    Never returns another customer's rows. Does not alter concept canonical identity.
    """
    cid=int(concept_id or 0)
    if not db_path or cid<=0:
        return []
    ck=_norm_customer_key(customer_key)
    c=_conn(db_path)
    try:
        sql="""SELECT file_id, file_path, source, IFNULL(customer_key,'') AS customer_key,
                     embedding, embedding_dim, embedding_backend
              FROM concept_examples
              WHERE concept_id=? AND role=? AND file_id>0"""
        params=[cid, role]
        if user_only:
            sql += " AND IFNULL(source,'user') NOT IN ('auto','autonomous','candidate')"
        if ck:
            sql += " AND (IFNULL(customer_key,'')=? OR IFNULL(customer_key,'')='')"
            params.append(ck)
        else:
            # Global path: only unscoped examples (unchanged behavior)
            sql += " AND IFNULL(customer_key,'')=''"
        sql += """ ORDER BY CASE WHEN IFNULL(customer_key,'')=? THEN 0 ELSE 1 END,
                         CASE WHEN IFNULL(source,'user')='user' THEN 0 ELSE 1 END, id ASC"""
        params.append(ck)
        rows=c.execute(sql, tuple(params)).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    c.close()
    out=[]
    seen=set()
    for r in rows:
        fid=int(r["file_id"] or 0)
        if fid<=0 or fid in seen:
            continue
        seen.add(fid)
        out.append({
            "file_id": fid,
            "file_path": str(r["file_path"] or ""),
            "source": str(r["source"] or ""),
            "customer_key": str(r["customer_key"] or ""),
            "embedding": bytes(r["embedding"]) if r["embedding"] else None,
            "embedding_dim": int(r["embedding_dim"] or 0),
            "embedding_backend": str(r["embedding_backend"] or ""),
        })
    return out

def customer_example_file_ids(db_path, concept_id, *, customer_key="", user_only=True):
    """Soft prior: customer-scoped file ids only (no global fallback)."""
    ck=_norm_customer_key(customer_key)
    if not ck:
        return []
    rows=example_rows_for_concept(
        db_path, concept_id, role="positive", user_only=user_only, customer_key=ck
    )
    return [int(r["file_id"]) for r in rows if str(r.get("customer_key") or "")==ck]
