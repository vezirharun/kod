"""Incremental Face Index scanner; never writes to the pattern index."""
from __future__ import annotations
import os
from pathlib import Path
try:
    import cv2
except Exception:
    cv2 = None
from core.db import Database
from core.face_identity import FaceIdentityEngine
from core.face_index import FaceIndexStore
from core.thumbnailer import Thumbnailer

class FaceIndexScanner:
    def __init__(self, pattern_db_path:str, face_db_path:str, *, threshold:float=0.62, min_margin:float=0.05,
                 model_name:str="buffalo_l", providers:list|None=None, det_size:int=640,
                 allow_vision_fallback:bool=False):
        self.pattern_db=Database(pattern_db_path, read_only=True)
        self.engine=FaceIdentityEngine(
            threshold=threshold, min_margin=min_margin, model_name=model_name,
            providers=providers, det_size=det_size,
            allow_vision_fallback=allow_vision_fallback,
        )
        pattern_abs = os.path.abspath(str(pattern_db_path or ""))
        face_abs = os.path.abspath(str(face_db_path or "")) if face_db_path else ""
        # Never point the secondary face index at the authoritative pattern DB.
        # A stale/old settings file can contain an empty or legacy face_db_path;
        # silently using patterns.db then produces "no such table: face_files"
        # and can kill the background face thread.
        if not face_abs or face_abs == pattern_abs:
            face_db_path = str(Path(pattern_abs).with_name("face_index.db"))
        Path(str(face_db_path)).parent.mkdir(parents=True, exist_ok=True)
        self.store=FaceIndexStore(
            face_db_path, threshold=self.engine.effective_threshold,
            min_margin=min_margin, identity_engine=self.engine.backend,
        )

    def reconcile(self) -> int:
        with self.pattern_db.connect() as con:
            rows=con.execute("SELECT id FROM files WHERE status!='missing'").fetchall()
        return self.store.reconcile_missing({int(r[0]) for r in rows})

    def scan_incremental(self, *, batch_size: int = 200) -> dict:
        """Scan only files that are missing/changed in the face index.

        Unlike the old after-id scanner, this also catches files modified after
        their first scan. It is safe to call repeatedly from a background loop.
        """
        if cv2 is None or not self.engine.available():
            return {"scanned": 0, "skipped": 0, "errors": 0, "faces": 0, "dependency_missing": True, "stats": self.store.stats(), "backend": self.engine.backend}
        scanned = skipped = errors = faces = 0
        # IMPORTANT: face_files lives in the dedicated face DB, not in the
        # authoritative pattern DB.  The old implementation joined face_files
        # against pattern_db, which caused:
        #   sqlite3.OperationalError: no such table: face_files
        # and silently killed the always-on face index loop.
        # Read the authoritative file candidates first, then compare their
        # current state against the isolated face index.
        with self.pattern_db.connect() as con:
            candidates = con.execute(
                """SELECT f.id,f.path,f.mtime,f.file_size,
                          COALESCE(f.feature_preview_path, '') AS feature_preview_path,
                          COALESCE(f.thumbnail_path, '') AS thumbnail_path
                   FROM files f
                   WHERE f.status!='missing'
                     AND (lower(f.path) LIKE '%.jpg' OR lower(f.path) LIKE '%.jpeg'
                       OR lower(f.path) LIKE '%.png' OR lower(f.path) LIKE '%.webp'
                       OR lower(f.path) LIKE '%.bmp' OR lower(f.path) LIKE '%.tif'
                       OR lower(f.path) LIKE '%.tiff')
                   ORDER BY f.id LIMIT ?""",
                (max(int(batch_size) * 4, int(batch_size)),),
            ).fetchall()

        # One connection for the whole comparison; do not open one SQLite
        # connection per file while the NAS/index is busy.
        with self.store._connect() as face_con:
            indexed = {
                int(r["file_id"]): (
                    str(r["path"] or ""),
                    float(r["mtime"] or 0),
                    int(r["file_size"] or 0),
                    str(r["status"] or ""),
                )
                for r in face_con.execute(
                    "SELECT file_id,path,mtime,file_size,status FROM face_files WHERE file_id IN (%s)"
                    % (",".join("?" for _ in candidates) if candidates else "0"),
                    [int(r["id"]) for r in candidates],
                ).fetchall()
            } if candidates else {}

        rows = []
        for r in candidates:
            fid = int(r["id"])
            old = indexed.get(fid)
            current = (str(r["path"] or ""), float(r["mtime"] or 0), int(r["file_size"] or 0))
            if old is None or old[3] != "done" or old[:3] != current:
                rows.append(r)
                if len(rows) >= int(batch_size):
                    break
        for r in rows:
            fid = int(r["id"]); path = str(r["path"]); mtime = float(r["mtime"] or 0); size = int(r["file_size"] or 0)
            feature_preview = str(r["feature_preview_path"] or "").strip()
            thumbnail = str(r["thumbnail_path"] or "").strip()
            # Face indexing must consume the same cached visual artifacts as the
            # rest of the pattern pipeline. Never reopen the source file here:
            # source paths may live on NAS/Unicode folders and are owned by the
            # preview pipeline. Prefer the high-quality feature/medium preview;
            # use the thumbnail only as a safe fallback. If neither exists yet,
            # leave the face record untouched so the next background pass retries
            # after the light pipeline has produced the preview.
            preview_path = feature_preview or thumbnail
            if not preview_path:
                skipped += 1
                continue
            try:
                rgb = Thumbnailer.load_image(preview_path)
                if rgb is None or getattr(rgb, "size", 0) == 0:
                    skipped += 1
                    continue
                # Thumbnailer/Pillow returns RGB; the face engines consume BGR.
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                obs = self.engine.analyze_image(image)
                assigned = self.store.replace_file_faces(fid, path, mtime, size, obs)
                faces += len(assigned); scanned += 1
            except Exception as exc:
                self.store.mark_error(fid, path, mtime, size, str(exc))
                errors += 1
        return {"scanned": scanned, "skipped": skipped, "errors": errors, "faces": faces, "stats": self.store.stats()}

    def scan(self, *, batch_size:int=100, after_id:int=0, limit:int=0):
        if cv2 is None or not self.engine.available():
            return {"scanned":0,"skipped":0,"errors":0,"faces":0,"dependency_missing":True,"stats":self.store.stats(),"last_id":after_id,"backend":self.engine.backend}
        scanned=skipped=errors=faces=0
        with self.pattern_db.connect() as con:
            rows=con.execute("SELECT id,path,mtime,file_size,status FROM files WHERE id>? AND status!='missing' ORDER BY id LIMIT ?", (int(after_id), int(limit) if limit else 1000000000)).fetchall()
        for r in rows:
            fid=int(r['id']); path=str(r['path']); mtime=float(r['mtime'] or 0); size=int(r['file_size'] or 0)
            if self.store.is_current(fid,path,mtime,size):
                skipped+=1; continue
            try:
                # Legacy scan() is retained for compatibility, but it must obey
                # the same preview-only contract as scan_incremental().
                with self.pattern_db.connect() as preview_con:
                    pr = preview_con.execute(
                        "SELECT COALESCE(feature_preview_path,'') AS feature_preview_path, "
                        "COALESCE(thumbnail_path,'') AS thumbnail_path FROM files WHERE id=?",
                        (fid,),
                    ).fetchone()
                preview_path = str((pr["feature_preview_path"] if pr else "") or "").strip()
                if not preview_path:
                    preview_path = str((pr["thumbnail_path"] if pr else "") or "").strip()
                if not preview_path:
                    skipped+=1; continue
                rgb=Thumbnailer.load_image(preview_path)
                if rgb is None or getattr(rgb, "size", 0)==0:
                    skipped+=1; continue
                image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                obs=self.engine.analyze_image(image)
                assigned=self.store.replace_file_faces(fid,path,mtime,size,obs)
                faces+=len(assigned); scanned+=1
            except Exception as exc:
                self.store.mark_error(fid,path,mtime,size,str(exc))
                errors+=1
        return {'scanned':scanned,'skipped':skipped,'errors':errors,'faces':faces,'stats':self.store.stats(),'last_id':int(rows[-1]['id']) if rows else after_id}
