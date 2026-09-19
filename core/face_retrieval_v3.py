"""
Face Retrieval V3
- Separate face ANN-style retrieval from Pattern/FAISS retrieval.
- Multi-prototype identity scoring.
- Does not hard-filter gender unless an explicit gender filter is requested.
- Keeps identity score separate from visual/pattern score.
- Supports known-person/celebrity reference gallery.
- Designed to be optional: failures must not break Pattern Search.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import math
import sqlite3
import json
import os

@dataclass
class FaceCandidate:
    person_id: str
    face_id: Optional[str]
    similarity: float
    identity_score: float
    category: str
    gender: str = "UNKNOWN"
    name: Optional[str] = None
    quality: float = 1.0
    metadata: dict = field(default_factory=dict)

def _norm(v):
    try:
        x=float(v)
    except Exception:
        return None
    return max(0.0,min(1.0,x))

def cosine(a,b):
    if not a or not b or len(a)!=len(b):
        return 0.0
    aa=sum(x*x for x in a); bb=sum(x*x for x in b)
    if aa<=0 or bb<=0:
        return 0.0
    return max(-1.0,min(1.0,sum(x*y for x,y in zip(a,b))/(aa**0.5*bb**0.5)))

class FaceRetrievalV3:
    """
    Generic retrieval layer. It can consume rows from the existing face DB
    without requiring changes to Pattern DB/FAISS.
    """
    def __init__(self, db_path: Optional[str]=None, top_k:int=200):
        self.db_path=db_path
        self.top_k=max(20,int(top_k))

    def _rows(self):
        if not self.db_path or not os.path.exists(self.db_path):
            return []
        con=sqlite3.connect(self.db_path)
        try:
            # tolerate different schemas by selecting the common fields if present
            cols={r[1] for r in con.execute("PRAGMA table_info(face_embeddings)").fetchall()}
            if not cols:
                return []
            wanted=["person_id","face_id","embedding","gender","name","quality","is_known"]
            actual=[c for c in wanted if c in cols]
            if "embedding" not in actual:
                return []
            q="SELECT "+",".join(actual)+" FROM face_embeddings"
            for row in con.execute(q):
                d=dict(zip(actual,row))
                emb=d.get("embedding")
                if isinstance(emb,(bytes,bytearray)):
                    try: emb=json.loads(emb.decode("utf-8"))
                    except Exception: emb=None
                elif isinstance(emb,str):
                    try: emb=json.loads(emb)
                    except Exception: emb=None
                if isinstance(emb,list):
                    d["embedding"]=emb
                    yield d
        finally:
            con.close()

    @staticmethod
    def classify(similarity: float, margin: float) -> str:
        # Identity categories are deliberately stricter than generic face similarity.
        if similarity >= 0.82 and margin >= 0.035:
            return "SAME_PERSON"
        if similarity >= 0.72 and margin >= 0.02:
            return "PROBABLE_SAME_PERSON"
        if similarity >= 0.58:
            return "SIMILAR_FACE"
        return "OTHER_FACE"

    def search(self, query_embedding: list[float], gender_filter: Optional[str]=None,
               top_k: Optional[int]=None) -> list[FaceCandidate]:
        rows=list(self._rows())
        scored=[]
        for r in rows:
            g=(r.get("gender") or "UNKNOWN").upper()
            if gender_filter and gender_filter.upper() not in ("ALL","ANY"):
                if g != gender_filter.upper():
                    continue
            sim=(cosine(query_embedding,r["embedding"])+1.0)/2.0
            quality=_norm(r.get("quality")) or 1.0
            scored.append((sim*0.9+quality*0.1,sim,r))
        scored.sort(key=lambda x:x[0], reverse=True)
        top=scored[:max(self.top_k,int(top_k or self.top_k))]
        # margin is computed against the next best candidate, not a fabricated fixed score
        result=[]
        for i,(rank,sim,r) in enumerate(top):
            next_sim=top[i+1][1] if i+1<len(top) else 0.0
            margin=sim-next_sim
            cat=self.classify(sim,margin)
            pid=str(r.get("person_id") or "UNKNOWN")
            result.append(FaceCandidate(
                person_id=pid,
                face_id=str(r.get("face_id")) if r.get("face_id") is not None else None,
                similarity=sim,
                identity_score=rank,
                category=cat,
                gender=(r.get("gender") or "UNKNOWN").upper(),
                name=r.get("name"),
                quality=quality,
                metadata={"margin":margin}
            ))
        return result

    @staticmethod
    def group_persons(candidates: Iterable[FaceCandidate]) -> list[dict]:
        groups={}
        for c in candidates:
            g=groups.setdefault(c.person_id,{
                "person_id":c.person_id,
                "name":c.name,
                "gender":c.gender,
                "best_similarity":0.0,
                "best_identity_score":0.0,
                "category":"OTHER_FACE",
                "evidence_count":0,
                "face_ids":[]
            })
            groups[c.person_id]["best_similarity"]=max(groups[c.person_id]["best_similarity"],c.similarity)
            groups[c.person_id]["best_identity_score"]=max(groups[c.person_id]["best_identity_score"],c.identity_score)
            groups[c.person_id]["evidence_count"]+=1
            if c.face_id: groups[c.person_id]["face_ids"].append(c.face_id)
            order={"OTHER_FACE":0,"SIMILAR_FACE":1,"PROBABLE_SAME_PERSON":2,"SAME_PERSON":3}
            if order[c.category]>order[groups[c.person_id]["category"]]:
                groups[c.person_id]["category"]=c.category
            if not groups[c.person_id]["name"] and c.name:
                groups[c.person_id]["name"]=c.name
        return sorted(groups.values(),key=lambda x:(x["best_identity_score"],x["best_similarity"]),reverse=True)
