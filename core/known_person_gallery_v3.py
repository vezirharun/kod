"""
Known Person / Celebrity Gallery V3.
A conservative name resolver: it only returns a name when a known-person
reference embedding is sufficiently strong and has a margin over the runner-up.
"""
from __future__ import annotations
from .face_retrieval_v3 import cosine

class KnownPersonGallery:
    def __init__(self, references=None, threshold=0.80, margin=0.035):
        self.references=references or []
        self.threshold=threshold
        self.margin=margin

    def add(self, name, embedding, person_id=None, metadata=None):
        self.references.append({
            "name":name,"embedding":embedding,
            "person_id":person_id or name,
            "metadata":metadata or {}
        })

    def identify(self, query_embedding):
        scored=[]
        for r in self.references:
            s=(cosine(query_embedding,r["embedding"])+1)/2
            scored.append((s,r))
        scored.sort(key=lambda x:x[0], reverse=True)
        if not scored:
            return {"name":None,"status":"UNKNOWN","score":0.0,"margin":0.0}
        best,r=scored[0]
        second=scored[1][0] if len(scored)>1 else 0.0
        margin=best-second
        if best>=self.threshold and margin>=self.margin:
            return {"name":r["name"],"status":"PROBABLE_IDENTITY","score":best,"margin":margin,
                    "person_id":r.get("person_id"),"metadata":r.get("metadata",{})}
        return {"name":None,"status":"UNKNOWN","score":best,"margin":margin,
                "candidate":r.get("name")}
