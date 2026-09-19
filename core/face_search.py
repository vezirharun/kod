"""Face query bridge used by SearchEngine without touching pattern ranking."""
from __future__ import annotations
import re
import os
from typing import Any
from core.face_index import FaceIndexStore
from core.face_identity import FaceIdentityEngine

_GENDER={'kadın':'FEMALE','kadin':'FEMALE','female':'FEMALE','women':'FEMALE','woman':'FEMALE',
         'erkek':'MALE','male':'MALE','men':'MALE','man':'MALE',
         'bilinmiyor':'UNKNOWN','unknown':'UNKNOWN'}


def parse_face_query(text:str):
    q=' '.join((text or '').strip().lower().split())
    if q in _GENDER:
        return {'kind':'gender','value':_GENDER[q]}
    m=re.fullmatch(r'(?:kişi|kisi|person)[ _-]?(\d{1,6})',q)
    if m:
        return {'kind':'person','value':f"person_{int(m.group(1)):04d}"}
    m=re.fullmatch(r'(?:kişi|kisi|person)[: ]+(.+)',q)
    if m:
        return {'kind':'name','value':m.group(1).strip()}
    m=re.fullmatch(r'person_(\d{1,6})',q)
    if m:
        return {'kind':'person','value':f"person_{int(m.group(1)):04d}"}
    return None


class FaceSearch:
    def __init__(self, db_path:str, *, threshold: float = 0.56, min_margin: float = 0.035):
        self.engine=FaceIdentityEngine(threshold=threshold, min_margin=min_margin)
        self.store=FaceIndexStore(
            db_path, threshold=self.engine.effective_threshold,
            min_margin=min_margin, identity_engine=self.engine.backend,
        )

    def file_ids(self,text:str,limit:int=400):
        q = ' '.join((text or '').strip().split())
        if not q:
            return []

        # First consult the persistent person-name/filename alias index. This
        # makes a bare query such as "Hülya Koçyiğit" work after the user has
        # labelled a cluster or when the source filename carries that name.
        try:
            named = self.store.search_by_name(q, limit=limit)
            if named:
                return named
        except Exception:
            pass

        spec=parse_face_query(q)
        if not spec:
            return []
        if spec['kind']=='gender':
            return self.store.search(gender=spec['value'],limit=limit)
        if spec['kind']=='name':
            return self.store.search_by_name(spec['value'],limit=limit)
        return self.store.search(person_id=spec['value'],limit=limit)

    def image_matches(self, image_path: str, *, threshold: float | None = None, limit: int = 400) -> dict[int, dict[str, Any]]:
        """Find every indexed file containing the same persistent person as a query image."""
        if not image_path or not os.path.isfile(image_path):
            return {}
        try:
            import cv2
            image = cv2.imread(image_path)
        except Exception:
            return {}
        if image is None:
            return {}
        observations = self.engine.analyze_image(image)
        embeddings = [o.embedding for o in observations if o.embedding is not None]
        if not embeddings:
            return {}
        return self.store.search_by_embeddings(embeddings, threshold=threshold, limit=limit)
