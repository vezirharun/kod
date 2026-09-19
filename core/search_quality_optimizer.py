"""Search Quality Optimizer — geri bildirimlerden öğrenerek skor ağırlıklarını ayarla.

Kullanım:
    from core.search_quality_optimizer import SearchQualityOptimizer
    opt = SearchQualityOptimizer(db, settings)
    opt.run()                  # analiz + ağırlık güncelleme
    report = opt.report()      # son durum
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Skor bileşen ağırlıkları (settings'ten okunur, buraya fallback) ──────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "filename_score":     0.30,
    "ocr_score":          0.25,
    "brand_alias_score":  0.20,
    "family_score":       0.15,
    "texture_score":      0.10,
    "nl_score":           0.08,
    "auto_tag_score":     0.08,
    "group_score":        0.06,
    "feedback_score":     0.05,
    "text_score":         0.04,
    "dna_score":          0.20,
    "phash":              0.25,
    "faiss_score":        0.20,
    "patch_sim":          0.15,
    "color_score":        0.10,
    "palette_similarity": 0.10,
}

# Hangi bileşenleri optimize edebiliriz (çok düşük/yüksek ayarı engelle)
_CLAMP: dict[str, tuple[float, float]] = {
    k: (0.02, 0.60) for k in DEFAULT_WEIGHTS
}


@dataclass
class OptimizationReport:
    timestamp: str
    total_feedback: int
    positive: int
    negative: int
    analyzed_results: int
    old_weights: dict[str, float]
    new_weights: dict[str, float]
    changed: list[str]
    notes: list[str] = field(default_factory=list)


class SearchQualityOptimizer:
    """Gerçek kullanıcı geri bildirimlerini analiz ederek ağırlıkları günceller."""

    def __init__(self, db: Any, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self._weights_path = Path(
            getattr(settings, "data_dir", "data")
        ) / "learned_weights.json"

    # ── Ağırlık yükleme / kaydetme ────────────────────────────────────────────

    def load_weights(self) -> dict[str, float]:
        if self._weights_path.exists():
            try:
                data = json.loads(self._weights_path.read_text(encoding="utf-8"))
                return {k: float(v) for k, v in data.items() if k in DEFAULT_WEIGHTS}
            except Exception:
                pass
        return dict(DEFAULT_WEIGHTS)

    def save_weights(self, weights: dict[str, float]) -> None:
        self._weights_path.parent.mkdir(parents=True, exist_ok=True)
        self._weights_path.write_text(
            json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Analiz ────────────────────────────────────────────────────────────────

    def _fetch_feedback(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Son N geri bildirimi döndür."""
        try:
            with self.db.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT uf.file_id, uf.label, uf.query_path,
                           uf.created_at,
                           f.pattern_family
                    FROM user_feedback uf
                    LEFT JOIN files f ON f.id = uf.file_id
                    ORDER BY uf.id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Geri bildirim alınamadı: %s", exc)
            return []

    def _fetch_breakdown_for(self, file_ids: list[int]) -> dict[int, dict[str, float]]:
        """Dosya ID'leri için kaydedilmiş breakdown (varsa debug_json'dan)."""
        if not file_ids:
            return {}
        result: dict[int, dict[str, float]] = {}
        try:
            ph = ",".join("?" * len(file_ids))
            with self.db.connect() as conn:
                rows = conn.execute(
                    f"SELECT file_id, debug_json FROM search_debug_log WHERE file_id IN ({ph})",
                    file_ids,
                ).fetchall()
            for row in rows:
                try:
                    d = json.loads(row["debug_json"] or "{}")
                    result[int(row["file_id"])] = d.get("breakdown", {})
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def analyze(self, feedback: list[dict]) -> dict[str, dict[str, float]]:
        """Her bileşen için ortalama skor (positif vs negatif) hesapla."""
        pos_sums: dict[str, float] = {k: 0.0 for k in DEFAULT_WEIGHTS}
        neg_sums: dict[str, float] = {k: 0.0 for k in DEFAULT_WEIGHTS}
        pos_cnt:  dict[str, int]   = {k: 0 for k in DEFAULT_WEIGHTS}
        neg_cnt:  dict[str, int]   = {k: 0 for k in DEFAULT_WEIGHTS}

        ids = [int(f["file_id"]) for f in feedback if f.get("file_id")]
        breakdowns = self._fetch_breakdown_for(ids)

        for fb in feedback:
            fid = int(fb.get("file_id") or 0)
            label = str(fb.get("label", "")).lower()
            is_positive = label in ("correct", "positive", "good", "1", "true")
            bd = breakdowns.get(fid, {})
            for key in DEFAULT_WEIGHTS:
                val = float(bd.get(key, 0) or 0)
                if is_positive:
                    pos_sums[key] += val
                    pos_cnt[key] += 1
                else:
                    neg_sums[key] += val
                    neg_cnt[key] += 1

        result: dict[str, dict[str, float]] = {}
        for key in DEFAULT_WEIGHTS:
            pos_avg = pos_sums[key] / max(pos_cnt[key], 1)
            neg_avg = neg_sums[key] / max(neg_cnt[key], 1)
            result[key] = {
                "pos_avg": round(pos_avg, 4),
                "neg_avg": round(neg_avg, 4),
                "pos_count": pos_cnt[key],
                "neg_count": neg_cnt[key],
                "discriminative": round(pos_avg - neg_avg, 4),
            }
        return result

    def compute_new_weights(
        self,
        old: dict[str, float],
        analysis: dict[str, dict[str, float]],
        *,
        learning_rate: float = 0.12,
    ) -> dict[str, float]:
        """Analiz sonuçlarına göre ağırlıkları güncelle (küçük adım, aşırı değişim yok)."""
        new = dict(old)
        for key in DEFAULT_WEIGHTS:
            disc = analysis.get(key, {}).get("discriminative", 0.0)
            pos_n = analysis.get(key, {}).get("pos_count", 0)
            if pos_n < 5:
                # Yeterli veri yok — dokunma
                continue
            # Discriminative > 0 → yükselt; < 0 → düşür
            delta = disc * learning_rate
            lo, hi = _CLAMP.get(key, (0.02, 0.60))
            new[key] = max(lo, min(hi, old.get(key, DEFAULT_WEIGHTS[key]) + delta))
        return {k: round(v, 4) for k, v in new.items()}

    # ── Ana çalıştırma ────────────────────────────────────────────────────────

    def run(self, *, dry_run: bool = False) -> OptimizationReport:
        t0 = time.perf_counter()
        feedback = self._fetch_feedback()
        positive = sum(1 for f in feedback if str(f.get("label","")).lower() in ("correct","positive","good","1","true"))
        negative = len(feedback) - positive

        notes: list[str] = []
        old_w = self.load_weights()

        if len(feedback) < 10:
            notes.append("Yeterli geri bildirim yok (min 10); ağırlıklar değiştirilmedi.")
            return OptimizationReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_feedback=len(feedback), positive=positive, negative=negative,
                analyzed_results=0, old_weights=old_w, new_weights=old_w,
                changed=[], notes=notes,
            )

        analysis  = self.analyze(feedback)
        new_w     = self.compute_new_weights(old_w, analysis)
        changed   = [k for k in new_w if abs(new_w[k] - old_w.get(k, DEFAULT_WEIGHTS[k])) > 1e-4]

        if not dry_run and changed:
            self.save_weights(new_w)
            logger.info(
                "Ağırlıklar güncellendi (%d değişiklik) — %.0fms",
                len(changed), (time.perf_counter() - t0) * 1000,
            )

        return OptimizationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_feedback=len(feedback), positive=positive, negative=negative,
            analyzed_results=len(feedback), old_weights=old_w, new_weights=new_w,
            changed=changed, notes=notes,
        )

    def report(self) -> dict[str, Any]:
        r = self.run(dry_run=True)
        return {
            "timestamp":        r.timestamp,
            "total_feedback":   r.total_feedback,
            "positive":         r.positive,
            "negative":         r.negative,
            "changed":          r.changed,
            "notes":            r.notes,
            "weights":          r.new_weights,
        }
