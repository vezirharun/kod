"""Index Engine V3 — fair multi-source scheduler (no Light starvation of Heavy)."""

from __future__ import annotations

from dataclasses import dataclass

from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind


@dataclass
class SourceBacklog:
    source_id: int
    light: int = 0
    preview: int = 0
    heavy: int = 0
    repair: int = 0


def measure_backlogs(store: JobStore, source_ids: list[int]) -> list[SourceBacklog]:
    """Yalnız verilen source_ids — SQL GROUP BY (tüm pending satırlarını çekmez)."""
    allowed = {int(s) for s in source_ids if int(s) > 0}
    by_src: dict[int, SourceBacklog] = {
        sid: SourceBacklog(source_id=sid) for sid in allowed
    }
    if not allowed:
        return []
    for sid, queue, n in store.count_pending_grouped(sorted(allowed)):
        if sid not in by_src:
            continue
        b = by_src[sid]
        if queue == QueueKind.LIGHT.value:
            b.light = n
        elif queue == QueueKind.PREVIEW.value:
            b.preview = n
        elif queue == QueueKind.HEAVY.value:
            b.heavy = n
        else:
            b.repair = n
    return [by_src[sid] for sid in sorted(allowed)]


def fair_claim_order(
    backlogs: list[SourceBacklog],
    queue: QueueKind,
) -> list[int]:
    """Round-robin style: sources with work on this queue, sorted by least recently favored.

    Critical rule: a source with only LIGHT work must not block another source's HEAVY
    when claiming HEAVY (and vice versa). Claim order is per-queue.
    """
    scored: list[tuple[int, int, int]] = []
    for b in backlogs:
        if queue == QueueKind.LIGHT:
            n = b.light
        elif queue == QueueKind.PREVIEW:
            n = b.preview
        elif queue == QueueKind.HEAVY:
            n = b.heavy
        else:
            n = b.repair
        if n <= 0:
            continue
        # Prefer smaller queues slightly for fairness (anti-starvation of small sources)
        scored.append((n, b.source_id, b.source_id))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [sid for _, sid, _ in scored]


def claim_fair(
    store: JobStore,
    queue: QueueKind,
    worker: str,
    source_ids: list[int],
    *,
    limit: int = 1,
    artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
) -> list[Job]:
    """Claim up to limit jobs, rotating across *scoped* sources only.

    Yetim/yabancı source job'ları ASLA claim edilmez (GENERAL_AI takılması).
    artifacts: if set, only those heavy stages (stage worker).

    Two-pass (HEAVY/REPAIR): önce dep_wait olmayan işler. Böylece küçük bir
    source'taki PATCH/OCR dep_wait döngüsü, büyük source'taki HASH…DNA
    kuyruğunu kalıcı olarak aç bırakmaz (fair_claim_order küçük backlog'u
    öne aldığı için).
    """
    allowed = [int(s) for s in source_ids if int(s) > 0]
    if not allowed:
        return []
    backlogs = measure_backlogs(store, allowed)
    order = fair_claim_order(backlogs, queue)
    if not order:
        return []
    skip_passes: tuple[bool, ...] = (
        (True, False)
        if queue in (QueueKind.HEAVY, QueueKind.REPAIR)
        else (False,)
    )
    claimed: list[Job] = []
    for skip_dep in skip_passes:
        for sid in order:
            if len(claimed) >= limit:
                return claimed
            need = limit - len(claimed)
            batch = store.claim(
                queue,
                worker,
                source_id=sid,
                limit=need,
                artifacts=artifacts,
                skip_dep_wait=skip_dep,
            )
            claimed.extend(batch)
        if claimed:
            return claimed
    return claimed


def heavy_artifact_lanes(
    worker_count: int,
) -> list[tuple[str, tuple[Artifact, ...] | None]]:
    """Map settings.worker_count to heavy stage workers.

    DINO+Patch share one lane (same DINO model). CLIP / Texture / OCR+CPU
    get their own lanes as count allows. worker_count<=1 keeps a single
    mixed heavy worker (tests / serial).
    """
    n = max(1, min(int(worker_count or 1), 4))
    dino = ("v3-heavy-dino", (Artifact.DINO, Artifact.PATCH))
    clip = ("v3-heavy-clip", (Artifact.CLIP,))
    tex = ("v3-heavy-texture", (Artifact.TEXTURE,))
    cpu = (
        "v3-heavy-cpu",
        (Artifact.OCR, Artifact.SEMANTIC, Artifact.DNA),
    )
    if n <= 1:
        return [("v3-heavy", None)]
    if n == 2:
        return [
            dino,
            (
                "v3-heavy-cpu",
                (
                    Artifact.CLIP,
                    Artifact.TEXTURE,
                    Artifact.OCR,
                    Artifact.SEMANTIC,
                    Artifact.DNA,
                ),
            ),
        ]
    if n == 3:
        return [
            dino,
            clip,
            (
                "v3-heavy-cpu",
                (Artifact.TEXTURE, Artifact.OCR, Artifact.SEMANTIC, Artifact.DNA),
            ),
        ]
    return [dino, clip, tex, cpu]
