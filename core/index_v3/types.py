"""Index Engine V3 — artifact-first types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Mode(str, Enum):
    FAST = "fast"
    GENERAL_AI = "general_ai"
    COMPLETE = "complete"
    REPAIR = "repair"
    PATCH = "patch"
    OCR = "ocr"
    POST_GA = "post_ga"


class Artifact(str, Enum):
    THUMBNAIL = "thumbnail"
    PREVIEW = "preview"
    HASH = "hash"
    METADATA = "metadata"
    DINO = "dino"
    CLIP = "clip"
    TEXTURE = "texture"
    SEMANTIC = "semantic"
    DNA = "dna"
    OBJECT_CONCEPT = "object_concept"
    OWLV2 = "owlv2"
    PATCH = "patch"
    OCR = "ocr"


class ArtifactStatus(str, Enum):
    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"


class QueueKind(str, Enum):
    LIGHT = "light"
    PREVIEW = "preview"
    HEAVY = "heavy"
    REPAIR = "repair"


class JobState(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    FAILED_PERMANENT = "failed_permanent"


# Hızlı İndeks yalnızca Preview havuzunu oluşturur ve Thumbnail'ı
# hazır Preview'dan üretir. Hash/Metadata Genel AI lane'ine aittir.
LIGHT_ARTIFACTS: tuple[Artifact, ...] = (
    Artifact.THUMBNAIL,
    Artifact.PREVIEW,
)

# Genel AI zorunlu hat (PATCH/OCR/Object yok): HASH → METADATA → DINO → CLIP →
# TEXTURE → SEMANTIC → DNA. Bu hat AI_FINAL=TRUE yapar. Object Index ve OWLv2
# bu hatta girmez (ayrı alt aşama; AI_FINAL yüzdesini düşürmez).
HEAVY_ARTIFACTS: tuple[Artifact, ...] = (
    Artifact.HASH,
    Artifact.METADATA,
    Artifact.DINO,
    Artifact.CLIP,
    Artifact.TEXTURE,
    Artifact.SEMANTIC,
    Artifact.DNA,
)

OBJECT_CONCEPT_ARTIFACTS: tuple[Artifact, ...] = (Artifact.OBJECT_CONCEPT,)
OWLV2_ARTIFACTS: tuple[Artifact, ...] = (Artifact.OWLV2,)
# Object Index + OWLv2: HASH…DNA hattının dışında; AI_FINAL'e girmez.
GA_SIDE_ARTIFACTS: tuple[Artifact, ...] = OBJECT_CONCEPT_ARTIFACTS + OWLV2_ARTIFACTS

# PATCH ve OCR, AI_FINAL sonrası bağımsız işler; birbirini ve GA'yı bloke etmez.
POST_GA_ARTIFACTS: tuple[Artifact, ...] = (
    Artifact.PATCH,
    Artifact.OCR,
)

POST_GA_MODES: frozenset[Mode] = frozenset(
    {Mode.PATCH, Mode.OCR, Mode.POST_GA}
)

OCR_ARTIFACT: Artifact = Artifact.OCR

# Genel AI tamam = HASH…DNA READY. PATCH AI_FINAL'e dahil değil.
AI_FINAL_REQUIRED: tuple[Artifact, ...] = HEAVY_ARTIFACTS

# Artifact-level predecessors (preview gate is separate). Empty = preview-only.
# PATCH DINO model kilidini paylaşır (scheduler lane). Otomatik enqueue
# AI_FINAL sonrası; manuel job READY bağımlılığı HASH…DNA beklemez.
# OCR/PATCH asla HASH…DNA seri hattına girmez.
ARTIFACT_DEPENDENCIES: dict[Artifact, tuple[Artifact, ...]] = {
    Artifact.DINO: (),
    Artifact.CLIP: (),
    Artifact.TEXTURE: (),
    Artifact.OCR: (),
    Artifact.PATCH: (),
    Artifact.SEMANTIC: (Artifact.TEXTURE,),
    Artifact.DNA: (Artifact.SEMANTIC,),
    Artifact.OBJECT_CONCEPT: (),
    Artifact.OWLV2: (),
}


@dataclass(frozen=True)
class Job:
    file_id: int
    artifact: Artifact
    queue: QueueKind
    source_id: int = 0
    path: str = ""
    owl_priority: int = 0

    @property
    def key(self) -> str:
        return f"{self.file_id}:{self.artifact.value}"


@dataclass
class FileArtifactReport:
    file_id: int
    source_id: int
    path: str
    status: dict[Artifact, ArtifactStatus] = field(default_factory=dict)

    def get(self, art: Artifact) -> ArtifactStatus:
        return self.status.get(art, ArtifactStatus.MISSING)

    def ready(self, art: Artifact) -> bool:
        return self.get(art) == ArtifactStatus.READY

    def missing(self, arts: Iterable[Artifact]) -> list[Artifact]:
        return [a for a in arts if self.get(a) != ArtifactStatus.READY]

    @property
    def light_complete(self) -> bool:
        return all(self.ready(a) for a in LIGHT_ARTIFACTS)

    @property
    def ai_final(self) -> bool:
        return all(self.ready(a) for a in AI_FINAL_REQUIRED)

    @property
    def preview_ready(self) -> bool:
        """AI giriş kapısı yalnızca gerçek Preview'dır.

        Thumbnail kullanıcı arayüzü için ayrı bir light artifact'tir;
        eksik olması DINO/CLIP/Texture/Semantic/DNA işlemlerini
        bloke etmez. PATCH/OCR AI_FINAL sonrası ayrıdır.
        """
        return self.ready(Artifact.PREVIEW)
