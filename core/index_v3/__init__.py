"""Index Engine V3 — public API."""

from core.index_v3.artifact_state import assess_file, assess_rows, plan_missing
from core.index_v3.engine import EngineReport, IndexEngineV3
from core.index_v3.planner import enqueue_post_ga, plan_jobs_for_file
from core.index_v3.progress import ProgressSnapshot, count_progress, session_delta
from core.index_v3.queues import JobStore
from core.index_v3.types import (
    AI_FINAL_REQUIRED,
    ARTIFACT_DEPENDENCIES,
    HEAVY_ARTIFACTS,
    LIGHT_ARTIFACTS,
    POST_GA_ARTIFACTS,
    POST_GA_MODES,
    Artifact,
    ArtifactStatus,
    Job,
    Mode,
    QueueKind,
)
from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.safe_decode import is_risky_path, run_with_timeout
from core.index_v3.worker import ArtifactProcessor, Worker

__all__ = [
    "AI_FINAL_REQUIRED",
    "ARTIFACT_DEPENDENCIES",
    "HEAVY_ARTIFACTS",
    "LIGHT_ARTIFACTS",
    "POST_GA_ARTIFACTS",
    "POST_GA_MODES",
    "Artifact",
    "ArtifactProcessor",
    "ArtifactStatus",
    "EngineReport",
    "IndexEngineV3",
    "Job",
    "JobStore",
    "Mode",
    "ProgressSnapshot",
    "QueueKind",
    "Worker",
    "assess_file",
    "assess_rows",
    "count_progress",
    "enqueue_post_ga",
    "plan_jobs_for_file",
    "plan_missing",
    "session_delta",
    "RealArtifactProcessor",
    "is_risky_path",
    "run_with_timeout",
]