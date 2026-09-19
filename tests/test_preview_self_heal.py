"""Preview self-heal — critical black/white rules + repair guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.preview_self_heal.repair import (
    begin_repair,
    clear_repair_state_for_tests,
    end_repair,
    repair_preview_artifact,
)
from core.preview_self_heal.validate import (
    Verdict,
    compare_source_vs_preview,
    validate_artifact_light,
    validate_preview,
)


def _write_solid(path: Path, rgb: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Path:
    from PIL import Image

    Image.new("RGB", size, rgb).save(path)
    return path


def _write_colorful(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    from PIL import Image
    import numpy as np

    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256)
    Image.fromarray(arr, "RGB").save(path)
    return path


@pytest.fixture(autouse=True)
def _clear_repair():
    clear_repair_state_for_tests()
    yield
    clear_repair_state_for_tests()


def test_white_source_white_preview_valid(tmp_path: Path):
    src = _write_solid(tmp_path / "src_w.png", (255, 255, 255))
    prev = _write_solid(tmp_path / "prev_w.webp", (255, 255, 255))
    light = validate_artifact_light(prev)
    assert light.verdict == Verdict.SUSPICIOUS
    full = compare_source_vs_preview(src, prev)
    assert full.verdict == Verdict.VALID
    assert "source_matches_mono" in full.reason


def test_black_source_black_preview_valid(tmp_path: Path):
    src = _write_solid(tmp_path / "src_b.png", (0, 0, 0))
    prev = _write_solid(tmp_path / "prev_b.webp", (0, 0, 0))
    full = validate_preview(prev, src, allow_source_compare=True)
    assert full.verdict == Verdict.VALID


def test_colorful_source_white_preview_invalid(tmp_path: Path):
    src = _write_colorful(tmp_path / "src_c.png")
    prev = _write_solid(tmp_path / "prev_w.webp", (255, 255, 255))
    full = validate_preview(prev, src, allow_source_compare=True)
    assert full.verdict == Verdict.INVALID
    assert "misrepresenting" in full.reason


def test_colorful_source_black_preview_invalid(tmp_path: Path):
    src = _write_colorful(tmp_path / "src_c2.png")
    prev = _write_solid(tmp_path / "prev_b.webp", (0, 0, 0))
    full = validate_preview(prev, src, allow_source_compare=True)
    assert full.verdict == Verdict.INVALID


def test_mono_alone_without_compare_is_suspicious_not_invalid(tmp_path: Path):
    prev = _write_solid(tmp_path / "only.webp", (255, 255, 255))
    r = validate_preview(prev, allow_source_compare=False)
    assert r.verdict == Verdict.SUSPICIOUS
    assert r.needs_repair is False


def test_missing_and_corrupt_invalid(tmp_path: Path):
    missing = validate_artifact_light(tmp_path / "nope.webp")
    assert missing.verdict == Verdict.INVALID
    assert missing.reason == "missing"

    bad = tmp_path / "bad.webp"
    bad.write_bytes(b"not-an-image" + b"\x00" * 128)
    corrupt = validate_artifact_light(bad)
    assert corrupt.verdict == Verdict.INVALID
    assert corrupt.reason in ("too_small",) or corrupt.reason.startswith("decode:")


def test_colorful_preview_valid_without_source(tmp_path: Path):
    prev = _write_colorful(tmp_path / "ok.webp")
    r = validate_artifact_light(prev)
    assert r.verdict == Verdict.VALID
    assert r.reason == "ok"


def test_repair_dedup_and_retry_limit(tmp_path: Path):
    src = str(tmp_path / "x.png")
    ok, n0 = begin_repair(src)
    assert ok and n0 == 0
    ok2, _ = begin_repair(src)
    assert ok2 is False  # inflight dup
    end_repair(src, success=False, final_fail=False)

    class Fake:
        success = False
        preview_path = ""
        error = "boom"

    # burn retries
    for _ in range(3):
        clear_repair_state_for_tests()  # only clear inflight between — use real counters
    clear_repair_state_for_tests()
    outcomes = []
    for _ in range(4):
        outcomes.append(
            repair_preview_artifact(
                src,
                artifact_path="",
                create_fn=lambda: Fake(),
                reason="test",
            )
        )
    assert any(o.action == "recreate_failed" for o in outcomes)
    # After max, further calls skip
    last = repair_preview_artifact(src, create_fn=lambda: Fake(), reason="test")
    assert last.action in ("skip_dup_or_limit", "recreate_failed")


def test_repair_refuses_non_cache_unlink(tmp_path: Path):
    src = _write_colorful(tmp_path / "src.png")
    # Path outside cache segments — unlink must refuse; recreate still runs
    outside = tmp_path / "outside.webp"
    _write_solid(outside, (255, 255, 255))

    class FakeOk:
        success = True
        preview_path = str(tmp_path / "feature_previews" / "new_fp.webp")
        error = ""

    (tmp_path / "feature_previews").mkdir()
    _write_colorful(Path(FakeOk.preview_path))

    out = repair_preview_artifact(
        str(src),
        artifact_path=str(outside),
        create_fn=lambda: FakeOk(),
        compare_path=str(src),
    )
    assert outside.is_file()  # never deleted source-adjacent non-cache
    assert out.ok is True


def test_gate_new_preview_uses_local_compare(tmp_path: Path):
    from core.preview_self_heal.hooks import gate_new_preview

    src = _write_colorful(tmp_path / "src.png")
    local = _write_colorful(tmp_path / "local_render.png")
    bad = _write_solid(tmp_path / "prev.webp", (255, 255, 255))
    # compare against colorful local render → INVALID misrepresenting
    r = gate_new_preview(str(bad), str(src), compare_path=str(local))
    assert r.verdict == Verdict.INVALID
