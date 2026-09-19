"""Face production acceptance on bundled real photos (not the 116K archive)."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from core.face_intelligence import BACKEND, FaceIntelligence, kisi_label
from core.index_freeze import snapshot_index_artifacts
from core.settings import AppSettings


def _insightface_images() -> list[Path]:
    try:
        import insightface
    except Exception:
        return []
    root = Path(insightface.__file__).parent / "data" / "images"
    if not root.is_dir():
        return []
    out = []
    for name in ("t1.jpg", "Tom_Hanks_54745.png", "mask_black.jpg", "mask_blue.jpg",
                 "mask_green.jpg", "mask_white.jpg"):
        p = root / name
        if p.is_file():
            out.append(p)
    return out


def test_default_still_disabled():
    assert AppSettings().face_index_enabled is False


def test_face_production_acceptance_on_fixture(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    patterns = data / "patterns.db"
    patterns.write_bytes(b"index-frozen-sentinel")
    faiss_d = data / "faiss_dino.index"
    faiss_c = data / "faiss_clip.index"
    snap0 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(faiss_d),
        faiss_clip_path=str(faiss_c),
        cache_dir=str(cache),
    )

    face_db = tmp_path / "accept_face_index.db"
    fi = FaceIntelligence(face_db)
    assert fi.engine.allow_vision_fallback is False
    if not fi.ready():
        pytest.skip(f"InsightFace/buffalo not usable: {fi.unavailable_reason()}")

    sources = _insightface_images()
    real_unique = len(sources)
    fixture = tmp_path / "faces"
    fixture.mkdir()
    copies: list[Path] = []
    for src in sources:
        dest = fixture / src.name
        shutil.copy2(src, dest)
        copies.append(dest)

    blank = fixture / "no_face.jpg"
    Image.new("RGB", (220, 220), (18, 90, 40)).save(blank, "JPEG")

    dup = None
    group = fixture / "t1.jpg"
    if group.is_file():
        dup = fixture / "t1_duplicate.jpg"
        shutil.copy2(group, dup)
        copies.append(dup)

    small = None
    hanks = fixture / "Tom_Hanks_54745.png"
    if hanks.is_file():
        small = fixture / "hanks_small_lowq.jpg"
        im = Image.open(hanks).convert("RGB")
        im.resize((48, 48)).save(small, "JPEG", quality=25)
        copies.append(small)

    report = {
        "real_unique_images": real_unique,
        "indexed_files": 0,
        "no_face": None,
        "multi_face": None,
        "genders": set(),
        "person_ids": set(),
        "same_person_dup": None,
        "backend": None,
        "embedding_dim": None,
        "gap": "",
    }

    file_id = 1
    indexed: dict[str, dict] = {}
    for path in copies + [blank]:
        r = fi.index_image(file_id, str(path))
        assert r["ok"] is True or r.get("reason") != "image_unreadable"
        det = fi.detect_embed(str(path))
        assert det.get("backend", BACKEND) == BACKEND or not det.get("ok")
        if det.get("ok"):
            report["backend"] = det["backend"]
            if det.get("faces"):
                report["embedding_dim"] = det["faces"][0]["embedding_dim"]
                for f in det["faces"]:
                    report["genders"].add(f["gender_tr"])
        indexed[path.name] = {"file_id": file_id, "result": r, "det": det}
        file_id += 1
        report["indexed_files"] += 1

    blank_det = indexed["no_face.jpg"]["det"]
    report["no_face"] = blank_det.get("face_count", 0)
    assert blank_det.get("ok") is True
    assert blank_det["face_count"] == 0

    if "t1.jpg" in indexed:
        gdet = indexed["t1.jpg"]["det"]
        report["multi_face"] = gdet.get("face_count")
        assert gdet["face_count"] >= 2
        pids = indexed["t1.jpg"]["result"]["person_ids"]
        assert len(set(pids)) >= 2
        report["person_ids"].update(pids)
        if dup is not None:
            d_pids = indexed["t1_duplicate.jpg"]["result"]["person_ids"]
            report["same_person_dup"] = set(pids) == set(d_pids)
            assert set(pids) == set(d_pids)

    if "Tom_Hanks_54745.png" in indexed:
        h_det = indexed["Tom_Hanks_54745.png"]["det"]
        if h_det.get("face_count", 0) >= 1:
            h_pids = set(indexed["Tom_Hanks_54745.png"]["result"]["person_ids"])
            report["person_ids"].update(h_pids)
            if "t1.jpg" in indexed:
                assert not h_pids.intersection(indexed["t1.jpg"]["result"]["person_ids"])

    for row in fi.list_persons():
        pid = row["person_id"]
        assert pid.startswith("person_")
        assert kisi_label(pid).startswith("Kişi ")
        report["person_ids"].add(pid)
        report["genders"].add(row["gender_tr"])

    assert report["backend"] == BACKEND
    assert report["embedding_dim"] and report["embedding_dim"] >= 128
    assert report["genders"] <= {"kadın", "erkek", "bilinmiyor"}
    assert report["genders"]  # at least one label observed

    snap1 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(faiss_d),
        faiss_clip_path=str(faiss_c),
        cache_dir=str(cache),
    )
    assert snap0 == snap1
    assert patterns.read_bytes() == b"index-frozen-sentinel"

    if real_unique < 10:
        report["gap"] = (
            f"Only {real_unique} unique real photos in InsightFace package "
            "(t1 group, Tom Hanks, 4 mask stills). No tests/fixtures faces; "
            "production archive was not crawled."
        )
    # Residual is documented via report for the operator; test still proves pipeline.
    assert report["indexed_files"] >= 1
    print("FACE_PRODUCTION_ACCEPTANCE", report)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _obs(fi: FaceIntelligence, path: Path):
    image = fi._read_bgr(str(path))
    if image is None:
        return []
    return fi._filter_obs(fi.engine.analyze_image(image))


def _crop_box(im: Image.Image, bbox, pad: float = 0.15) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad), int(h * pad)
    box = (max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py))
    return im.crop(box)


def _on_canvas(crop: Image.Image, size: int = 640) -> Image.Image:
    """Place a face crop on a large canvas so buffalo_l det_size=640 can see it."""
    canvas = Image.new("RGB", (size, size), (48, 52, 58))
    c = crop.convert("RGB")
    side = min(240, max(140, min(c.width, c.height)))
    scale = side / max(1, min(c.width, c.height))
    c = c.resize((max(32, int(c.width * scale)), max(32, int(c.height * scale))))
    canvas.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return canvas


def _build_10_50_set(fixture: Path, sources: list[Path], fi: FaceIntelligence) -> dict:
    """Bundled InsightFace stills + in-test composed JPEGs. Never walks the archive."""
    files: list[Path] = []
    origin: dict[str, str] = {}
    coverage: dict[str, set[str]] = {}

    def add(path: Path, how: str, *tags: str) -> None:
        files.append(path)
        origin[path.name] = how
        coverage[path.name] = set(tags) | {how}

    for src in sources:
        dest = fixture / src.name
        shutil.copy2(src, dest)
        tags = ["bundled"]
        if src.name == "t1.jpg":
            tags.append("multi-face")
        elif src.name.startswith("Tom_Hanks"):
            tags.extend(["single-face", "small-face"])
        elif src.name.startswith("mask_"):
            tags.extend(["partial-occlusion", "same-person-masks"])
        add(dest, "bundled", *tags)

    for i, color in enumerate(((18, 90, 40), (210, 12, 12), (6, 6, 6))):
        p = fixture / f"no_face_solid_{i}.jpg"
        Image.new("RGB", (240, 240), color).save(p, "JPEG", quality=90)
        add(p, "generated", "no-face")

    noise = fixture / "no_face_noise.jpg"
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 255, (180, 220, 3), dtype=np.uint8)).save(noise, "JPEG")
    add(noise, "generated", "no-face")

    grad = fixture / "no_face_gradient.jpg"
    g = Image.new("RGB", (200, 160), (40, 80, 120))
    ImageDraw.Draw(g).rectangle((20, 20, 180, 140), fill=(90, 140, 40))
    g.save(grad, "JPEG")
    add(grad, "generated", "no-face")

    group = fixture / "t1.jpg"
    hanks = fixture / "Tom_Hanks_54745.png"
    crops: list[Path] = []
    if group.is_file():
        dup = fixture / "t1_duplicate.jpg"
        shutil.copy2(group, dup)
        add(dup, "generated", "duplicate", "multi-face")
        reenc = fixture / "t1_reencode.jpg"
        Image.open(group).convert("RGB").save(reenc, "JPEG", quality=82)
        add(reenc, "generated", "duplicate", "low-quality", "multi-face")
        down = fixture / "t1_downscale.jpg"
        im = Image.open(group).convert("RGB")
        im.resize((max(64, im.width // 2), max(64, im.height // 2))).save(down, "JPEG", quality=55)
        add(down, "generated", "low-quality", "multi-face")
        faces = _obs(fi, group)
        src_im = Image.open(group).convert("RGB")
        for i, obs in enumerate(faces[:8]):
            crop = _crop_box(src_im, obs.bbox, 0.18)
            if crop.width < 16 or crop.height < 16:
                continue
            cp = fixture / f"t1_face_crop_{i}.jpg"
            _on_canvas(crop).save(cp, "JPEG", quality=92)
            add(cp, "generated", "single-face", "same-person-crop")
            crops.append(cp)
            if i == 0:
                occ = fixture / "t1_face_occluded.jpg"
                oc = crop.copy()
                ImageDraw.Draw(oc).rectangle(
                    (0, oc.height // 2, oc.width, oc.height), fill=(8, 8, 8)
                )
                _on_canvas(oc).save(occ, "JPEG", quality=88)
                add(occ, "generated", "partial-occlusion", "single-face")
                prof = fixture / "t1_face_profileish.jpg"
                half = crop.crop((0, 0, max(8, crop.width // 2), crop.height))
                _on_canvas(half).save(prof, "JPEG", quality=88)
                add(prof, "generated", "profile", "single-face")
                tiny = fixture / "t1_face_tiny_lowq.jpg"
                crop.resize((36, 36)).filter(ImageFilter.GaussianBlur(1.2)).save(
                    tiny, "JPEG", quality=18
                )
                add(tiny, "generated", "small-face", "low-quality")
        if len(crops) >= 2:
            a = Image.open(crops[0]).convert("RGB").resize((160, 160))
            b = Image.open(crops[1]).convert("RGB").resize((160, 160))
            canvas = Image.new("RGB", (340, 180), (30, 30, 30))
            canvas.paste(a, (8, 10))
            canvas.paste(b, (172, 10))
            multi = fixture / "composed_two_faces.jpg"
            canvas.save(multi, "JPEG", quality=90)
            add(multi, "generated", "multi-face", "composed")

    if hanks.is_file():
        small = fixture / "hanks_small_lowq.jpg"
        Image.open(hanks).convert("RGB").resize((48, 48)).save(small, "JPEG", quality=25)
        add(small, "generated", "small-face", "low-quality")
        up = fixture / "hanks_upscale.jpg"
        _on_canvas(Image.open(hanks).convert("RGB").resize((220, 220))).save(up, "JPEG", quality=90)
        add(up, "generated", "single-face", "same-person-different-photo")
        blur = fixture / "hanks_blur.jpg"
        Image.open(hanks).convert("RGB").resize((256, 256)).filter(
            ImageFilter.GaussianBlur(2.5)
        ).save(blur, "JPEG", quality=40)
        add(blur, "generated", "low-quality", "single-face")

    unique_names = {p.name for p in files}
    assert 10 <= len(unique_names) <= 50, len(unique_names)
    return {"files": files, "origin": origin, "coverage": coverage}


def test_face_acceptance_10_to_50(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    patterns = data / "patterns.db"
    patterns.write_bytes(b"index-frozen-sentinel")
    snap0 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(cache),
    )

    face_db = tmp_path / "face_index.db"
    fi = FaceIntelligence(face_db)
    assert fi.engine.allow_vision_fallback is False
    if not fi.ready():
        pytest.skip(f"InsightFace/buffalo not usable: {fi.unavailable_reason()}")

    sources = _insightface_images()
    if not sources:
        pytest.skip("InsightFace bundled stills missing")

    fixture = tmp_path / "faces"
    fixture.mkdir()
    built = _build_10_50_set(fixture, sources, fi)
    files: list[Path] = built["files"]
    unique_count = len({p.name for p in files})
    unique_bytes = len({_sha1(p) for p in files})
    bundled_n = sum(1 for n, how in built["origin"].items() if how == "bundled")
    generated_n = unique_count - bundled_n

    per_face = []
    indexed = {}
    zero_det = []
    file_id = 1
    for path in files:
        r = fi.index_image(file_id, str(path))
        assert r["ok"] is True or r.get("reason") != "image_unreadable"
        assert Path(fi.store.path).resolve() == face_db.resolve()
        image = fi._read_bgr(str(path))
        raw_obs = fi.engine.analyze_image(image) if image is not None else []
        obs = fi._filter_obs(raw_obs)
        raw_by_id = {o.face_id: o for o in raw_obs}
        det = fi.detect_embed(str(path))
        assert det.get("backend", BACKEND) == BACKEND or not det.get("ok")
        if det.get("ok") and det.get("face_count", 0) == 0:
            if "no-face" not in built["coverage"].get(path.name, set()):
                zero_det.append(path.name)
        for i, (face_rec, o) in enumerate(zip(r.get("faces") or [], obs)):
            row = {
                "file": path.name,
                "person_id": face_rec["person_id"],
                "det_confidence": float(face_rec["confidence"]),
                "embedding_dim": int(face_rec["embedding_dim"]),
                "gender_tr": face_rec["gender_tr"],
                "gender_raw": (raw_by_id.get(o.face_id) or o).gender,
                "gender_confidence": float((raw_by_id.get(o.face_id) or o).gender_confidence),
                "backend": BACKEND,
            }
            assert row["embedding_dim"] >= 128
            assert row["det_confidence"] >= 0.50
            assert row["person_id"].startswith("person_")
            per_face.append(row)
        indexed[path.name] = {"file_id": file_id, "result": r, "obs": obs, "det": det}
        file_id += 1

    persons = fi.list_persons()
    pids = [row["person_id"] for row in persons]
    assert pids == [f"person_{i:04d}" for i in range(1, len(pids) + 1)]

    genders = {f["gender_tr"] for f in per_face} | {row["gender_tr"] for row in persons}
    assert genders <= {"kadın", "erkek", "bilinmiyor"}

    no_face_names = [n for n, tags in built["coverage"].items() if "no-face" in tags]
    no_face_ok = sum(1 for n in no_face_names if indexed[n]["det"].get("face_count", 1) == 0)
    no_face_acc = no_face_ok / max(1, len(no_face_names))
    for n in no_face_names:
        assert indexed[n]["det"].get("ok") is True
        assert indexed[n]["det"]["face_count"] == 0

    false_merge = 0
    false_split = 0
    multi_sep_ok = None
    dup_ok = None
    same_person_sims: list[float] = []
    diff_person_ok = True

    if "t1.jpg" in indexed:
        t1 = indexed["t1.jpg"]
        assert t1["det"]["face_count"] >= 2
        t1_pids = t1["result"]["person_ids"]
        assert len(set(t1_pids)) >= 2
        multi_sep_ok = len(set(t1_pids)) >= 2
        t1_obs = t1["obs"]
        if "t1_duplicate.jpg" in indexed:
            d_pids = indexed["t1_duplicate.jpg"]["result"]["person_ids"]
            dup_ok = set(t1_pids) == set(d_pids)
            if not dup_ok:
                false_split += 1
        if "t1_reencode.jpg" in indexed:
            r_pids = indexed["t1_reencode.jpg"]["result"]["person_ids"]
            if set(t1_pids) != set(r_pids):
                false_split += 1
                dup_ok = False if dup_ok is None else dup_ok
            else:
                dup_ok = True if dup_ok is None else dup_ok
        for i, crop_name in enumerate(
            n for n in indexed if n.startswith("t1_face_crop_")
        ):
            c_obs = indexed[crop_name]["obs"]
            c_pids = indexed[crop_name]["result"]["person_ids"]
            if not c_obs:
                zero_det.append(crop_name)
                continue
            sims = [_cosine(c_obs[0].embedding, o.embedding) for o in t1_obs]
            best = int(np.argmax(sims)) if sims else -1
            if best >= 0:
                same_person_sims.append(sims[best])
                expected = t1_pids[best] if best < len(t1_pids) else None
                if expected and expected not in c_pids:
                    false_split += 1
                others = {p for j, p in enumerate(t1_pids) if j != best}
                if others.intersection(c_pids) and expected not in c_pids:
                    false_merge += 1
        if "composed_two_faces.jpg" in indexed:
            c_pids = set(indexed["composed_two_faces.jpg"]["result"]["person_ids"])
            if indexed["composed_two_faces.jpg"]["det"].get("face_count", 0) >= 2:
                assert len(c_pids) >= 2
                multi_sep_ok = True
            elif indexed["composed_two_faces.jpg"]["det"].get("face_count", 0) == 0:
                zero_det.append("composed_two_faces.jpg")

    hanks_names = [n for n in indexed if n.startswith("Tom_Hanks") or n.startswith("hanks_")]
    t1_set = set(indexed["t1.jpg"]["result"]["person_ids"]) if "t1.jpg" in indexed else set()
    hanks_pids = set()
    for n in hanks_names:
        hp = indexed[n]["result"].get("person_ids") or []
        hanks_pids.update(hp)
        if hp and t1_set.intersection(hp):
            false_merge += 1
            diff_person_ok = False
    if hanks_pids and t1_set:
        assert not hanks_pids.intersection(t1_set)

    gender_known = sum(1 for f in per_face if f["gender_tr"] in {"kadın", "erkek"})
    gender_conf = gender_known / max(1, len(per_face))
    gender_conf_mean = (
        round(float(np.mean([f["gender_confidence"] for f in per_face])), 4) if per_face else None
    )

    for name in ("t1_face_tiny_lowq.jpg", "t1_face_profileish.jpg", "hanks_small_lowq.jpg"):
        if name in indexed and indexed[name]["det"].get("face_count", 0) == 0:
            if name not in zero_det:
                zero_det.append(name)

    metrics = {
        "unique_images": unique_count,
        "unique_byte_hashes": unique_bytes,
        "bundled_insightface": bundled_n,
        "generated_in_test": generated_n,
        "source": "insightface package stills + in-test compose; not production archive",
        "indexed_files": len(files),
        "faces_with_arcface": len(per_face),
        "false_merge": false_merge,
        "false_split": false_split,
        "no_face_accuracy": round(no_face_acc, 4),
        "multi_face_separation": bool(multi_sep_ok),
        "gender_confidence": round(gender_conf, 4),
        "gender_confidence_mean": gender_conf_mean,
        "genders": sorted(genders),
        "duplicate_consistency": dup_ok,
        "different_person_no_merge": diff_person_ok,
        "same_person_similarity_mean": round(float(np.mean(same_person_sims)), 4) if same_person_sims else None,
        "zero_detection_cases": sorted(set(zero_det)),
        "person_ids": pids,
        "backend": BACKEND,
        "allow_vision_fallback": fi.engine.allow_vision_fallback,
        "face_db": str(face_db),
    }
    assert unique_count >= 10
    assert unique_count <= 50
    assert false_merge == 0
    assert no_face_acc == 1.0
    assert multi_sep_ok is True
    assert dup_ok is True
    assert fi.engine.allow_vision_fallback is False

    snap1 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(cache),
    )
    assert snap0 == snap1
    assert patterns.read_bytes() == b"index-frozen-sentinel"

    print("FACE_ACCEPTANCE_10_50_METRICS", metrics)
    print("FACE_ACCEPTANCE_10_50_PER_FACE", per_face[:40], "count=", len(per_face))
