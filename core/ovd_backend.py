"""Open-vocab detector backends. Index-time only; search never constructs these."""
from __future__ import annotations

from typing import Any, Protocol


class OVDBackend(Protocol):
    name: str
    version: str

    def detect(self, image: Any, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
        """Return boxes: label, confidence, xyxy (pixel)."""
        ...


class NullOVDBackend:
    name = "null"
    version = "0"

    def detect(self, image: Any, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
        return []


_GDINO_SINGLETON: "GroundingDINOBackend | None" = None


def get_grounding_dino_backend(*, use_gpu: bool = False) -> "GroundingDINOBackend":
    global _GDINO_SINGLETON
    if _GDINO_SINGLETON is None:
        _GDINO_SINGLETON = GroundingDINOBackend(use_gpu=use_gpu)
    return _GDINO_SINGLETON


class GroundingDINOBackend:
    """Grounding DINO-Tiny. Index-time only; search never constructs this."""

    name = "grounding_dino"
    version = "tiny-v1"

    def __init__(self, *, use_gpu: bool = False) -> None:
        self._proc = None
        self._model = None
        self._torch = None
        self._use_gpu = bool(use_gpu)
        self._device = "cpu"

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        name = "IDEA-Research/grounding-dino-tiny"
        self._proc = AutoProcessor.from_pretrained(name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(name).eval()
        self._torch = torch
        self._device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def detect(self, image: Any, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
        if not prompts:
            return []
        self._ensure()
        torch = self._torch
        im = image
        w, h = im.size
        m = max(w, h)
        scale = 1.0
        if m > 768:
            scale = 768.0 / m
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        text = " . ".join(p.strip() for p in prompts if str(p).strip())
        if text and not text.endswith("."):
            text += "."
        with torch.no_grad():
            inputs = self._proc(images=im, text=text, return_tensors="pt")
            if self._device != "cpu":
                inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
            out = self._model(**inputs)
            r = self._proc.post_process_grounded_object_detection(
                out,
                input_ids=inputs["input_ids"],
                threshold=float(threshold),
                text_threshold=float(threshold),
                target_sizes=[im.size[::-1]],
            )[0]
        raw_boxes = r.get("boxes")
        raw_scores = r.get("scores")
        labs = r.get("text_labels")
        if labs is None:
            labs = r.get("labels")
        if raw_boxes is None:
            return []
        n = int(len(raw_boxes))
        boxes: list[dict[str, Any]] = []
        inv = (1.0 / scale) if scale else 1.0
        for i in range(n):
            box = raw_boxes[i]
            xy = [float(x) * inv for x in box.tolist()]
            if labs is None:
                lab = prompts[0] if prompts else ""
            else:
                item = labs[i]
                lab = str(item.item() if hasattr(item, "item") and not isinstance(item, str) else item)
            conf = raw_scores[i] if raw_scores is not None else 0.0
            boxes.append({
                "label": lab.strip().rstrip("."),
                "confidence": float(conf),
                "xyxy": xy,
            })
        return boxes


_OWLV2_SINGLETON: "Owlv2Backend | None" = None


def get_owlv2_backend(*, use_gpu: bool = False) -> "Owlv2Backend":
    global _OWLV2_SINGLETON
    if _OWLV2_SINGLETON is None:
        _OWLV2_SINGLETON = Owlv2Backend(use_gpu=use_gpu)
    return _OWLV2_SINGLETON


class Owlv2Backend:
    """OWLv2. Index-time only; search never constructs this."""

    name = "owlv2"
    version = "base-patch16"

    def __init__(self, *, use_gpu: bool = False) -> None:
        self._proc = None
        self._model = None
        self._torch = None
        self._use_gpu = bool(use_gpu)
        self._device = "cpu"

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        name = "google/owlv2-base-patch16"
        self._proc = Owlv2Processor.from_pretrained(name)
        self._model = Owlv2ForObjectDetection.from_pretrained(name).eval()
        self._torch = torch
        self._device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
        if self._device != "cpu":
            self._model = self._model.to(self._device)

    def detect(self, image: Any, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
        if not prompts:
            return []
        self._ensure()
        torch = self._torch
        im = image
        w, h = im.size
        m = max(w, h)
        sx = sy = 1.0
        if m > 768:
            scale = 768.0 / m
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            im = im.resize((nw, nh))
            sx, sy = w / float(nw), h / float(nh)
        inputs = self._proc(text=list(prompts), images=im, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        res = self._proc.post_process_grounded_object_detection(
            out, threshold=float(threshold), target_sizes=torch.tensor([im.size[::-1]]),
        )[0]
        raw_boxes = res.get("boxes")
        if raw_boxes is None:
            return []
        n = int(len(raw_boxes))
        texts = res.get("text_labels")
        labs = res.get("labels")
        scores = res.get("scores")
        boxes: list[dict[str, Any]] = []
        for i in range(n):
            xy = [float(x) for x in raw_boxes[i].tolist()]
            xy = [xy[0] * sx, xy[1] * sy, xy[2] * sx, xy[3] * sy]
            sc = float(scores[i]) if scores is not None else 0.0
            if texts is not None:
                lab = str(texts[i])
            else:
                lab = prompts[int(labs[i])]
            boxes.append({"label": lab, "confidence": sc, "xyxy": xy})
        return boxes
