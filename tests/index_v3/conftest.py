"""V3 tests must not load the real OWLv2 weights."""

from __future__ import annotations

import pytest

from core.ovd_backend import NullOVDBackend


@pytest.fixture(autouse=True)
def _null_owlv2_backend(monkeypatch):
    monkeypatch.setattr(
        "core.ovd_backend.get_owlv2_backend",
        lambda use_gpu=False: NullOVDBackend(),
    )
