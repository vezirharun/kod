"""Stable progressive-result ordering for live score updates."""

from __future__ import annotations

from typing import Any


def stable_live_order(
    previous: list[Any],
    final: list[Any],
    *,
    tolerance: float = 0.03,
    lock_ranking: bool = True,
) -> list[Any]:
    """Keep prior order inside near-tie score bands while allowing clear moves."""
    if not lock_ranking or not previous or not final:
        return list(final)
    prior_rank = {
        int(getattr(item, "file_id", 0)): rank for rank, item in enumerate(previous)
    }
    ranked = sorted(
        final, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True
    )
    output: list[Any] = []
    start = 0
    while start < len(ranked):
        end = start + 1
        anchor = float(getattr(ranked[start], "score", 0.0))
        while end < len(ranked):
            score = float(getattr(ranked[end], "score", 0.0))
            if anchor - score >= tolerance:
                break
            end += 1
        band = ranked[start:end]
        band.sort(
            key=lambda item: prior_rank.get(int(getattr(item, "file_id", 0)), 10**9)
        )
        output.extend(band)
        start = end
    return output
