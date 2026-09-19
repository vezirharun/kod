from pathlib import Path


def test_result_card_contains_visual_rank_badge():
    src = Path("ui/result_card.py").read_text(encoding="utf-8")
    assert "#" in src
    assert "score_text = f"#{self._rank} · {score_text}"" in src
    assert "self.rank_lbl = QLabel" not in src


def test_virtual_list_passes_rank_to_card():
    src = Path("ui/virtual_results_list.py").read_text(encoding="utf-8")
    assert "rank=int(rank or 0)" in src


def test_virtual_list_refreshes_rank_badge_when_recycling_card():
    src = Path("ui/virtual_results_list.py").read_text(encoding="utf-8")
    assert "card._rank = int(rank or 0)\n                    card._update_rank_badge()" in src
