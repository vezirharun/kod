from core.adaptive_retrieve import (
    is_relevant_hit,
    next_retrieve_k,
    should_stop_expansion,
)


class _R:
    def __init__(self, score, **kw):
        self.score = score
        self.is_self_match = kw.get("self", False)
        self.debug = kw.get("debug") or {}
        self.same_pattern_family = kw.get("fam", False)
        self.same_animal_family = False
        self.cluster_group = kw.get("cg", "")


def test_next_k_doubles_until_index():
    assert next_retrieve_k(800, 11769) == 1600
    assert next_retrieve_k(1600, 11769) == 3200
    assert next_retrieve_k(6400, 11769) == 11769
    assert next_retrieve_k(11769, 11769) is None


def test_stop_when_no_new_and_when_similar():
    stop, why = should_stop_expansion(
        new_id_count=0,
        relevant_new=0,
        max_new_score=0,
        faiss_median_new=None,
        faiss_tail_ref=0.4,
        floor=0.4,
    )
    assert stop and why == "no_new_ids"
    stop, why = should_stop_expansion(
        new_id_count=800,
        relevant_new=1,
        max_new_score=0.2,
        faiss_median_new=0.1,
        faiss_tail_ref=0.4,
        floor=0.4,
    )
    assert stop
    stop, why = should_stop_expansion(
        new_id_count=800,
        relevant_new=40,
        max_new_score=0.82,
        faiss_median_new=0.38,
        faiss_tail_ref=0.4,
        floor=0.4,
    )
    assert not stop


def test_relevant_uses_evidence_not_faiss_membership():
    assert is_relevant_hit(_R(0.2, cg="unrelated"), 0.4) is False
    assert is_relevant_hit(_R(0.2, fam=True), 0.4) is True
    assert is_relevant_hit(_R(0.9), 0.4) is True
    from core.adaptive_retrieve import is_expand_worthy
    assert is_expand_worthy(_R(0.41)) is False
    assert is_expand_worthy(_R(0.70)) is True
    assert is_expand_worthy(_R(0.2, fam=True)) is True
