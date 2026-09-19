
from core.query_intent_router import classify_query
from core.universal_visual_intel import parse_universal_query, clip_nodes_for_query, node_clip_prompt

def test_gender_queries_are_universal_semantic_concepts():
    for text, node in (("kadın", "female_person"), ("erkek", "male_person")):
        assert classify_query(text).kind == "gender"
        q = parse_universal_query(text)
        assert q.node_id == node
        nodes = clip_nodes_for_query(q)
        assert nodes[0] == node
        assert "person" in nodes
        prompt = node_clip_prompt(node).lower()
        assert "person" in prompt
        assert "textile" in prompt

def test_gender_queries_are_distinct():
    assert parse_universal_query("kadın").node_id != parse_universal_query("erkek").node_id
