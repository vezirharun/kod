import ast
from pathlib import Path


def test_merge_semantic_intel_has_no_undefined_limit_reference():
    src = Path(__file__).resolve().parents[1] / "core" / "search_engine.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_merge_semantic_intel"
    )
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    # The old implementation referenced `limit` although only
    # `max_results` existed in this function's signature.
    assert "limit" not in (names - args), "undefined local 'limit' remains"


def test_merge_semantic_intel_uses_max_results_variable():
    src = Path(__file__).resolve().parents[1] / "core" / "search_engine.py"
    text = src.read_text(encoding="utf-8")
    assert "max_results=result_limit" in text
