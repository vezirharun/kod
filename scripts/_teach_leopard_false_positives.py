"""Teach confirmed non-leopard hits from the latest leopard search."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.category_learning import apply_manual_category
from core.feedback_learn import apply_wrong_match_correction
from core.pattern_classifier import filename_pattern_hint
from core.settings import AppSettings

QUERY = r"C:\Users\HARUN\Desktop\unnamed (14).jpg"


def main() -> None:
    settings = AppSettings.load()
    print("fiori hint", filename_pattern_hint("10197DA fiori  .jpg", ""))
    print("catene hint", filename_pattern_hint("6938 SL catene r9.jpg", ""))

    floral = apply_wrong_match_correction(
        settings,
        query_path=QUERY,
        result_file_id=154712,
        category_path="Floral/Mixed Floral",
        reject_query_family="animal_print",
        pattern_family="floral",
        pattern_subtype="mixed_floral",
        tag="fiori",
    )
    print("taught floral 154712", floral)

    mixed1 = apply_manual_category(
        settings,
        155847,
        "Animal Print/Mixed Animal Print",
        propagate_exact=True,
        refresh_visual=True,
    )
    print("taught mixed 155847 INC0900909", mixed1)

    mixed2 = apply_manual_category(
        settings,
        155478,
        "Animal Print/Mixed Animal Print",
        propagate_exact=True,
        refresh_visual=True,
    )
    print("taught mixed 155478 catene", mixed2)


if __name__ == "__main__":
    main()
