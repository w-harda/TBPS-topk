from pathlib import Path

from attributes.canonicalizer import Canonicalizer
from attributes.ontology import AttributeOntology


ROOT = Path(__file__).resolve().parents[1]


def make_canonicalizer() -> Canonicalizer:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    return Canonicalizer.load(ontology, ROOT / "ontology" / "alias_map.json")


def test_query_example_preserves_raw_text_and_character_spans() -> None:
    text = "A man wearing a red shirt with a backpack."
    attributes = make_canonicalizer().extract(text)
    by_canonical = {attribute.canonical: attribute for attribute in attributes}
    assert by_canonical["gender:male"].raw == "man"
    assert by_canonical["upper_red:positive"].raw == "red shirt"
    assert by_canonical["upper_red:positive"].span == (16, 25)
    assert by_canonical["backpack:positive"].span == (33, 41)
    for attribute in attributes:
        assert text[slice(*attribute.span)] == attribute.raw


def test_long_negative_alias_suppresses_overlapping_positive_alias() -> None:
    attributes = make_canonicalizer().extract("A woman without a hat and no backpack.")
    canonical = {attribute.canonical for attribute in attributes}
    assert "hat:negative" in canonical
    assert "hat:positive" not in canonical
    assert "backpack:negative" in canonical
    assert "backpack:positive" not in canonical


def test_unknown_attributes_are_omitted_not_turned_negative() -> None:
    attributes = make_canonicalizer().extract("A person standing near a wall.")
    canonical = {attribute.canonical for attribute in attributes}
    assert "backpack:negative" not in canonical
    assert "hat:negative" not in canonical


def test_mapping_candidate_and_unmapped_output() -> None:
    mapped, unmapped = make_canonicalizer().generate_candidates(
        {"dark red shirt": 50, "unusual garment": 45}
    )
    assert mapped[0].candidates == ("upper_red:positive",)
    assert mapped[0].match_type == "contained_alias"
    assert unmapped == [{"raw": "unusual garment", "count": 45}]


def test_one_phrase_can_map_to_multiple_attribute_dimensions() -> None:
    attributes = make_canonicalizer().extract("A person in black shorts.")
    canonical = {attribute.canonical for attribute in attributes}
    assert {
        "lower_black:positive",
        "lower_length:short",
        "lower_type:pants_or_shorts",
    } <= canonical


def test_controlled_composition_rules_cover_real_cuhk_phrases() -> None:
    canonicalizer = make_canonicalizer()
    cases = {
        "short black hair": {"hair_length:short"},
        "long dark hair": {"hair_length:long"},
        "white t-shirt": {"upper_white:positive"},
        "black leather jacket": {"upper_black:positive"},
        "black and grey striped tank": {
            "upper_black:positive",
            "upper_gray:positive",
        },
        "green leggings": {
            "lower_green:positive",
            "lower_type:pants_or_shorts",
        },
        "gray slacks": {
            "lower_gray:positive",
            "lower_type:pants_or_shorts",
        },
    }

    for text, expected in cases.items():
        attributes = canonicalizer.extract(text)
        assert expected <= {attribute.canonical for attribute in attributes}
        assert all(text[slice(*attribute.span)] == attribute.raw for attribute in attributes)


def test_upper_color_rule_does_not_cross_lower_garment() -> None:
    attributes = make_canonicalizer().extract("black pants and white shirt")
    canonical = {attribute.canonical for attribute in attributes}
    assert "upper_white:positive" in canonical
    assert "upper_black:positive" not in canonical
    assert "lower_black:positive" in canonical


def test_lower_color_rule_does_not_cross_upper_garment() -> None:
    attributes = make_canonicalizer().extract("white shirt and black pants")
    canonical = {attribute.canonical for attribute in attributes}
    assert "lower_black:positive" in canonical
    assert "lower_white:positive" not in canonical
    assert "upper_white:positive" in canonical


def test_literal_negative_alias_still_wins_over_composed_positive() -> None:
    attributes = make_canonicalizer().extract("not wearing a black shirt")
    canonical = {attribute.canonical for attribute in attributes}
    assert "upper_black:negative" in canonical
    assert "upper_black:positive" not in canonical


def test_phrase_candidate_generation_uses_composition_rules() -> None:
    mapped, unmapped = make_canonicalizer().generate_candidates(
        {
            "short black hair": 8,
            "black leather jacket": 7,
            "gray slacks": 5,
        }
    )
    by_raw = {candidate.raw: set(candidate.candidates) for candidate in mapped}
    assert by_raw["short black hair"] == {"hair_length:short"}
    assert by_raw["black leather jacket"] == {"upper_black:positive"}
    assert by_raw["gray slacks"] == {
        "lower_gray:positive",
        "lower_type:pants_or_shorts",
    }
    assert unmapped == []


def test_conservative_lower_types_do_not_infer_length() -> None:
    canonicalizer = make_canonicalizer()
    for garment in ("leggings", "slacks", "tights"):
        canonical = {
            attribute.canonical for attribute in canonicalizer.extract(f"black {garment}")
        }
        assert "lower_type:pants_or_shorts" in canonical
        assert not any(value.startswith("lower_length:") for value in canonical)
