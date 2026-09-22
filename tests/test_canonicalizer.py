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
