from pathlib import Path
import hashlib
import json

from attributes.ontology import AttributeOntology


ROOT = Path(__file__).resolve().parents[1]


def test_official_prompt_order_and_count() -> None:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    assert len(ontology.attributes) == 27
    assert len(ontology.prompts) == 54
    assert ontology.prompt_texts[:4] == (
        "the person is a woman",
        "the person is a man",
        "the person is younger than 18 years old",
        "the person is older than 18 years old",
    )
    assert ontology.prompt_texts[-2:] == (
        "the person wears brown lower clothes",
        "the person does not wear brown lower clothes",
    )
    prompt_digest = hashlib.sha256(
        json.dumps(
            list(ontology.prompt_texts), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert prompt_digest == "27aeb2965dd4efa97b52776a9e18640f2b99c3af6269b120f018f161b31e7559"


def test_prompt_semantics_are_explicit_not_position_assumptions() -> None:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    hat = ontology.attributes[3]
    assert hat.prompts[0].canonical == "hat:positive"
    assert hat.prompts[1].canonical == "hat:negative"
    assert hat.prompts[0].semantic == "present"
    assert hat.prompts[1].semantic == "absent"
