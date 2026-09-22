from __future__ import annotations

import json
from pathlib import Path

from attributes.canonicalizer import Canonicalizer
from attributes.gallery_index import GalleryAttributeIndex
from attributes.ontology import AttributeOntology
from attributes.subset import (
    QueryRecord,
    evaluate_query_subset,
    load_gallery_manifest,
    load_query_records,
)


ROOT = Path(__file__).resolve().parents[1]


def make_canonicalizer() -> Canonicalizer:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    return Canonicalizer.load(ontology, ROOT / "ontology" / "alias_map.json")


def test_gallery_manifest_limit_and_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in ("one.jpg", "two.jpg"):
        (dataset_root / name).write_bytes(b"image")
    manifest = tmp_path / "gallery.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_id": "train", "path": "two.jpg", "split": "train"},
                {"image_id": "g1", "file_path": "one.jpg", "split": "test"},
                {"image_id": "g1", "file_path": "one.jpg", "split": "test"},
                {"image_id": "g2", "image": "two.jpg", "split": "test"},
            ]
        ),
        encoding="utf-8",
    )
    images = load_gallery_manifest(manifest, dataset_root, limit=2, split_name="test")
    assert images == [
        ("g1:one.jpg", (dataset_root / "one.jpg").resolve()),
        ("g2:two.jpg", (dataset_root / "two.jpg").resolve()),
    ]


def test_gallery_manifest_keeps_different_images_for_the_same_person(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in ("one.jpg", "two.jpg"):
        (dataset_root / name).write_bytes(b"image")
    manifest = tmp_path / "gallery.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_id": 12004, "image": "one.jpg"},
                {"image_id": 12004, "image": "two.jpg"},
            ]
        ),
        encoding="utf-8",
    )

    images = load_gallery_manifest(manifest, dataset_root)

    assert images == [
        ("12004:one.jpg", (dataset_root / "one.jpg").resolve()),
        ("12004:two.jpg", (dataset_root / "two.jpg").resolve()),
    ]


def test_gallery_manifest_deduplicates_repeated_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "one.jpg").write_bytes(b"image")
    manifest = tmp_path / "gallery.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_id": 12004, "image": "one.jpg"},
                {"image_id": 99999, "image": "one.jpg"},
            ]
        ),
        encoding="utf-8",
    )

    images = load_gallery_manifest(manifest, dataset_root)

    assert images == [("12004:one.jpg", (dataset_root / "one.jpg").resolve())]


def test_query_loader_supports_cuhk_caption_lists_and_processed_records(tmp_path: Path) -> None:
    raw_annotations = tmp_path / "raw.json"
    raw_annotations.write_text(
        json.dumps(
            [
                {"id": 10, "split": "train", "captions": ["ignored train"]},
                {"id": 20, "split": "test", "captions": ["first", "second"]},
            ]
        ),
        encoding="utf-8",
    )
    raw_queries = load_query_records(raw_annotations, limit=2, split_name="test")
    assert [query.text for query in raw_queries] == ["first", "second"]
    assert len({query.query_id for query in raw_queries}) == 2

    processed_annotations = tmp_path / "processed.json"
    processed_annotations.write_text(
        json.dumps([{"image_id": "p1", "caption": "a red shirt"}]), encoding="utf-8"
    )
    processed_queries = load_query_records(processed_annotations, split_name="test")
    assert processed_queries[0].text == "a red shirt"


def test_subset_results_and_coverage_summary() -> None:
    queries = [
        QueryRecord("q1", "A man in a red shirt with a backpack."),
        QueryRecord("q2", "A person standing beside a wall."),
        QueryRecord("q3", "A backpack."),
    ]
    gallery = GalleryAttributeIndex(
        {
            "g1": {"gender:male", "upper_red:positive", "backpack:positive"},
            "g2": {"gender:male", "upper_red:positive", "backpack:negative"},
            "g3": {"gender:female", "backpack:positive"},
        }
    )
    results, summary = evaluate_query_subset(
        queries,
        make_canonicalizer(),
        gallery,
        alpha=1.0,
        beta=0.6,
        k_min=1,
        k_max=5,
        low_coverage_max_attributes=1,
    )
    assert results[0]["query_id"] == "q1"
    assert results[0]["attributes"]
    assert all("span" in attribute for attribute in results[0]["attributes"])
    assert all("score" in score for score in results[0]["scores"])
    assert results[0]["dynamic_k"] >= 1
    assert results[1]["coverage_status"] == "none"
    assert results[1]["dynamic_k"] == 0
    assert results[2]["coverage_status"] == "low"
    assert summary["total_queries"] == 3
    assert summary["queries_with_attributes"] == 2
    assert summary["zero_attribute_queries"] == 1
    assert summary["low_coverage_queries"] == 1
    assert summary["k_distribution"]["0"] == 1
    assert summary["canonical_attribute_frequency"]["backpack:positive"] == 2
