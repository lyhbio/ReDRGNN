from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from redrgnn.config import DataConfig, ModelConfig, TrainingConfig


MECHANISM_FILES = (
    "Drug__TARGETS__Protein.tsv",
    "Drug__BINDS__Protein.tsv",
    "Drug__OFF_TARGETS__Protein.tsv",
    "Drug__UPREGULATES__Gene.tsv",
    "Drug__DOWNREGULATES__Gene.tsv",
    "Gene__ASSOCIATES_WITH__Disease.tsv",
    "Protein__ASSOCIATED_WITH__Disease.tsv",
    "Protein__THERAPEUTIC_TARGET_IN__Disease.tsv",
    "Disease__IS_A__Disease.tsv",
)


def _write_rows(path: Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


@pytest.fixture()
def tiny_inputs(tmp_path: Path):
    positive = tmp_path / "RePUN-P.txt"
    negative = tmp_path / "RePUN-N.txt"
    test = tmp_path / "RePUN-T.txt"
    holdout = tmp_path / "holdout.tsv"
    _write_rows(
        positive,
        [
            ("d1", "x1", 1),
            ("d1", "x2", 1),
            ("d2", "x2", 1),
            ("d3", "x3", 1),
            ("d4", "x4", 1),
        ],
    )
    _write_rows(
        negative,
        [
            ("d1", "x3", 0),
            ("d2", "x4", 0),
            ("d3", "x1", 0),
            ("d4", "x2", 0),
            ("d2", "x1", 0),
        ],
    )
    _write_rows(test, [("d1", "x1", 1), ("d1", "x3", 0)])
    _write_rows(holdout, [("d4", "x4", 1)])

    drug_embeddings = tmp_path / "drug_embeddings.npy"
    disease_embeddings = tmp_path / "disease_embeddings.npy"
    kg_embeddings = tmp_path / "entity_embeddings.npy"
    rng = np.random.default_rng(7)
    np.save(drug_embeddings, rng.normal(size=(5, 6)).astype(np.float32))
    np.save(disease_embeddings, rng.normal(size=(5, 6)).astype(np.float32))

    drug_index = tmp_path / "drug_index.tsv"
    disease_index = tmp_path / "disease_index.tsv"
    index_header = (
        "row_index",
        "source_line",
        "entity_type",
        "entity",
        "norm_entity",
        "standard_name",
        "description_source",
        "description_words",
    )
    _write_rows(
        drug_index,
        [index_header]
        + [(index, index + 1, "DRUG", f"d{index + 1}", f"d{index + 1}", f"d{index + 1}", "test", 10) for index in range(5)],
    )
    _write_rows(
        disease_index,
        [index_header]
        + [(index, index + 1, "DISEASE", f"x{index + 1}", f"x{index + 1}", f"x{index + 1}", "test", 12) for index in range(5)],
    )
    quality_metadata = tmp_path / "entity_quality.tsv"
    quality_header = (
        "entity_type",
        "entity",
        "needs_review",
        "description_source",
        "description_words",
        "empty_description",
    )
    _write_rows(
        quality_metadata,
        [quality_header]
        + [("DRUG", f"d{index}", int(index == 1), "test", 10, 0) for index in range(1, 6)]
        + [("DISEASE", f"x{index}", 0, "test", 12, 0) for index in range(1, 6)],
    )

    entities = [
        *(f"Drug::d{index}" for index in range(1, 6)),
        *(f"Disease::x{index}" for index in range(1, 6)),
        "Protein::p1",
        "Gene::g1",
    ]
    np.save(kg_embeddings, rng.normal(size=(len(entities), 4)).astype(np.float32))
    kg_index = tmp_path / "entities.tsv"
    _write_rows(kg_index, [("id", "entity")] + [(index, entity) for index, entity in enumerate(entities)])

    edges = tmp_path / "edges_by_relation"
    edges.mkdir()
    for filename in MECHANISM_FILES:
        _write_rows(edges / filename, [("head_name", "relation", "tail_name")])
    _write_rows(
        edges / "Drug__TARGETS__Protein.tsv",
        [("head_name", "relation", "tail_name"), ("d1", "TARGETS", "p1"), ("d2", "TARGETS", "p1")],
    )
    _write_rows(
        edges / "Gene__ASSOCIATES_WITH__Disease.tsv",
        [("head_name", "relation", "tail_name"), ("g1", "ASSOCIATES_WITH", "x1"), ("g1", "ASSOCIATES_WITH", "x2")],
    )
    _write_rows(
        edges / "Disease__IS_A__Disease.tsv",
        [("head_name", "relation", "tail_name"), ("x1", "IS_A", "x5")],
    )
    _write_rows(
        edges / "Drug__TREATS__Disease.tsv",
        [("head_name", "relation", "tail_name"), ("d2", "TREATS", "x2")],
    )
    _write_rows(
        edges / "Drug__CONTRAINDICATES__Disease.tsv",
        [("head_name", "relation", "tail_name"), ("d2", "CONTRAINDICATES", "x4")],
    )

    data = DataConfig(
        positive_pairs=positive,
        negative_pairs=negative,
        test_pairs=test,
        holdout_pairs=holdout,
        drug_embeddings=drug_embeddings,
        disease_embeddings=disease_embeddings,
        kg_entity_embeddings=kg_embeddings,
        drug_index=drug_index,
        disease_index=disease_index,
        quality_metadata=quality_metadata,
        kg_entity_index=kg_index,
        kg_edges_dir=edges,
    )
    model = ModelConfig(
        hidden_dim=8,
        topology_layers=1,
        similarity_layers=1,
        knn_k=2,
        max_neighbors=3,
        dropout=0.0,
    )
    training = TrainingConfig(
        seeds=(11,),
        validation_fraction=0.34,
        epochs=2,
        batch_size=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        positive_weight=1.0,
        negative_weight=2.0,
        contrastive_weight=0.05,
        temperature=0.2,
        minimum_epochs=1,
        patience=1,
        gradient_clip_norm=5.0,
    )
    return data, model, training
