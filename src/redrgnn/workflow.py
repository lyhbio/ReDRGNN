from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import ExperimentConfig
from .data import (
    PairRecord,
    PreparedPairs,
    load_holdout_keys,
    pair_entity_sets,
    prepare_supervised_pairs,
)
from .graph import ALL_RELATIONS, PreparedGraph, prepare_graph


@dataclass(frozen=True)
class PreparedRun:
    pairs: PreparedPairs
    graph_data: PreparedGraph
    audit: Mapping[str, object]


def prepare_run(
    config: ExperimentConfig,
    seed: int,
) -> PreparedRun:
    pairs = prepare_supervised_pairs(
        config.data.positive_pairs,
        config.data.negative_pairs,
        config.data.test_pairs,
        config.data.holdout_pairs,
        config.data.kg_edges_dir,
        config.training.validation_fraction,
        seed,
    )
    supervised_and_test: Sequence[PairRecord] = pairs.train + pairs.validation + pairs.test
    seed_drugs, seed_diseases = pair_entity_sets(supervised_and_test)
    holdout_keys = load_holdout_keys(config.data.holdout_pairs)
    seed_drugs.update(drug for drug, _ in holdout_keys)
    seed_diseases.update(disease for _, disease in holdout_keys)

    leakage_exclusions = {
        record.key for record in pairs.validation + pairs.test
    } | holdout_keys
    graph_data = prepare_graph(
        config.data,
        config.model,
        seed_drugs,
        seed_diseases,
        seed,
        leakage_exclusions,
    )
    coverage = coverage_audit(supervised_and_test, graph_data)
    audit = {
        "pairs": pairs.audit,
        "graph": graph_data.audit,
        "coverage": coverage,
    }
    return PreparedRun(pairs, graph_data, audit)


def coverage_audit(records: Sequence[PairRecord], graph_data: PreparedGraph) -> dict[str, object]:
    features = graph_data.features
    text_pairs = 0
    kg_pairs = 0
    any_pairs = 0
    for record in records:
        drug = features.drug_to_node[record.drug]
        disease = features.disease_to_node[record.disease]
        drug_text = bool(features.text_available[drug])
        disease_text = bool(features.text_available[disease])
        drug_kg = bool(features.kg_available[drug])
        disease_kg = bool(features.kg_available[disease])
        text_pairs += int(drug_text and disease_text)
        kg_pairs += int(drug_kg and disease_kg)
        any_pairs += int((drug_text or drug_kg) and (disease_text or disease_kg))
    total = len(records)
    return {
        "pairs": total,
        "both_text": text_pairs,
        "both_text_rate": text_pairs / max(total, 1),
        "both_kg": kg_pairs,
        "both_kg_rate": kg_pairs / max(total, 1),
        "both_any_embedding": any_pairs,
        "both_any_embedding_rate": any_pairs / max(total, 1),
    }


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(config: ExperimentConfig) -> dict[str, object]:
    paths = {
        "positive_pairs": config.data.positive_pairs,
        "negative_pairs": config.data.negative_pairs,
        "test_pairs": config.data.test_pairs,
        "drug_embeddings": config.data.drug_embeddings,
        "disease_embeddings": config.data.disease_embeddings,
        "kg_entity_embeddings": config.data.kg_entity_embeddings,
        "drug_index": config.data.drug_index,
        "disease_index": config.data.disease_index,
        "quality_metadata": config.data.quality_metadata,
        "kg_entity_index": config.data.kg_entity_index,
    }
    if config.data.holdout_pairs is not None:
        paths["holdout_pairs"] = config.data.holdout_pairs
    for relation in ALL_RELATIONS:
        paths[f"kg_edge:{relation.relation}"] = config.data.kg_edges_dir / relation.filename
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in paths.items()
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def aggregate_seed_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metric_names = ("AUROC", "AUPR", "Accuracy", "Precision", "Recall", "Specificity", "F1")
    output: dict[str, object] = {"seeds": len(rows)}
    for metric in metric_names:
        values = np.asarray(
            [float(row["test_metrics_at_0p5"][metric]) for row in rows],
            dtype=np.float64,
        )
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return output
