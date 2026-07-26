from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import DataConfig, ModelConfig
from .data import normalize_name


@dataclass(frozen=True)
class RelationSpec:
    filename: str
    relation: str
    head_type: str
    tail_type: str
    center_side: str


DIRECT_RELATIONS: tuple[RelationSpec, ...] = (
    RelationSpec("Drug__TREATS__Disease.tsv", "TREATS", "DRUG", "DISEASE", "head"),
    RelationSpec(
        "Drug__CONTRAINDICATES__Disease.tsv",
        "CONTRAINDICATES",
        "DRUG",
        "DISEASE",
        "head",
    ),
)

MECHANISM_RELATIONS: tuple[RelationSpec, ...] = (
    RelationSpec("Drug__TARGETS__Protein.tsv", "TARGETS", "DRUG", "PROTEIN", "head"),
    RelationSpec("Drug__BINDS__Protein.tsv", "BINDS", "DRUG", "PROTEIN", "head"),
    RelationSpec("Drug__OFF_TARGETS__Protein.tsv", "OFF_TARGETS", "DRUG", "PROTEIN", "head"),
    RelationSpec("Drug__UPREGULATES__Gene.tsv", "UPREGULATES", "DRUG", "GENE", "head"),
    RelationSpec("Drug__DOWNREGULATES__Gene.tsv", "DOWNREGULATES", "DRUG", "GENE", "head"),
    RelationSpec("Gene__ASSOCIATES_WITH__Disease.tsv", "ASSOCIATES_WITH", "GENE", "DISEASE", "tail"),
    RelationSpec(
        "Protein__ASSOCIATED_WITH__Disease.tsv",
        "ASSOCIATED_WITH",
        "PROTEIN",
        "DISEASE",
        "tail",
    ),
    RelationSpec(
        "Protein__THERAPEUTIC_TARGET_IN__Disease.tsv",
        "THERAPEUTIC_TARGET_IN",
        "PROTEIN",
        "DISEASE",
        "tail",
    ),
    RelationSpec("Disease__IS_A__Disease.tsv", "IS_A", "DISEASE", "DISEASE", "both"),
)

ALL_RELATIONS: tuple[RelationSpec, ...] = DIRECT_RELATIONS + MECHANISM_RELATIONS
RELATION_NAMES: tuple[str, ...] = tuple(sorted(spec.relation for spec in MECHANISM_RELATIONS))
NODE_TYPES: tuple[str, ...] = ("DRUG", "DISEASE", "PROTEIN", "GENE")


@dataclass(frozen=True)
class TextEntry:
    row: int


@dataclass(frozen=True)
class TextTable:
    embeddings: np.ndarray
    aliases: Mapping[str, TextEntry]
    primary_metadata: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class FeatureStore:
    node_names: tuple[str, ...]
    node_types: tuple[str, ...]
    text_embeddings: np.ndarray
    kg_embeddings: np.ndarray
    quality_features: np.ndarray
    missing_flags: np.ndarray
    text_available: np.ndarray
    kg_available: np.ndarray
    drug_to_node: Mapping[str, int]
    disease_to_node: Mapping[str, int]
    source_vocabulary: tuple[str, ...]
    type_vocabulary: tuple[str, ...]

    @property
    def text_dim(self) -> int:
        return int(self.text_embeddings.shape[1])

    @property
    def kg_dim(self) -> int:
        return int(self.kg_embeddings.shape[1])

    @property
    def quality_dim(self) -> int:
        return int(self.quality_features.shape[1])


@dataclass(frozen=True)
class GraphBundle:
    relations: tuple[str, ...]
    topology_edges: Mapping[str, tuple[np.ndarray, np.ndarray]]
    similarity_edges: tuple[np.ndarray, np.ndarray]
    stats: Mapping[str, object]


@dataclass(frozen=True)
class PreparedGraph:
    features: FeatureStore
    graph: GraphBundle
    audit: Mapping[str, object]


TypedNode = tuple[str, str]
SampledNeighbors = dict[str, dict[TypedNode, list[TypedNode]]]


def _load_text_table(index_path: Path, embedding_path: Path, expected_type: str) -> TextTable:
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix: {embedding_path}")

    aliases: dict[str, TextEntry] = {}
    primary_metadata: dict[str, dict[str, object]] = {}
    row_count = 0
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"row_index", "entity_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Text index is missing required columns {sorted(required)}: {index_path}")
        for line_number, row in enumerate(reader, 2):
            entity_type = str(row.get("entity_type", "")).strip().upper()
            if entity_type != expected_type:
                raise ValueError(
                    f"Unexpected entity_type at {index_path}:{line_number}: {entity_type!r}"
                )
            embedding_row = int(row["row_index"])
            if not 0 <= embedding_row < embeddings.shape[0]:
                raise ValueError(f"row_index out of bounds at {index_path}:{line_number}")
            try:
                words = float(row.get("description_words", 0) or 0)
            except ValueError:
                words = 0.0
            entry = TextEntry(embedding_row)
            primary = normalize_name(row.get("norm_entity", ""))
            if primary:
                primary_metadata[primary] = {
                    "description_source": str(row.get("description_source", "")),
                    "description_words": words,
                }
            names = (
                primary,
                normalize_name(row.get("entity", "")),
                normalize_name(row.get("standard_name", "")),
            )
            for name in names:
                if name:
                    aliases.setdefault(name, entry)
            row_count += 1
    if row_count != embeddings.shape[0]:
        raise ValueError(
            f"Index/embedding row mismatch for {index_path}: {row_count} vs {embeddings.shape[0]}"
        )
    return TextTable(embeddings, aliases, primary_metadata)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_quality_metadata(path: Path) -> dict[TypedNode, dict[str, object]]:
    metadata: dict[TypedNode, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "entity_type",
            "entity",
            "needs_review",
            "description_source",
            "description_words",
            "empty_description",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Quality metadata is missing columns {sorted(required)}: {path}")
        for line_number, row in enumerate(reader, 2):
            node_type = str(row["entity_type"]).strip().upper()
            name = normalize_name(row["entity"])
            if node_type not in {"DRUG", "DISEASE"} or not name:
                raise ValueError(f"Invalid quality metadata key at {path}:{line_number}")
            try:
                words = float(row["description_words"] or 0)
            except ValueError as exc:
                raise ValueError(f"Invalid description_words at {path}:{line_number}") from exc
            metadata[(node_type, name)] = {
                "needs_review": _parse_bool(row["needs_review"]),
                "description_source": str(row["description_source"]),
                "description_words": words,
                "empty_description": _parse_bool(row["empty_description"]),
            }
    return metadata


def _reservoir_add(
    values: list[TypedNode],
    neighbor: TypedNode,
    seen_before: int,
    maximum: int,
    rng: random.Random,
) -> None:
    if len(values) < maximum:
        values.append(neighbor)
        return
    replacement = rng.randint(0, seen_before)
    if replacement < maximum:
        values[replacement] = neighbor


def sample_relation_neighbors(
    edges_dir: Path,
    seed_drugs: set[str],
    seed_diseases: set[str],
    max_neighbors: int,
    seed: int,
    relation_specs: Sequence[RelationSpec],
    leak_pairs_to_remove: set[tuple[str, str]] | None = None,
    allowed_nodes: set[TypedNode] | None = None,
) -> tuple[SampledNeighbors, dict[str, object]]:
    rng = random.Random(seed)
    sampled: SampledNeighbors = {
        spec.relation: defaultdict(list) for spec in relation_specs
    }
    scanned: dict[str, int] = {}
    eligible: dict[str, int] = {}
    excluded = leak_pairs_to_remove or set()

    for spec in relation_specs:
        path = edges_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing KG edge file: {path}")
        seen_by_center: dict[TypedNode, int] = defaultdict(int)
        scanned_rows = 0
        eligible_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"head_name", "relation", "tail_name"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Invalid edge header in {path}")
            for row in reader:
                scanned_rows += 1
                head = normalize_name(row["head_name"])
                tail = normalize_name(row["tail_name"])
                if spec in DIRECT_RELATIONS and (head, tail) in excluded:
                    continue
                candidates: list[tuple[TypedNode, TypedNode]] = []
                if spec.center_side == "head" and head in seed_drugs:
                    candidates.append((("DRUG", head), (spec.tail_type, tail)))
                elif spec.center_side == "tail" and tail in seed_diseases:
                    candidates.append((("DISEASE", tail), (spec.head_type, head)))
                elif spec.center_side == "both":
                    if head in seed_diseases:
                        candidates.append((("DISEASE", head), ("DISEASE", tail)))
                    if tail in seed_diseases:
                        candidates.append((("DISEASE", tail), ("DISEASE", head)))
                for center, neighbor in candidates:
                    if allowed_nodes is not None and (
                        center not in allowed_nodes or neighbor not in allowed_nodes
                    ):
                        continue
                    eligible_rows += 1
                    seen_before = seen_by_center[center]
                    _reservoir_add(
                        sampled[spec.relation][center],
                        neighbor,
                        seen_before,
                        max_neighbors,
                        rng,
                    )
                    seen_by_center[center] += 1
        scanned[spec.relation] = scanned_rows
        eligible[spec.relation] = eligible_rows

    retained = {
        relation: sum(len(neighbors) for neighbors in centers.values())
        for relation, centers in sampled.items()
    }
    return sampled, {
        "edge_rows_scanned": scanned,
        "eligible_seed_center_edges": eligible,
        "retained_neighbors": retained,
        "max_neighbors_per_relation": max_neighbors,
    }


def _collect_nodes(
    seed_drugs: Iterable[str],
    seed_diseases: Iterable[str],
    sampled_groups: Sequence[SampledNeighbors],
) -> list[TypedNode]:
    seed_nodes: list[TypedNode] = [
        *(("DRUG", name) for name in sorted(set(seed_drugs))),
        *(("DISEASE", name) for name in sorted(set(seed_diseases))),
    ]
    present = set(seed_nodes)
    extras: set[TypedNode] = set()
    for sampled in sampled_groups:
        for centers in sampled.values():
            for center, neighbors in centers.items():
                if center not in present:
                    extras.add(center)
                extras.update(neighbor for neighbor in neighbors if neighbor not in present)
    return seed_nodes + sorted(extras)


def _load_selected_kg_rows(index_path: Path, wanted: set[TypedNode]) -> dict[TypedNode, int]:
    rows: dict[TypedNode, int] = {}
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"id", "entity"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid KG entity index: {index_path}")
        for row in reader:
            raw_entity = row["entity"]
            if "::" not in raw_entity:
                continue
            prefix, name = raw_entity.split("::", 1)
            node = prefix.strip().upper(), normalize_name(name)
            if node in wanted:
                rows.setdefault(node, int(row["id"]))
    return rows


def build_feature_store(
    data_config: DataConfig,
    node_specs: Sequence[TypedNode],
) -> tuple[FeatureStore, dict[str, object]]:
    drug_text = _load_text_table(data_config.drug_index, data_config.drug_embeddings, "DRUG")
    disease_text = _load_text_table(data_config.disease_index, data_config.disease_embeddings, "DISEASE")
    if drug_text.embeddings.shape[1] != disease_text.embeddings.shape[1]:
        raise ValueError("Drug and disease text embedding dimensions differ")
    kg_embeddings = np.load(data_config.kg_entity_embeddings, mmap_mode="r", allow_pickle=False)
    if kg_embeddings.ndim != 2:
        raise ValueError("KG entity embeddings must be a 2D matrix")

    description_metadata = _load_quality_metadata(data_config.quality_metadata)
    merged_metadata: list[dict[str, object]] = []
    sources: list[str] = []
    for node_type, name in node_specs:
        base: dict[str, object] = {
            "needs_review": False,
            "description_source": "missing",
            "description_words": 0.0,
            "empty_description": True,
        }
        if node_type == "DRUG" and name in drug_text.primary_metadata:
            base.update(drug_text.primary_metadata[name])
        elif node_type == "DISEASE" and name in disease_text.primary_metadata:
            base.update(disease_text.primary_metadata[name])
        if (node_type, name) in description_metadata:
            base.update(description_metadata[(node_type, name)])
        source = str(base.get("description_source", "missing") or "missing")
        merged_metadata.append(base)
        sources.append(source)

    source_vocabulary = tuple(sorted(set(sources) | {"missing"}))
    source_to_index = {value: index for index, value in enumerate(source_vocabulary)}
    type_vocabulary = tuple(sorted({node_type for node_type, _ in node_specs}))
    type_to_index = {value: index for index, value in enumerate(type_vocabulary)}

    n_nodes = len(node_specs)
    text_dim = int(drug_text.embeddings.shape[1])
    kg_dim = int(kg_embeddings.shape[1])
    quality_dim = 3 + len(source_vocabulary) + len(type_vocabulary)
    text_matrix = np.zeros((n_nodes, text_dim), dtype=np.float32)
    kg_matrix = np.zeros((n_nodes, kg_dim), dtype=np.float32)
    quality = np.zeros((n_nodes, quality_dim), dtype=np.float32)
    text_available = np.zeros(n_nodes, dtype=np.float32)
    kg_available = np.zeros(n_nodes, dtype=np.float32)
    drug_to_node: dict[str, int] = {}
    disease_to_node: dict[str, int] = {}

    wanted = set(node_specs)
    kg_rows = _load_selected_kg_rows(data_config.kg_entity_index, wanted)
    for index, (node_type, name) in enumerate(node_specs):
        if node_type == "DRUG":
            drug_to_node[name] = index
            text_entry = drug_text.aliases.get(name)
            text_source = drug_text.embeddings
        elif node_type == "DISEASE":
            disease_to_node[name] = index
            text_entry = disease_text.aliases.get(name)
            text_source = disease_text.embeddings
        else:
            text_entry = None
            text_source = None

        if text_entry is not None and text_source is not None:
            text_matrix[index] = np.asarray(text_source[text_entry.row], dtype=np.float32)
            text_available[index] = 1.0
        kg_row = kg_rows.get((node_type, name))
        if kg_row is not None:
            if not 0 <= kg_row < kg_embeddings.shape[0]:
                raise ValueError(f"KG row out of bounds for {(node_type, name)}: {kg_row}")
            kg_matrix[index] = np.asarray(kg_embeddings[kg_row], dtype=np.float32)
            kg_available[index] = 1.0

        metadata = merged_metadata[index]
        source = str(metadata.get("description_source", "missing") or "missing")
        words = max(float(metadata.get("description_words", 0.0) or 0.0), 0.0)
        quality[index, 0] = float(bool(metadata.get("needs_review", False)))
        quality[index, 1] = min(math.log1p(words) / math.log(512.0), 1.0)
        quality[index, 2] = float(bool(metadata.get("empty_description", words <= 0.0)))
        quality[index, 3 + source_to_index[source]] = 1.0
        quality[index, 3 + len(source_vocabulary) + type_to_index[node_type]] = 1.0

    missing = np.stack((1.0 - text_available, 1.0 - kg_available), axis=1).astype(np.float32)
    features = FeatureStore(
        node_names=tuple(name for _, name in node_specs),
        node_types=tuple(node_type for node_type, _ in node_specs),
        text_embeddings=text_matrix,
        kg_embeddings=kg_matrix,
        quality_features=quality,
        missing_flags=missing,
        text_available=text_available,
        kg_available=kg_available,
        drug_to_node=drug_to_node,
        disease_to_node=disease_to_node,
        source_vocabulary=source_vocabulary,
        type_vocabulary=type_vocabulary,
    )
    stats = {
        "nodes": n_nodes,
        "text_dim": text_dim,
        "kg_dim": kg_dim,
        "quality_dim": quality_dim,
        "quality_metadata_entries": len(description_metadata),
        "text_available": int(text_available.sum()),
        "kg_available": int(kg_available.sum()),
        "both_embeddings_missing": int(((text_available == 0) & (kg_available == 0)).sum()),
        "node_type_counts": {
            node_type: int(sum(value == node_type for value in features.node_types))
            for node_type in NODE_TYPES
        },
    }
    return features, stats


def topology_edges_from_samples(
    sampled: SampledNeighbors,
    node_to_index: Mapping[TypedNode, int],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for relation in RELATION_NAMES:
        edges: set[tuple[int, int]] = set()
        for center, neighbors in sampled[relation].items():
            center_index = node_to_index[center]
            for neighbor in neighbors:
                neighbor_index = node_to_index[neighbor]
                edges.add((neighbor_index, center_index))
                edges.add((center_index, neighbor_index))
        ordered = sorted(edges)
        src = np.asarray([edge[0] for edge in ordered], dtype=np.int64)
        dst = np.asarray([edge[1] for edge in ordered], dtype=np.int64)
        if ordered:
            result[relation] = src, dst
            counts[relation] = len(ordered)
    return result, counts


def build_text_knn_edges(features: FeatureStore, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    edges: set[tuple[int, int]] = set()
    stats: dict[str, object] = {"k": k}

    for node_type, label in (("DRUG", "drug"), ("DISEASE", "disease")):
        node_ids = [
            index
            for index, value in enumerate(features.node_types)
            if value == node_type and features.text_available[index] > 0
        ]
        stats[f"{label}_nodes"] = len(node_ids)
        if len(node_ids) <= 1:
            stats[f"{label}_edges_before_dedup"] = 0
            continue
        matrix = features.text_embeddings[node_ids].astype(np.float32, copy=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        matrix[valid] /= norms[valid]
        matrix[~valid] = 0.0
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -np.inf)
        neighbors_per_node = min(k, len(node_ids) - 1)
        local_edges = 0
        for local_index, node_index in enumerate(node_ids):
            if not valid[local_index]:
                continue
            nearest = np.argpartition(-similarity[local_index], neighbors_per_node - 1)[:neighbors_per_node]
            nearest = nearest[np.argsort(-similarity[local_index, nearest])]
            for neighbor_local in nearest:
                if np.isfinite(similarity[local_index, neighbor_local]):
                    edges.add((node_ids[int(neighbor_local)], node_index))
                    local_edges += 1
        stats[f"{label}_edges_before_dedup"] = local_edges

    ordered = sorted(edges)
    src = np.asarray([edge[0] for edge in ordered], dtype=np.int64)
    dst = np.asarray([edge[1] for edge in ordered], dtype=np.int64)
    stats["directed_edges"] = len(ordered)
    return src, dst, stats


def prepare_graph(
    data_config: DataConfig,
    model_config: ModelConfig,
    seed_drugs: set[str],
    seed_diseases: set[str],
    seed: int,
    leak_pairs_to_remove: set[tuple[str, str]] | None = None,
) -> PreparedGraph:
    full_sampled, full_scan_stats = sample_relation_neighbors(
        data_config.kg_edges_dir,
        seed_drugs,
        seed_diseases,
        model_config.max_neighbors,
        seed,
        ALL_RELATIONS,
        leak_pairs_to_remove,
    )
    mechanism_sampled, mechanism_scan_stats = sample_relation_neighbors(
        data_config.kg_edges_dir,
        seed_drugs,
        seed_diseases,
        model_config.max_neighbors,
        seed,
        MECHANISM_RELATIONS,
        leak_pairs_to_remove,
    )
    node_specs = _collect_nodes(
        seed_drugs,
        seed_diseases,
        (full_sampled, mechanism_sampled),
    )
    features, feature_stats = build_feature_store(data_config, node_specs)
    node_to_index = {node: index for index, node in enumerate(node_specs)}
    topology_allowed_nodes = {
        node
        for index, node in enumerate(node_specs)
        if node[0] in {"DRUG", "DISEASE"} or features.kg_available[index] > 0
    }
    topology_sampled, topology_scan_stats = sample_relation_neighbors(
        data_config.kg_edges_dir,
        seed_drugs,
        seed_diseases,
        model_config.max_neighbors,
        seed,
        MECHANISM_RELATIONS,
        leak_pairs_to_remove,
        topology_allowed_nodes,
    )
    topology_edges, topology_edge_counts = topology_edges_from_samples(
        topology_sampled,
        node_to_index,
    )
    similarity_src, similarity_dst, similarity_stats = build_text_knn_edges(features, model_config.knn_k)
    graph = GraphBundle(
        relations=tuple(topology_edges),
        topology_edges=topology_edges,
        similarity_edges=(similarity_src, similarity_dst),
        stats={
            "full_relation_neighbor_sampling": full_scan_stats,
            "mechanism_neighbor_sampling": mechanism_scan_stats,
            "topology_neighbor_sampling": topology_scan_stats,
            "topology_directed_edges": topology_edge_counts,
            "similarity": similarity_stats,
        },
    )
    return PreparedGraph(
        features=features,
        graph=graph,
        audit={"features": feature_stats, "graph": graph.stats},
    )
