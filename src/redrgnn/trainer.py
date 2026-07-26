from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .data import PairRecord
from .graph import FeatureStore, GraphBundle
from .losses import final_training_loss
from .metrics import best_f1_threshold, binary_metrics
from .model import (
    EvidenceDualRouteGNN,
    ModelDimensions,
    dimensions_from_dict,
    dimensions_to_dict,
)


@dataclass(frozen=True)
class NodeTensors:
    text: torch.Tensor
    kg: torch.Tensor
    quality: torch.Tensor
    missing: torch.Tensor
    normalized_text: torch.Tensor

    def scalar_features(
        self,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
    ) -> torch.Tensor:
        text_cosine = (
            self.normalized_text[drug_indices] * self.normalized_text[disease_indices]
        ).sum(dim=1, keepdim=True)
        # The final model disables direct DistMult TREATS/CONTRAINDICATES scores.
        return torch.cat((text_cosine, torch.zeros_like(text_cosine).expand(-1, 3)), dim=1)


@dataclass(frozen=True)
class GraphTensors:
    topology: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    similarity: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PredictionResult:
    labels: np.ndarray
    logits: np.ndarray
    scores: np.ndarray


@dataclass
class TrainingResult:
    model_state: dict[str, torch.Tensor]
    best_validation_aupr: float
    best_threshold: float
    epochs_run: int
    seconds: float
    history: list[dict[str, float]]
    validation_metrics: dict[str, float | int]
    test_metrics_at_validation_threshold: dict[str, float | int]
    test_metrics_at_0p5: dict[str, float | int]


class PairDataset(Dataset):
    def __init__(self, records: Sequence[PairRecord], features: FeatureStore):
        self.drug_indices = np.asarray(
            [features.drug_to_node[record.drug] for record in records],
            dtype=np.int64,
        )
        self.disease_indices = np.asarray(
            [features.disease_to_node[record.disease] for record in records],
            dtype=np.int64,
        )
        self.labels = np.asarray([record.label for record in records], dtype=np.float32)
        if np.any((self.labels != 0) & (self.labels != 1)):
            raise ValueError("Training/evaluation datasets accept only labels 0 and 1")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.drug_indices[index], dtype=torch.long),
            torch.tensor(self.disease_indices[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_tensors(
    features: FeatureStore,
    graph: GraphBundle,
    device: torch.device,
) -> tuple[NodeTensors, GraphTensors]:
    text = torch.from_numpy(features.text_embeddings).to(device)
    kg = torch.from_numpy(features.kg_embeddings).to(device)
    quality = torch.from_numpy(features.quality_features).to(device)
    missing = torch.from_numpy(features.missing_flags).to(device)
    normalized_text = F.normalize(text, dim=1)
    topology = {
        relation: (
            torch.from_numpy(graph.topology_edges[relation][0]).to(device),
            torch.from_numpy(graph.topology_edges[relation][1]).to(device),
        )
        for relation in graph.relations
    }
    similarity = (
        torch.from_numpy(graph.similarity_edges[0]).to(device),
        torch.from_numpy(graph.similarity_edges[1]).to(device),
    )
    return (
        NodeTensors(text, kg, quality, missing, normalized_text),
        GraphTensors(topology, similarity),
    )


def build_model_dimensions(
    features: FeatureStore,
    graph: GraphBundle,
    *,
    hidden_dim: int,
    topology_layers: int,
    similarity_layers: int,
    dropout: float,
) -> ModelDimensions:
    return ModelDimensions(
        text_dim=features.text_dim,
        kg_dim=features.kg_dim,
        quality_dim=features.quality_dim,
        hidden_dim=hidden_dim,
        relations=graph.relations,
        topology_layers=topology_layers,
        similarity_layers=similarity_layers,
        dropout=dropout,
    )


def _loader(
    records: Sequence[PairRecord],
    features: FeatureStore,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PairDataset(records, features),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def encode_nodes_for_inference(
    model: EvidenceDualRouteGNN,
    nodes: NodeTensors,
    graph: GraphTensors,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        embeddings, _, _, _ = model.encode_nodes(
            nodes.text,
            nodes.kg,
            nodes.quality,
            nodes.missing,
            graph.topology,
            graph.similarity,
        )
    return embeddings


def score_index_arrays(
    model: EvidenceDualRouteGNN,
    node_embeddings: torch.Tensor,
    nodes: NodeTensors,
    drug_indices: np.ndarray,
    disease_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(drug_indices) != len(disease_indices):
        raise ValueError("Drug and disease index arrays have different lengths")
    output = np.empty(len(drug_indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(drug_indices), batch_size):
            end = min(start + batch_size, len(drug_indices))
            drug = torch.from_numpy(drug_indices[start:end]).to(device=device, dtype=torch.long)
            disease = torch.from_numpy(disease_indices[start:end]).to(device=device, dtype=torch.long)
            scalar = nodes.scalar_features(drug, disease)
            logits = model.decode_pairs(node_embeddings, drug, disease, scalar)
            output[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return output


def predict_records(
    model: EvidenceDualRouteGNN,
    records: Sequence[PairRecord],
    features: FeatureStore,
    nodes: NodeTensors,
    graph: GraphTensors,
    batch_size: int,
    device: torch.device,
) -> PredictionResult:
    dataset = PairDataset(records, features)
    node_embeddings = encode_nodes_for_inference(model, nodes, graph)
    scores = score_index_arrays(
        model,
        node_embeddings,
        nodes,
        dataset.drug_indices,
        dataset.disease_indices,
        batch_size,
        device,
    )
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return PredictionResult(dataset.labels.astype(np.int64), logits, scores.astype(np.float64))


def train_model(
    model: EvidenceDualRouteGNN,
    train_records: Sequence[PairRecord],
    validation_records: Sequence[PairRecord],
    test_records: Sequence[PairRecord],
    features: FeatureStore,
    graph_bundle: GraphBundle,
    training: TrainingConfig,
    *,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> TrainingResult:
    nodes, graph = make_tensors(features, graph_bundle, device)
    train_loader = _loader(
        train_records,
        features,
        training.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_aupr = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, training.epochs + 1):
        model.train()
        total_loss = 0.0
        total_classification = 0.0
        total_contrastive = 0.0
        seen = 0
        for drug_indices, disease_indices, labels in train_loader:
            drug_indices = drug_indices.to(device)
            disease_indices = disease_indices.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scalar = nodes.scalar_features(drug_indices, disease_indices)
            outputs = model(
                drug_indices,
                disease_indices,
                nodes.text,
                nodes.kg,
                nodes.quality,
                nodes.missing,
                graph.topology,
                graph.similarity,
                scalar,
            )
            loss, components = final_training_loss(
                model,
                outputs,
                labels,
                positive_weight=training.positive_weight,
                negative_weight=training.negative_weight,
                contrastive_weight=training.contrastive_weight,
                temperature=training.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            batch_size = len(labels)
            seen += batch_size
            total_loss += float(loss.item()) * batch_size
            total_classification += float(components["classification"].item()) * batch_size
            total_contrastive += float(components["contrastive"].item()) * batch_size

        validation_prediction = predict_records(
            model,
            validation_records,
            features,
            nodes,
            graph,
            training.batch_size,
            device,
        )
        threshold = best_f1_threshold(validation_prediction.labels, validation_prediction.scores)
        validation_metrics = binary_metrics(
            validation_prediction.labels,
            validation_prediction.scores,
            threshold,
        )
        validation_aupr = float(validation_metrics["AUPR"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(seen, 1),
                "classification_loss": total_classification / max(seen, 1),
                "contrastive_loss": total_contrastive / max(seen, 1),
                "validation_AUPR": validation_aupr,
                "validation_AUROC": float(validation_metrics["AUROC"]),
                "validation_F1": float(validation_metrics["F1"]),
                "threshold": float(threshold),
            }
        )
        if validation_aupr > best_validation_aupr + 1e-6:
            best_validation_aupr = validation_aupr
            best_threshold = float(threshold)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= training.minimum_epochs and epochs_without_improvement >= training.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_prediction = predict_records(
        model,
        validation_records,
        features,
        nodes,
        graph,
        training.batch_size,
        device,
    )
    test_prediction = predict_records(
        model,
        test_records,
        features,
        nodes,
        graph,
        training.batch_size,
        device,
    )
    return TrainingResult(
        model_state=best_state,
        best_validation_aupr=best_validation_aupr,
        best_threshold=best_threshold,
        epochs_run=len(history),
        seconds=time.time() - start_time,
        history=history,
        validation_metrics=binary_metrics(
            validation_prediction.labels,
            validation_prediction.scores,
            best_threshold,
        ),
        test_metrics_at_validation_threshold=binary_metrics(
            test_prediction.labels,
            test_prediction.scores,
            best_threshold,
        ),
        test_metrics_at_0p5=binary_metrics(test_prediction.labels, test_prediction.scores, 0.5),
    )


def save_checkpoint(
    path: Path,
    result: TrainingResult,
    dimensions: ModelDimensions,
    training: TrainingConfig,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": result.model_state,
            "dimensions": dimensions_to_dict(dimensions),
            "seed": seed,
            "best_threshold": result.best_threshold,
            "best_validation_aupr": result.best_validation_aupr,
            "training": asdict(training),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[EvidenceDualRouteGNN, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    dimensions = dimensions_from_dict(payload["dimensions"])
    model = EvidenceDualRouteGNN(dimensions)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
