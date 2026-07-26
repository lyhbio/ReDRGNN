from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


TopologyTensors = Mapping[str, tuple[torch.Tensor, torch.Tensor]]
SimilarityTensors = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ModelDimensions:
    text_dim: int
    kg_dim: int
    quality_dim: int
    hidden_dim: int
    relations: tuple[str, ...]
    topology_layers: int
    similarity_layers: int
    dropout: float


class FeatureGatedInitializer(nn.Module):
    def __init__(self, text_dim: int, kg_dim: int, quality_dim: int, hidden_dim: int):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.kg_projection = nn.Linear(kg_dim, hidden_dim)
        self.quality_projection = nn.Sequential(
            nn.Linear(quality_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(quality_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        kg: torch.Tensor,
        quality: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        text_hidden = self.text_projection(text)
        kg_hidden = self.kg_projection(kg)
        quality_and_missing = torch.cat((quality, missing), dim=-1)
        gate = torch.sigmoid(self.modality_gate(quality_and_missing))
        text_missing = missing[:, 0:1]
        kg_missing = missing[:, 1:2]
        gate = torch.where(text_missing > 0.5, torch.zeros_like(gate), gate)
        gate = torch.where(kg_missing > 0.5, torch.ones_like(gate), gate)
        context = self.quality_projection(quality_and_missing)
        return self.normalization(gate * text_hidden + (1.0 - gate) * kg_hidden + context)


class RelationMeanLayer(nn.Module):
    def __init__(self, hidden_dim: int, relations: tuple[str, ...], dropout: float):
        super().__init__()
        self.relations = relations
        self.relation_linears = nn.ModuleDict(
            {relation: nn.Linear(hidden_dim, hidden_dim, bias=False) for relation in relations}
        )
        self.relation_logits = nn.Parameter(torch.zeros(len(relations)))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, edges: TopologyTensors) -> torch.Tensor:
        aggregated_relations = torch.zeros_like(hidden)
        relation_weights = torch.softmax(self.relation_logits, dim=0)
        for relation_index, relation in enumerate(self.relations):
            src, dst = edges[relation]
            if src.numel() == 0:
                continue
            messages = self.relation_linears[relation](hidden[src])
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros(hidden.size(0), device=hidden.device, dtype=hidden.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=hidden.dtype))
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(1)
            aggregated_relations = aggregated_relations + relation_weights[relation_index] * aggregated
        return self.normalization(hidden + self.dropout(aggregated_relations))


class TopologyEncoder(nn.Module):
    def __init__(self, hidden_dim: int, relations: tuple[str, ...], layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            RelationMeanLayer(hidden_dim, relations, dropout) for _ in range(max(1, layers))
        )

    def forward(self, hidden: torch.Tensor, edges: TopologyTensors) -> torch.Tensor:
        output = hidden
        for layer in self.layers:
            output = layer(output, edges)
        return output


class SimilarityMeanLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0:
            return hidden
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, dst, hidden[src])
        degree = torch.zeros(hidden.size(0), device=hidden.device, dtype=hidden.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=hidden.dtype))
        neighbor_mean = aggregated / degree.clamp_min(1.0).unsqueeze(1)
        update = self.update(torch.cat((hidden, neighbor_mean), dim=-1))
        return self.normalization(hidden + update)


class SimilarityEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            SimilarityMeanLayer(hidden_dim, dropout) for _ in range(max(1, layers))
        )

    def forward(self, hidden: torch.Tensor, edges: SimilarityTensors) -> torch.Tensor:
        src, dst = edges
        output = hidden
        for layer in self.layers:
            output = layer(output, src, dst)
        return output


class PairDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        drug: torch.Tensor,
        disease: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = torch.cat(
            (
                drug,
                disease,
                drug * disease,
                torch.abs(drug - disease),
                scalar_features,
            ),
            dim=-1,
        )
        return self.network(pair_features).squeeze(-1)


class EvidenceDualRouteGNN(nn.Module):
    def __init__(self, dimensions: ModelDimensions):
        super().__init__()
        self.dimensions = dimensions
        self.initializer = FeatureGatedInitializer(
            dimensions.text_dim,
            dimensions.kg_dim,
            dimensions.quality_dim,
            dimensions.hidden_dim,
        )
        self.topology_encoder = TopologyEncoder(
            dimensions.hidden_dim,
            dimensions.relations,
            dimensions.topology_layers,
            dimensions.dropout,
        )
        self.similarity_encoder = SimilarityEncoder(
            dimensions.hidden_dim,
            dimensions.similarity_layers,
            dimensions.dropout,
        )
        self.route_gate = nn.Sequential(
            nn.Linear(dimensions.hidden_dim * 2 + 2, dimensions.hidden_dim),
            nn.ReLU(),
            nn.Linear(dimensions.hidden_dim, 2),
        )
        self.decoder = PairDecoder(dimensions.hidden_dim, dimensions.dropout)
        self.drug_projection = nn.Sequential(
            nn.Linear(dimensions.hidden_dim, dimensions.hidden_dim),
            nn.ReLU(),
            nn.Linear(dimensions.hidden_dim, dimensions.hidden_dim),
        )
        self.disease_projection = nn.Sequential(
            nn.Linear(dimensions.hidden_dim, dimensions.hidden_dim),
            nn.ReLU(),
            nn.Linear(dimensions.hidden_dim, dimensions.hidden_dim),
        )

    def encode_nodes(
        self,
        text: torch.Tensor,
        kg: torch.Tensor,
        quality: torch.Tensor,
        missing: torch.Tensor,
        topology_edges: TopologyTensors,
        similarity_edges: SimilarityTensors,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self.initializer(text, kg, quality, missing)
        topology = self.topology_encoder(initial, topology_edges)
        similarity = self.similarity_encoder(initial, similarity_edges)
        route_logits = self.route_gate(torch.cat((topology, similarity, missing), dim=-1))
        # A KG-missing node receives a fixed prior toward the text-similarity route.
        route_logits = route_logits.clone()
        route_logits[:, 0] = route_logits[:, 0] - 1.5 * missing[:, 1]
        route_logits[:, 1] = route_logits[:, 1] + 1.5 * missing[:, 1]
        route_weights = torch.softmax(route_logits, dim=-1)
        fused = route_weights[:, 0:1] * topology + route_weights[:, 1:2] * similarity
        return fused, topology, similarity, route_weights

    def decode_pairs(
        self,
        node_embeddings: torch.Tensor,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            node_embeddings[drug_indices],
            node_embeddings[disease_indices],
            scalar_features,
        )

    def forward(
        self,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        text: torch.Tensor,
        kg: torch.Tensor,
        quality: torch.Tensor,
        missing: torch.Tensor,
        topology_edges: TopologyTensors,
        similarity_edges: SimilarityTensors,
        scalar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embeddings, topology, similarity, route_weights = self.encode_nodes(
            text,
            kg,
            quality,
            missing,
            topology_edges,
            similarity_edges,
        )
        drug_embeddings = node_embeddings[drug_indices]
        disease_embeddings = node_embeddings[disease_indices]
        logits = self.decoder(drug_embeddings, disease_embeddings, scalar_features)
        return {
            "logits": logits,
            "drug_embeddings": drug_embeddings,
            "disease_embeddings": disease_embeddings,
            "node_embeddings": node_embeddings,
            "topology_embeddings": topology,
            "similarity_embeddings": similarity,
            "route_weights": route_weights,
        }


def dimensions_to_dict(dimensions: ModelDimensions) -> dict[str, object]:
    return {
        "text_dim": dimensions.text_dim,
        "kg_dim": dimensions.kg_dim,
        "quality_dim": dimensions.quality_dim,
        "hidden_dim": dimensions.hidden_dim,
        "relations": list(dimensions.relations),
        "topology_layers": dimensions.topology_layers,
        "similarity_layers": dimensions.similarity_layers,
        "dropout": dimensions.dropout,
    }


def dimensions_from_dict(raw: Mapping[str, object]) -> ModelDimensions:
    return ModelDimensions(
        text_dim=int(raw["text_dim"]),
        kg_dim=int(raw["kg_dim"]),
        quality_dim=int(raw["quality_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        relations=tuple(str(value) for value in raw["relations"]),
        topology_layers=int(raw["topology_layers"]),
        similarity_layers=int(raw["similarity_layers"]),
        dropout=float(raw["dropout"]),
    )

