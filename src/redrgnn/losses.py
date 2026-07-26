from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_infonce(
    drug_projection: torch.Tensor,
    disease_projection: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive_mask = labels > 0.5
    if int(positive_mask.sum().item()) == 0:
        return torch.zeros((), device=drug_projection.device, dtype=drug_projection.dtype)
    similarities = drug_projection @ disease_projection.t() / temperature
    row_log_probability = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    transposed = similarities.t()
    column_log_probability = transposed - torch.logsumexp(transposed, dim=1, keepdim=True)
    indices = torch.arange(labels.size(0), device=labels.device)
    positive_indices = indices[positive_mask]
    drug_to_disease = -row_log_probability[positive_indices, positive_indices].mean()
    disease_to_drug = -column_log_probability[positive_indices, positive_indices].mean()
    return 0.5 * (drug_to_disease + disease_to_drug)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("Weighted BCE accepts only labels 0 and 1")
    weights = torch.where(
        labels > 0.5,
        torch.as_tensor(positive_weight, device=labels.device, dtype=labels.dtype),
        torch.as_tensor(negative_weight, device=labels.device, dtype=labels.dtype),
    )
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def final_training_loss(
    model,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    positive_weight: float,
    negative_weight: float,
    contrastive_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = weighted_bce_with_logits(
        outputs["logits"],
        labels,
        positive_weight,
        negative_weight,
    )
    projected_drugs = F.normalize(model.drug_projection(outputs["drug_embeddings"]), dim=1)
    projected_diseases = F.normalize(model.disease_projection(outputs["disease_embeddings"]), dim=1)
    contrastive = supervised_infonce(projected_drugs, projected_diseases, labels, temperature)
    total = classification + contrastive_weight * contrastive
    return total, {"classification": classification, "contrastive": contrastive}

