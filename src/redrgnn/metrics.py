from __future__ import annotations

from typing import Sequence

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    precision = true_positives / (np.arange(len(sorted_labels)) + 1)
    return float((precision * (sorted_labels == 1)).sum() / positives)


def best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(scores_array) == 0 or len(np.unique(scores_array)) == 1:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.quantile(scores_array, np.linspace(0.05, 0.95, 91)):
        predictions = scores_array >= threshold
        true_positive = int(((labels_array == 1) & predictions).sum())
        false_positive = int(((labels_array == 0) & predictions).sum())
        false_negative = int(((labels_array == 1) & ~predictions).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    predictions = scores_array >= threshold
    true_positive = int(((labels_array == 1) & predictions).sum())
    true_negative = int(((labels_array == 0) & ~predictions).sum())
    false_positive = int(((labels_array == 0) & predictions).sum())
    false_negative = int(((labels_array == 1) & ~predictions).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": int(len(labels_array)),
        "positives": int((labels_array == 1).sum()),
        "negatives": int((labels_array == 0).sum()),
        "AUROC": roc_auc(labels_array, scores_array),
        "AUPR": average_precision(labels_array, scores_array),
        "Accuracy": float((true_positive + true_negative) / max(len(labels_array), 1)),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "FPR": float(1.0 - specificity),
        "F1": float(f1),
        "Brier": float(np.mean((scores_array - labels_array) ** 2)),
        "threshold": float(threshold),
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
    }
