from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    positive_pairs: Path
    negative_pairs: Path
    test_pairs: Path
    holdout_pairs: Path | None
    drug_embeddings: Path
    disease_embeddings: Path
    kg_entity_embeddings: Path
    drug_index: Path
    disease_index: Path
    quality_metadata: Path
    kg_entity_index: Path
    kg_edges_dir: Path

    def required_training_paths(self) -> tuple[Path, ...]:
        paths = (
            self.positive_pairs,
            self.negative_pairs,
            self.test_pairs,
            self.drug_embeddings,
            self.disease_embeddings,
            self.kg_entity_embeddings,
            self.drug_index,
            self.disease_index,
            self.quality_metadata,
            self.kg_entity_index,
            self.kg_edges_dir,
        )
        return paths + ((self.holdout_pairs,) if self.holdout_pairs else ())


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    topology_layers: int = 1
    similarity_layers: int = 1
    knn_k: int = 7
    max_neighbors: int = 30
    dropout: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (20260613, 20260614, 20260615, 20260616, 20260617)
    validation_fraction: float = 0.2
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_weight: float = 1.0
    negative_weight: float = 2.0
    contrastive_weight: float = 0.05
    temperature: float = 0.2
    minimum_epochs: int = 5
    patience: int = 5
    gradient_clip_norm: float = 5.0


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = True
    output_dir: Path = Path("runs")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    source_path: Path

    def with_overrides(
        self,
        *,
        device: str | None = None,
        output_dir: Path | None = None,
        seeds: tuple[int, ...] | None = None,
    ) -> "ExperimentConfig":
        runtime = self.runtime
        training = self.training
        if device is not None:
            runtime = replace(runtime, device=device)
        if output_dir is not None:
            runtime = replace(runtime, output_dir=output_dir.expanduser().resolve())
        if seeds is not None:
            training = replace(training, seeds=seeds)
        return replace(self, runtime=runtime, training=training)

    def validate_paths(self) -> None:
        required = list(self.data.required_training_paths())
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Required input paths do not exist:\n{formatted}")


def _resolve_path(base: Path, value: str, *, optional: bool = False) -> Path | None:
    value = value.strip()
    if optional and not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML section [{name}]")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    base = source.parent
    data_raw = _require_section(raw, "data")
    model_raw = _require_section(raw, "model")
    train_raw = _require_section(raw, "training")
    runtime_raw = _require_section(raw, "runtime")

    data = DataConfig(
        positive_pairs=_resolve_path(base, str(data_raw["positive_pairs"])),
        negative_pairs=_resolve_path(base, str(data_raw["negative_pairs"])),
        test_pairs=_resolve_path(base, str(data_raw["test_pairs"])),
        holdout_pairs=_resolve_path(base, str(data_raw.get("holdout_pairs", "")), optional=True),
        drug_embeddings=_resolve_path(base, str(data_raw["drug_embeddings"])),
        disease_embeddings=_resolve_path(base, str(data_raw["disease_embeddings"])),
        kg_entity_embeddings=_resolve_path(base, str(data_raw["kg_entity_embeddings"])),
        drug_index=_resolve_path(base, str(data_raw["drug_index"])),
        disease_index=_resolve_path(base, str(data_raw["disease_index"])),
        quality_metadata=_resolve_path(base, str(data_raw["quality_metadata"])),
        kg_entity_index=_resolve_path(base, str(data_raw["kg_entity_index"])),
        kg_edges_dir=_resolve_path(base, str(data_raw["kg_edges_dir"])),
    )
    model = ModelConfig(**model_raw)
    training = TrainingConfig(
        seeds=tuple(int(seed) for seed in train_raw.get("seeds", [])),
        **{key: value for key, value in train_raw.items() if key != "seeds"},
    )
    runtime = RuntimeConfig(
        output_dir=_resolve_path(base, str(runtime_raw.get("output_dir", "../runs"))),
        **{key: value for key, value in runtime_raw.items() if key != "output_dir"},
    )
    if not training.seeds:
        raise ValueError("training.seeds must contain at least one seed")
    if not 0.0 < training.validation_fraction < 1.0:
        raise ValueError("training.validation_fraction must be between 0 and 1")
    if model.hidden_dim <= 0 or model.knn_k <= 0 or model.max_neighbors <= 0:
        raise ValueError("Model dimensions and graph neighborhood sizes must be positive")
    return ExperimentConfig(data=data, model=model, training=training, runtime=runtime, source_path=source)
