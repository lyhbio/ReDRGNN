from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .metrics import binary_metrics
from .model import dimensions_to_dict
from .trainer import (
    build_model_dimensions,
    load_checkpoint,
    make_tensors,
    predict_records,
    resolve_device,
    set_seed,
)
from .workflow import prepare_run, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one ReDRGNN checkpoint on RePUN-T.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config).with_overrides(device=args.device)
    config.validate_paths()
    device = resolve_device(config.runtime.device)
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    seed = int(checkpoint["seed"])
    set_seed(seed, config.runtime.deterministic)
    prepared = prepare_run(config, seed)
    expected_dimensions = build_model_dimensions(
        prepared.graph_data.features,
        prepared.graph_data.graph,
        hidden_dim=config.model.hidden_dim,
        topology_layers=config.model.topology_layers,
        similarity_layers=config.model.similarity_layers,
        dropout=config.model.dropout,
    )
    if dimensions_to_dict(model.dimensions) != dimensions_to_dict(expected_dimensions):
        raise ValueError("Checkpoint dimensions do not match the graph/features built from this config")
    nodes, graph = make_tensors(prepared.graph_data.features, prepared.graph_data.graph, device)
    prediction = predict_records(
        model,
        prepared.pairs.test,
        prepared.graph_data.features,
        nodes,
        graph,
        config.training.batch_size,
        device,
    )
    threshold = float(checkpoint["best_threshold"])
    result = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "seed": seed,
        "validation_threshold": threshold,
        "metrics_at_validation_threshold": binary_metrics(prediction.labels, prediction.scores, threshold),
        "metrics_at_0p5": binary_metrics(prediction.labels, prediction.scores, 0.5),
    }
    if args.output is not None:
        write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
