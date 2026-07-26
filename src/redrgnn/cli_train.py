from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import load_config
from .data import write_pairs
from .model import EvidenceDualRouteGNN
from .trainer import (
    build_model_dimensions,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_model,
)
from .workflow import aggregate_seed_metrics, input_manifest, prepare_run, write_json


def _parse_seeds(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final ReDRGNN model on RePUN-P/RePUN-N.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="Override runtime.device, for example cpu or cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, build splits/features/graphs, and stop before optimization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config = config.with_overrides(
        device=args.device,
        output_dir=args.output_dir,
        seeds=_parse_seeds(args.seeds),
    )
    config.validate_paths()
    output_dir = config.runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.source_path, output_dir / "resolved_source_config.toml")
    write_json(output_dir / "input_manifest.json", input_manifest(config))
    device = resolve_device(config.runtime.device)
    print(f"device={device}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    if config.data.holdout_pairs is None:
        print(
            "warning: no additional holdout_pairs file is configured; "
            "RePUN-T leakage is removed, but the original Authority70 exclusion is not reproduced.",
            flush=True,
        )

    seed_rows: list[dict[str, object]] = []
    for seed in config.training.seeds:
        print(f"[seed {seed}] preparing clean split and dual-route graph", flush=True)
        set_seed(seed, config.runtime.deterministic)
        prepared = prepare_run(config, seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_json(seed_dir / "data_graph_audit.json", prepared.audit)
        write_pairs(seed_dir / "train_pairs.tsv", prepared.pairs.train)
        write_pairs(seed_dir / "validation_pairs.tsv", prepared.pairs.validation)
        write_pairs(seed_dir / "test_pairs.tsv", prepared.pairs.test)
        if args.dry_run:
            print(f"[seed {seed}] dry-run complete", flush=True)
            continue

        dimensions = build_model_dimensions(
            prepared.graph_data.features,
            prepared.graph_data.graph,
            hidden_dim=config.model.hidden_dim,
            topology_layers=config.model.topology_layers,
            similarity_layers=config.model.similarity_layers,
            dropout=config.model.dropout,
        )
        model = EvidenceDualRouteGNN(dimensions).to(device)
        print(f"[seed {seed}] training", flush=True)
        result = train_model(
            model,
            prepared.pairs.train,
            prepared.pairs.validation,
            prepared.pairs.test,
            prepared.graph_data.features,
            prepared.graph_data.graph,
            config.training,
            seed=seed,
            device=device,
            num_workers=config.runtime.num_workers,
        )
        checkpoint_path = seed_dir / "model.pt"
        save_checkpoint(checkpoint_path, result, dimensions, config.training, seed)
        row: dict[str, object] = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "epochs_run": result.epochs_run,
            "seconds": result.seconds,
            "best_validation_AUPR": result.best_validation_aupr,
            "best_threshold": result.best_threshold,
            "validation_metrics": result.validation_metrics,
            "test_metrics_at_validation_threshold": result.test_metrics_at_validation_threshold,
            "test_metrics_at_0p5": result.test_metrics_at_0p5,
            "history": result.history,
        }
        seed_rows.append(row)
        write_json(seed_dir / "metrics.json", row)
        print(
            f"[seed {seed}] test AUROC={result.test_metrics_at_0p5['AUROC']:.4f} "
            f"AUPR={result.test_metrics_at_0p5['AUPR']:.4f}",
            flush=True,
        )
        if device.type == "cuda":
            del model
            import torch

            torch.cuda.empty_cache()

    if args.dry_run:
        write_json(output_dir / "dry_run.json", {"seeds": list(config.training.seeds), "status": "ok"})
        return 0
    summary = {
        "model": "dual_route_gnn_cleanval_no_direct_dd_topology_no_relation_score",
        "loss": "positive_weight_1_negative_weight_2_weighted_bce_plus_0.05_infonce",
        "per_seed": seed_rows,
        "aggregate_at_0p5": aggregate_seed_metrics(seed_rows),
    }
    write_json(output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
