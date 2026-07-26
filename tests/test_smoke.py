from pathlib import Path

import torch

from redrgnn.data import (
    pair_entity_sets,
    prepare_supervised_pairs,
)
from redrgnn.graph import prepare_graph
from redrgnn.model import EvidenceDualRouteGNN
from redrgnn.trainer import (
    build_model_dimensions,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_model,
)


def test_end_to_end_train_and_checkpoint(tiny_inputs, tmp_path: Path):
    data, model_config, training = tiny_inputs
    set_seed(11, deterministic=True)
    pairs = prepare_supervised_pairs(
        data.positive_pairs,
        data.negative_pairs,
        data.test_pairs,
        data.holdout_pairs,
        data.kg_edges_dir,
        training.validation_fraction,
        11,
    )
    drugs, diseases = pair_entity_sets(pairs.train + pairs.validation + pairs.test)
    prepared = prepare_graph(data, model_config, drugs, diseases, 11)
    dimensions = build_model_dimensions(
        prepared.features,
        prepared.graph,
        hidden_dim=model_config.hidden_dim,
        topology_layers=model_config.topology_layers,
        similarity_layers=model_config.similarity_layers,
        dropout=model_config.dropout,
    )
    model = EvidenceDualRouteGNN(dimensions)
    result = train_model(
        model,
        pairs.train,
        pairs.validation,
        pairs.test,
        prepared.features,
        prepared.graph,
        training,
        seed=11,
        device=torch.device("cpu"),
        num_workers=0,
    )
    assert result.epochs_run >= 1
    assert 0.0 <= result.best_threshold <= 1.0
    assert result.test_metrics_at_0p5["n"] == 2
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, result, dimensions, training, 11)
    restored, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert payload["seed"] == 11
    assert set(restored.state_dict()) == set(model.state_dict())
