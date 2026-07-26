import numpy as np
import torch

from redrgnn.data import pair_entity_sets, prepare_supervised_pairs
from redrgnn.graph import prepare_graph
from redrgnn.losses import weighted_bce_with_logits
from redrgnn.model import EvidenceDualRouteGNN
from redrgnn.trainer import build_model_dimensions, make_tensors


def test_graph_excludes_direct_answer_relations_and_model_shapes(tiny_inputs):
    data, model_config, training = tiny_inputs
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
    assert "TREATS" not in prepared.graph.relations
    assert "CONTRAINDICATES" not in prepared.graph.relations
    assert prepared.features.quality_dim == 9
    d1 = prepared.features.drug_to_node["d1"]
    assert prepared.features.quality_features[d1, 0] == 1.0
    dimensions = build_model_dimensions(
        prepared.features,
        prepared.graph,
        hidden_dim=model_config.hidden_dim,
        topology_layers=model_config.topology_layers,
        similarity_layers=model_config.similarity_layers,
        dropout=model_config.dropout,
    )
    model = EvidenceDualRouteGNN(dimensions)
    nodes, graph = make_tensors(prepared.features, prepared.graph, torch.device("cpu"))
    encoded, topology, similarity, weights = model.encode_nodes(
        nodes.text,
        nodes.kg,
        nodes.quality,
        nodes.missing,
        graph.topology,
        graph.similarity,
    )
    assert encoded.shape == (len(prepared.features.node_names), model_config.hidden_dim)
    assert topology.shape == encoded.shape
    assert similarity.shape == encoded.shape
    assert weights.shape == (len(prepared.features.node_names), 2)
    np.testing.assert_allclose(weights.detach().numpy().sum(axis=1), 1.0, atol=1e-6)


def test_weighted_bce_rejects_unknown_label():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, -1.0])
    try:
        weighted_bce_with_logits(logits, labels, 1.0, 2.0)
    except ValueError as error:
        assert "labels 0 and 1" in str(error)
    else:
        raise AssertionError("Unknown label was accepted by BCE")
