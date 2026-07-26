from redrgnn.data import prepare_supervised_pairs


def test_repun_split_removes_test_and_holdout(tiny_inputs):
    data, _, training = tiny_inputs
    prepared = prepare_supervised_pairs(
        data.positive_pairs,
        data.negative_pairs,
        data.test_pairs,
        data.holdout_pairs,
        data.kg_edges_dir,
        training.validation_fraction,
        11,
    )
    train_and_validation = {record.key for record in prepared.train + prepared.validation}
    test = {record.key for record in prepared.test}
    assert not train_and_validation & test
    assert ("d4", "x4") not in train_and_validation
    assert all(record.label in {0, 1} for record in prepared.train + prepared.validation)
    assert all(record.key != ("d2", "x2") for record in prepared.validation)
    assert all(record.key != ("d2", "x4") for record in prepared.validation)
