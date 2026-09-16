"""Artificial arrays only; no private data or recorded benchmark outputs."""
import json
import numpy as np
import pytest
from sklearn.svm import SVC
from image_classification.data import validate, load_data, channel_means
from image_classification.evaluation import checked_folds, evaluate, metrics
from image_classification.cnn import split_groups


def sample():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(40) % 2, 2).astype(int)
    groups = np.repeat([f"object_{i}" for i in range(40)], 2)
    images = rng.normal(size=(80, 4, 4, 3)).astype(np.float32)
    return images, labels, groups


def test_data_roundtrip(tmp_path):
    images, labels, groups = sample()
    path = tmp_path / "input.npz"
    np.savez(path, images=images, labels=labels, groups=groups)
    actual, y, g = load_data(path)
    np.testing.assert_array_equal(actual, images)
    np.testing.assert_array_equal(y, labels)
    np.testing.assert_array_equal(g, groups)
    np.testing.assert_allclose(channel_means(actual), images.mean((1, 2), dtype=np.float64))


def test_invalid_inputs():
    images, labels, groups = sample()
    bad = labels.copy()
    bad[0] = 1 - bad[0]
    with pytest.raises(ValueError, match="consistent"):
        validate(images, bad, groups)
    images[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate(images, labels, groups)


def test_object_arrays_rejected(tmp_path):
    images, labels, groups = sample()
    path = tmp_path / "input.npz"
    np.savez(path, images=images, labels=labels, groups=groups.astype(object))
    with pytest.raises(ValueError):
        load_data(path)


def test_nested_groups_and_evaluation():
    images, y, g = sample()
    X = channel_means(images)
    for train, test in checked_folds(X, y, g, 3, 42):
        assert not set(g[train]) & set(g[test])
        for a, b in checked_folds(X[train], y[train], g[train], 2, 7):
            assert not set(g[train][a]) & set(g[train][b])
            assert not (set(g[train][a]) | set(g[train][b])) & set(g[test])
    result = evaluate(X, y, g, grid=[{"clf": [SVC(kernel="linear")], "clf__C": [1]}])
    assert len(result["folds"]) == 3
    json.dumps(result, allow_nan=False)


def test_insufficient_groups_fail():
    with pytest.raises(ValueError, match="Insufficient"):
        checked_folds(np.ones((4, 3)), np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"]), 2, 42)


def test_group_vote_tie():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), np.array(["a", "a", "b", "b"]))
    assert result["group"]["accuracy"] == 1.0
    assert result["patch"]["accuracy"] == 0.75


def test_cnn_split():
    _, y, groups = sample()
    train, val, test = split_groups(y, groups, 42)
    assert len(set(train) | set(val) | set(test)) == len(y)
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[val]) & set(groups[test])


def test_cli(tmp_path):
    from image_classification.__main__ import main
    images, labels, groups = sample()
    source, output = tmp_path / "input.npz", tmp_path / "local.json"
    np.savez(source, images=images, labels=labels, groups=groups)
    main(["--data", str(source), "--output", str(output)])
    assert len(json.loads(output.read_text())["folds"]) == 3
    with pytest.raises(SystemExit):
        main(["--data", str(source), "--output", str(output)])


def test_cli_rejects_project_output():
    from pathlib import Path
    from image_classification.__main__ import main
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit):
        main(["--data", "unused.npz", "--output", str(root / "result.json")])


@pytest.mark.parametrize("kind,shape", [("cnn1d", (6, 1)), ("cnn2d", (8, 8, 3))])
def test_optional_cnn(kind, shape):
    pytest.importorskip("tensorflow")
    from image_classification.cnn import build_model
    model = build_model(shape, kind)
    X = np.zeros((2, *shape), dtype=np.float32)
    loss = model.train_on_batch(X, np.array([0, 1]))
    assert np.isfinite(loss)
    assert model(X, training=False).shape == (2, 1)
