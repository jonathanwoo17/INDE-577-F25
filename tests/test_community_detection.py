import importlib.util
from pathlib import Path

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent

module_path = root / "src" / "rice_ml" / "unsupervised_learning" / "community_detection.py"
spec = importlib.util.spec_from_file_location("community_detection", module_path)
community_detection_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(community_detection_module)
community_detection = community_detection_module.community_detection

import pytest


def test_run_simple_graph_produces_stable_communities():
    vertices = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c")]
    detector = community_detection(vertices, edges, max_iters=20, epsilon=0.0)

    labels = detector.run(seed=123)

    # Nodes in the same connected component should share a label.
    assert labels["a"] == labels["b"] == labels["c"]
    # Isolated vertex keeps its own label.
    assert labels["d"] != labels["a"]

    # Compressed vector should place the connected component in one community
    # and the isolated vertex in another.
    assert detector.get_label_vector(compress=True) == [0, 0, 0, 1]


def test_get_label_vector_requires_run_first():
    detector = community_detection([1, 2], [(1, 2)], max_iters=5, epsilon=0.5)
    with pytest.raises(RuntimeError):
        detector.get_label_vector()


def test_validation_checks_duplicate_vertices_and_missing_edges():
    with pytest.raises(ValueError):
        community_detection([1, 1, 2], [(1, 2)], max_iters=5, epsilon=0.5)

    with pytest.raises(ValueError):
        community_detection([1, 2], [(1, 3)], max_iters=5, epsilon=0.5)


def test_empty_graph_returns_no_labels():
    detector = community_detection([], [], max_iters=10, epsilon=0.1)
    assert detector.run(seed=999) == {}
    assert detector.get_communities() == {}


