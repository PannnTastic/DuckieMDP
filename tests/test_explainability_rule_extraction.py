import numpy as np
import pytest

from src.explainability.rule_extraction import (
    classification_metrics,
    classification_metrics_by_stratum,
    export_rule_text,
    extract_leaf_rules,
    file_sha256,
    fit_action_regressor,
    fit_classifier,
    library_manifest,
    regression_metrics,
    rules_by_prediction,
    save_model,
    tree_complexity,
)


def test_exact_classifier_and_rule_export(tmp_path):
    features = np.asarray([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [2, 0], [2, 1],
    ], dtype=float)
    labels = np.asarray(["brake", "brake", "left", "right", "left", "right"])
    model = fit_classifier(features, labels, max_depth=None, min_samples_leaf=1)
    predicted = model.predict(features)
    metrics = classification_metrics(labels, predicted)
    assert metrics.fidelity == pytest.approx(1.0)
    complexity = tree_complexity(model, ("risk", "side"))
    assert complexity.leaves >= 3
    assert set(complexity.features_used) == {"risk", "side"}
    rules = extract_leaf_rules(model, ("risk", "side"))
    assert len(rules) == complexity.leaves
    assert sum(rule["samples"] for rule in rules) == len(features)
    assert "brake" in rules_by_prediction(rules)
    assert "risk" in export_rule_text(model, ("risk", "side"))
    path = save_model(model, tmp_path / "classifier.joblib")
    assert len(file_sha256(path)) == 64


def test_classification_metrics_are_stratified():
    labels = np.asarray([0, 0, 1, 1])
    predicted = np.asarray([0, 1, 1, 1])
    reports = classification_metrics_by_stratum(
        labels,
        predicted,
        {
            "supported": np.asarray([True, False, True, False]),
            "empty": np.zeros(4, dtype=bool),
        },
    )
    assert reports["all"].fidelity == pytest.approx(0.75)
    assert reports["supported"].fidelity == pytest.approx(1.0)
    assert reports["empty"].fidelity is None


def test_multioutput_regression_reports_both_action_errors():
    features = np.arange(20, dtype=float).reshape(-1, 1)
    actions = np.column_stack((0.02 * features[:, 0], -0.1 * features[:, 0]))
    model = fit_action_regressor(
        features, actions, max_depth=None, min_samples_leaf=1
    )
    predicted = model.predict(features)
    metrics = regression_metrics(actions, predicted)
    assert metrics.mae_v == pytest.approx(0.0)
    assert metrics.mae_omega == pytest.approx(0.0)
    rules = extract_leaf_rules(model, ("state",))
    assert isinstance(rules[0]["prediction"], list)
    assert len(rules[0]["prediction"]) == 2


def test_invalid_training_shapes_are_rejected():
    with pytest.raises(ValueError):
        fit_classifier(np.zeros((0, 2)), np.zeros((0,)))
    with pytest.raises(ValueError):
        fit_action_regressor(np.zeros((3, 2)), np.zeros((3, 1)))


def test_library_manifest_marks_post_hoc_role():
    manifest = library_manifest()
    assert manifest["scikit_learn"] == "1.6.1"
    assert manifest["random_state"] == 0
    assert manifest["role"] == "post_hoc_surrogate_only"
