"""Solver-aware decision-tree surrogates for M10.

The models in this module summarize frozen policies. They are never presented
as the original policy unless exact fidelity has been demonstrated on the
explicitly named domain.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


RULE_EXTRACTION_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TreeComplexity:
    depth: int
    leaves: int
    nodes: int
    features_used: Tuple[str, ...]
    schema_version: str = RULE_EXTRACTION_SCHEMA_VERSION


@dataclass(frozen=True)
class ClassificationMetrics:
    samples: int
    fidelity: Optional[float]
    balanced_fidelity: Optional[float]
    macro_f1: Optional[float]
    per_class_recall: Mapping[str, Optional[float]]
    schema_version: str = RULE_EXTRACTION_SCHEMA_VERSION


@dataclass(frozen=True)
class RegressionMetrics:
    samples: int
    mae_v: Optional[float]
    mae_omega: Optional[float]
    schema_version: str = RULE_EXTRACTION_SCHEMA_VERSION


def library_manifest():
    return {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "random_state": 0,
        "role": "post_hoc_surrogate_only",
    }


def fit_classifier(
    features,
    labels,
    max_depth=None,
    min_samples_leaf=1,
    class_weight=None,
    random_state=0,
):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("classification training data must be non-empty 2-D")
    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        class_weight=class_weight,
        random_state=int(random_state),
    )
    model.fit(x, y)
    return model


def fit_action_regressor(
    features,
    actions,
    max_depth=14,
    min_samples_leaf=3,
    random_state=0,
):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(actions, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 2 or len(x) != len(y):
        raise ValueError("action regression requires X[n,d] and y[n,2]")
    model = DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
    )
    model.fit(x, y)
    return model


def tree_complexity(model, feature_names):
    names = tuple(str(name) for name in feature_names)
    used = sorted({
        names[int(index)]
        for index in model.tree_.feature
        if int(index) >= 0
    })
    return TreeComplexity(
        depth=int(model.tree_.max_depth),
        leaves=int(model.tree_.n_leaves),
        nodes=int(model.tree_.node_count),
        features_used=tuple(used),
    )


def classification_metrics(labels, predictions):
    y = np.asarray(labels)
    predicted = np.asarray(predictions)
    if y.shape != predicted.shape:
        raise ValueError("classification labels and predictions differ in shape")
    if len(y) == 0:
        return ClassificationMetrics(0, None, None, None, {})
    classes = np.unique(np.concatenate([y, predicted]))
    recall = {}
    for value in classes:
        mask = y == value
        recall[str(value)] = (
            None if not np.any(mask)
            else float(np.mean(predicted[mask] == y[mask]))
        )
    return ClassificationMetrics(
        samples=len(y),
        fidelity=float(accuracy_score(y, predicted)),
        balanced_fidelity=float(balanced_accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
        per_class_recall=recall,
    )


def classification_metrics_by_stratum(labels, predictions, strata):
    y = np.asarray(labels)
    predicted = np.asarray(predictions)
    result = {"all": classification_metrics(y, predicted)}
    for name, selector in strata.items():
        mask = np.asarray(selector, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError("stratum %s has wrong shape" % name)
        result[str(name)] = classification_metrics(y[mask], predicted[mask])
    return result


def regression_metrics(actions, predictions):
    y = np.asarray(actions, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if y.shape != predicted.shape or (y.ndim == 2 and y.shape[1] != 2):
        raise ValueError("regression labels and predictions differ in shape")
    if len(y) == 0:
        return RegressionMetrics(0, None, None)
    errors = mean_absolute_error(y, predicted, multioutput="raw_values")
    return RegressionMetrics(len(y), float(errors[0]), float(errors[1]))


def regression_metrics_by_stratum(actions, predictions, strata):
    y = np.asarray(actions, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    result = {"all": regression_metrics(y, predicted)}
    for name, selector in strata.items():
        mask = np.asarray(selector, dtype=bool)
        if mask.shape != (len(y),):
            raise ValueError("stratum %s has wrong shape" % name)
        result[str(name)] = regression_metrics(y[mask], predicted[mask])
    return result


def extract_leaf_rules(model, feature_names):
    """Return deterministic root-to-leaf rule records."""
    feature_names = tuple(str(name) for name in feature_names)
    tree = model.tree_
    classifier = isinstance(model, DecisionTreeClassifier)
    records = []

    def walk(node, conditions):
        feature_index = int(tree.feature[node])
        if feature_index < 0:
            if classifier:
                class_index = int(np.argmax(tree.value[node][0]))
                prediction = model.classes_[class_index]
                prediction = prediction.item() if hasattr(prediction, "item") else prediction
                distribution = {
                    str(label): float(value)
                    for label, value in zip(model.classes_, tree.value[node][0])
                }
            else:
                values = np.asarray(tree.value[node]).reshape(-1)
                prediction = [float(value) for value in values]
                distribution = None
            records.append({
                "leaf_id": int(node),
                "conditions": list(conditions),
                "prediction": prediction,
                "samples": int(tree.n_node_samples[node]),
                "weighted_samples": float(tree.weighted_n_node_samples[node]),
                "impurity": float(tree.impurity[node]),
                "class_distribution": distribution,
            })
            return
        feature = feature_names[feature_index]
        threshold = float(tree.threshold[node])
        walk(
            int(tree.children_left[node]),
            conditions + ({"feature": feature, "operator": "<=", "threshold": threshold},),
        )
        walk(
            int(tree.children_right[node]),
            conditions + ({"feature": feature, "operator": ">", "threshold": threshold},),
        )

    walk(0, tuple())
    return tuple(records)


def rules_by_prediction(rules):
    grouped = {}
    for rule in rules:
        key = str(rule["prediction"])
        grouped.setdefault(key, []).append(rule)
    return grouped


def export_rule_text(model, feature_names, decimals=4):
    return export_text(
        model,
        feature_names=[str(name) for name in feature_names],
        decimals=int(decimals),
        show_weights=True,
    )


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    return path


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
