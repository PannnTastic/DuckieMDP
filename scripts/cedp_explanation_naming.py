"""Explanation-derived naming: learn a transparent rule set that maps the
explanation signature (counterfactual + outcome + verification, no raw state)
to a primitive name. A shallow decision tree yields explicit if/then rules the
user can read, cross-check, and edit -- and proves the name is recoverable from
the explanation itself.

Usage: cedp_explanation_naming.py "dir[:solvers]" ...
"""

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

SOURCES = []
for arg in sys.argv[1:]:
    if ":" in arg:
        p, s = arg.split(":", 1)
        SOURCES.append((Path(p), set(s.split(","))))
    else:
        SOURCES.append((Path(arg), None))

# Curated, human-legible explanation-only features. Every one is computed by the
# M1--M13 machinery (why / what-happens / verification); none is a raw input.
CF = ["stop_distance_flip", "stop_satisfied_flip", "duck_risk_flip",
      "curvature_flip", "lateral_flip", "heading_flip", "speed_flip"]
OUT = ["factual_full_stops", "factual_brake_ratio",
       "factual_max_abs_lateral_error_m", "factual_minimum_duck_clearance_m",
       "delta_stop_violations", "delta_minimum_duck_clearance_m",
       "delta_max_abs_lateral_error_m", "foil_lane_departure"]
VER = ["stop_applicable", "stop_pass", "pedestrian_applicable", "pedestrian_pass",
       "curvature_applicable", "curvature_pass",
       "lane_symmetry_applicable", "lane_symmetry_pass"]
FEATURES = (["cf__" + c for c in CF] + ["out__" + c for c in OUT]
            + ["ver__" + v for v in VER] + ["cf__minimum_flip_distance"])


def family(s):
    if s.get("stop_present") and s.get("stop_distance") is not None:
        return "StopSign"
    if s.get("duck_present") or s.get("duck_active"):
        return "PedestrianCrossing"
    k = float(s.get("curvature") or 0.0)
    if s.get("curvature_class") in ("curve_left", "curve_right") or abs(k) >= 0.8:
        return "CurveNegotiation"
    return "LaneKeeping"


def feats(inst):
    cp = inst["decision_evidence"]["counterfactual_profile"]
    oe = inst["outcome_evidence"]["physical_profile"]
    ve = inst["verification_evidence"]["verification_profile"]
    row = []
    for c in CF:
        row.append(float(cp.get(c, 0.0)))
    for o in OUT:
        row.append(float(oe.get(o, 0.0)))
    for v in VER:
        row.append(float(ve.get(v, 0.0)))
    row.append(float(cp.get("minimum_flip_distance", 0.0)))
    return row


def main():
    anchors = {}
    for d, keep in SOURCES:
        for line in open(d / "full_decision_anchors.jsonl", encoding="utf-8"):
            a = json.loads(line)
            if keep and a["solver"] not in keep:
                continue
            anchors[(a["solver"], a["episode_id"], a["step_index"])] = a
    X, y = [], []
    for d, keep in SOURCES:
        for p in glob.glob(str(d / "instance_shards" / "*.json")):
            x = json.load(open(p, encoding="utf-8"))
            if x["status"] != "CERTIFIED" or (keep and x["solver"] not in keep):
                continue
            a = anchors.get((x["solver"], x["episode_id"], x["step_index"]))
            if not a:
                continue
            X.append(feats(x))
            y.append(family(a["state"]))
    X, y = np.array(X), np.array(y)
    print("dataset:", len(y), "| label dist:", dict(Counter(y)))
    print("features (all explanation-only):", len(FEATURES))

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40,
                                  class_weight="balanced", random_state=0)
    sc = cross_val_score(tree, X, y, cv=5, scoring="balanced_accuracy")
    print("5-fold balanced accuracy (explanation-only rules -> name): %.3f +- %.3f"
          % (sc.mean(), sc.std()))
    tree.fit(X, y)
    print()
    print("=== EXPLANATION-DERIVED NAMING RULES (cross-check & edit these) ===")
    print(export_text(tree, feature_names=FEATURES, max_depth=4))


if __name__ == "__main__":
    main()
