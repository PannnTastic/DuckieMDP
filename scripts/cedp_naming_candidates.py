"""Evidence-grounded naming candidates + M1--M13 evidence display.

Reads a gated C-EDDP output directory and, per certified explanation, proposes a
sharpened driving-primitive name from the decision context and the counterfactual
trigger (M5/M6), then attaches the metamorphic persistence (M7) and the
factual-vs-foil outcome. Prints a JSON summary consumed by the report artifact.
"""

import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/explanations/cedp_v2_gated")

STOP_SPEED = 0.05
YIELD_SPEED = 0.05
CURVE_KAPPA = 0.8
LANE_D = 0.055
HEAD_PHI = 0.14


def sharp_name(state, action, cp):
    """Human-readable driving-primitive name from context + counterfactual."""
    d = float(state["d"])
    phi = float(state["phi"])
    v = float(action["v_cmd"])
    kappa = float(state.get("curvature") or 0.0)
    cclass = state.get("curvature_class")
    minflip = cp.get("minimum_flip_concept")

    # 1. Stop sign — a present, unresolved stop dominates the behaviour.
    if state.get("stop_present") and state.get("stop_distance") is not None:
        if state.get("stop_satisfied"):
            return "ResumeAfterStop" if v > STOP_SPEED else "StopHold", "stop"
        if v < STOP_SPEED:
            return "StopHold", "stop"
        return "StopApproach", "stop"

    # 2. Pedestrian — a Duckie in the forward corridor.
    duck_long = state.get("duck_longitudinal")
    if state.get("duck_present") and (
        state.get("duck_active")
        or (duck_long is not None and abs(float(duck_long)) < 0.8)
    ):
        return ("PedestrianYield" if v < YIELD_SPEED else "PedestrianApproach"), "duck"

    # 3. Curve — the road bends (categorical class, kappa, or curvature trigger).
    if (
        cclass in ("curve_left", "curve_right")
        or abs(kappa) >= CURVE_KAPPA
        or minflip == "curvature"
    ):
        direction = "Left" if (kappa > 0 or cclass == "curve_left") else "Right"
        return "CurveNegotiation" + direction, "curvature"

    # 4. Lane correction — heading phi and lateral d disagree with the lane.
    if abs(d) >= LANE_D or abs(phi) >= HEAD_PHI or minflip in ("lateral", "heading"):
        # Steer back toward the centre line: positive d/phi means steer right.
        direction = "Right" if (d + phi) > 0 else "Left"
        return "LaneCorrection" + direction, "lane"

    # 5. Cruise — stable straight travel, no dominant correction.
    return "Cruise", "cruise"


def main():
    anchors = {}
    for line in open(RUN / "full_decision_anchors.jsonl", encoding="utf-8"):
        a = json.loads(line)
        anchors[(a["solver"], a["episode_id"], a["step_index"])] = a

    records = []
    for path in glob.glob(str(RUN / "instance_shards" / "*.json")):
        d = json.load(open(path, encoding="utf-8"))
        if d["status"] != "CERTIFIED":
            continue
        key = (d["solver"], d["episode_id"], d["step_index"])
        anchor = anchors.get(key)
        if anchor is None:
            continue
        cp = d["decision_evidence"]["counterfactual_profile"]
        ve = d["verification_evidence"]["verification_profile"]
        oe = d["outcome_evidence"]["physical_profile"]
        name, family = sharp_name(anchor["state"], anchor["selected_action"], cp)
        records.append({
            "name": name,
            "family": family,
            "solver": d["solver"],
            "seed": d["seed"],
            "step": d["step_index"],
            "state": {
                k: anchor["state"].get(k) for k in
                ("d", "phi", "v", "curvature", "curvature_class",
                 "stop_present", "stop_distance", "stop_satisfied",
                 "duck_present", "duck_active", "duck_longitudinal")
            },
            "action": {
                "v_cmd": anchor["selected_action"]["v_cmd"],
                "omega_cmd": anchor["selected_action"]["omega_cmd"],
            },
            # M5/M6: which state feature, minimally changed, flips the action.
            "m5_m6_counterfactual": {
                "minimum_flip_concept": cp.get("minimum_flip_concept"),
                "minimum_flip_distance": round(float(cp.get("minimum_flip_distance", 0)), 4),
                "attempts": cp.get("attempts"),
                "valid_attempts": cp.get("valid_attempts"),
                "flips": {
                    c: bool(cp.get(c + "_flip"))
                    for c in ("lateral", "heading", "curvature",
                              "speed", "stop_distance", "duck_risk")
                },
            },
            # M7: metamorphic relations that hold under semantic perturbation.
            "m7_metamorphic": {
                rel: (
                    "PASS" if ve.get(rel + "_pass")
                    else "FAIL" if ve.get(rel + "_fail")
                    else "n/a"
                )
                for rel in ("stop", "pedestrian", "curvature", "lane_symmetry")
            },
            "outcome": {
                "delta_max_abs_lateral_error_m": round(float(oe.get("delta_max_abs_lateral_error_m", 0)), 3),
                "delta_max_abs_heading_error_rad": round(float(oe.get("delta_max_abs_heading_error_rad", 0)), 3),
                "delta_stop_violations": round(float(oe.get("delta_stop_violations", 0)), 2),
                "delta_minimum_duck_clearance_m": round(float(oe.get("delta_minimum_duck_clearance_m", 0)), 3),
                "factual_full_stops": round(float(oe.get("factual_full_stops", 0)), 1),
            },
        })

    by_name = defaultdict(list)
    for r in records:
        by_name[r["name"]].append(r)

    summary = {"total_certified": len(records), "distribution": {}, "primitives": []}
    for name in sorted(by_name, key=lambda n: -len(by_name[n])):
        group = by_name[name]
        solvers = Counter(r["solver"] for r in group)
        summary["distribution"][name] = {"count": len(group), "solvers": dict(solvers)}
        # representatives: prefer the smallest flip distance (sharpest trigger).
        reps = sorted(group, key=lambda r: r["m5_m6_counterfactual"]["minimum_flip_distance"])[:2]
        summary["primitives"].append({
            "name": name,
            "family": group[0]["family"],
            "count": len(group),
            "solvers": dict(solvers),
            "representatives": reps,
        })

    (RUN / "naming_candidates.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"total": len(records),
                      "distribution": {k: v["count"] for k, v in summary["distribution"].items()},
                      "solver_split": {k: v["solvers"] for k, v in summary["distribution"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
