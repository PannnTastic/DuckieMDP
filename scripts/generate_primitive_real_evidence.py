"""Generate real-data provenance cards for four explanation-derived primitives."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


PRIMITIVES = (
    "LaneKeeping",
    "CurveNegotiation",
    "StopCompliance",
    "PedestrianYield",
)


def _primitive(state):
    if state.get("stop_present") and state.get("stop_distance") is not None:
        return "StopCompliance"
    if state.get("duck_present") or state.get("duck_active"):
        return "PedestrianYield"
    curvature = abs(float(state.get("curvature") or 0.0))
    if (
        state.get("curvature_class") in ("curve_left", "curve_right")
        or curvature >= 0.8
    ):
        return "CurveNegotiation"
    return "LaneKeeping"


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load(root):
    anchors = {}
    for row in _read_jsonl(root / "full_decision_anchors.jsonl"):
        key = (row["solver"], row["episode_id"], int(row["step_index"]))
        anchors[key] = row
    instances = {}
    for name in (
        "certified_explanation_instances.jsonl",
        "abstained_explanations.jsonl",
    ):
        path = root / name
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            key = (row["solver"], row["episode_id"], int(row["step_index"]))
            instances[key] = row
    return anchors, instances


def _runs(anchors, instances):
    by_episode = defaultdict(list)
    for key, anchor in anchors.items():
        if key not in instances:
            continue
        solver, episode, step = key
        by_episode[(solver, episode)].append((step, anchor))
    runs = []
    for (solver, episode), rows in sorted(by_episode.items()):
        rows.sort(key=lambda item: item[0])
        start = 0
        while start < len(rows):
            primitive = _primitive(rows[start][1]["state"])
            end = start
            while (
                end + 1 < len(rows)
                and rows[end + 1][0] == rows[end][0] + 1
                and _primitive(rows[end + 1][1]["state"]) == primitive
            ):
                end += 1
            chunk = rows[start : end + 1]
            runs.append({
                "primitive": primitive,
                "solver": solver,
                "episode_id": episode,
                "seed": int(chunk[0][1]["seed"]),
                "start_step": int(chunk[0][0]),
                "end_step": int(chunk[-1][0]),
                "keys": [
                    (solver, episode, int(step)) for step, _ in chunk
                ],
                "states": [anchor["state"] for _, anchor in chunk],
            })
            start = end + 1
    return runs


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return ("%%.%df" % digits) % float(value)


def _why(profile):
    concepts = (
        "lateral",
        "heading",
        "speed",
        "curvature",
        "stop_distance",
        "stop_satisfied",
        "duck_risk",
    )
    flips = [
        concept for concept in concepts
        if bool(profile.get(concept + "_flip", False))
    ]
    minimum = profile.get("minimum_flip_concept")
    distance = profile.get("minimum_flip_distance")
    return (
        "The selected action changes under %s interventions; the nearest "
        "recorded decision boundary is %s at distance %s."
        % (
            ", ".join(flips) if flips else "no tested one-feature",
            minimum or "n/a",
            _fmt(distance, 4),
        )
    )


def _what_if(primitive, physical):
    if primitive == "StopCompliance":
        return (
            "Selected vs foil: full stops %s vs %s; stop violations %s vs %s; "
            "brake ratio %s vs %s."
            % (
                _fmt(physical.get("factual_full_stops"), 0),
                _fmt(physical.get("foil_full_stops"), 0),
                _fmt(physical.get("factual_stop_violations"), 0),
                _fmt(physical.get("foil_stop_violations"), 0),
                _fmt(physical.get("factual_brake_ratio")),
                _fmt(physical.get("foil_brake_ratio")),
            )
        )
    if primitive == "PedestrianYield":
        available = bool(
            physical.get("factual_minimum_duck_clearance_available", False)
        )
        clearance = (
            _fmt(physical.get("factual_minimum_duck_clearance_m"))
            if available else "n/a"
        )
        foil_clearance = (
            _fmt(physical.get("foil_minimum_duck_clearance_m"))
            if physical.get("foil_minimum_duck_clearance_available", False)
            else "n/a"
        )
        return (
            "Selected vs foil: minimum Duckie clearance %s m vs %s m; "
            "Duckie collision %s vs %s; brake ratio %s vs %s."
            % (
                clearance,
                foil_clearance,
                _fmt(physical.get("factual_duck_collision")),
                _fmt(physical.get("foil_duck_collision")),
                _fmt(physical.get("factual_brake_ratio")),
                _fmt(physical.get("foil_brake_ratio")),
            )
        )
    return (
        "Selected vs foil: max |d| %s m vs %s m; max |phi| %s rad vs "
        "%s rad; lane departure %s vs %s; steering jerk %s vs %s."
        % (
            _fmt(physical.get("factual_max_abs_lateral_error_m")),
            _fmt(physical.get("foil_max_abs_lateral_error_m")),
            _fmt(physical.get("factual_max_abs_heading_error_rad")),
            _fmt(physical.get("foil_max_abs_heading_error_rad")),
            _fmt(physical.get("factual_lane_departure")),
            _fmt(physical.get("foil_lane_departure")),
            _fmt(physical.get("factual_cumulative_steering_jerk")),
            _fmt(physical.get("foil_cumulative_steering_jerk")),
        )
    )


def _verification(profile):
    values = []
    for relation in ("stop", "pedestrian", "curvature", "lane_symmetry"):
        if not profile.get(relation + "_applicable", False):
            continue
        status = (
            "PASS" if profile.get(relation + "_pass", False)
            else "FAIL" if profile.get(relation + "_fail", False)
            else "ABSTAIN"
        )
        values.append("%s=%s" % (relation, status))
    return "; ".join(values) if values else "No relation applicable at this anchor."


def _temporal(primitive, states):
    speeds = [float(state["v"]) for state in states]
    d_values = [float(state["d"]) for state in states]
    phi_values = [float(state["phi"]) for state in states]
    phrases = {
        "LaneKeeping": "deviation -> corrective steering -> recenter",
        "CurveNegotiation": "curve detected -> decelerate -> steer -> recover",
        "StopCompliance": "approach -> decelerate -> stop -> hold -> resume",
        "PedestrianYield": "detect crossing -> yield/decelerate -> hold -> resume",
    }
    return (
        "%s; %d decisions; v %.3f -> min %.3f -> %.3f; "
        "d %.3f -> %.3f; phi %.3f -> %.3f."
        % (
            phrases[primitive],
            len(states),
            speeds[0],
            min(speeds),
            speeds[-1],
            d_values[0],
            d_values[-1],
            phi_values[0],
            phi_values[-1],
        )
    )


def _representative(primitive, runs):
    candidates = [run for run in runs if run["primitive"] == primitive]
    sac = [run for run in candidates if run["solver"] == "sac"]
    pool = sac or candidates
    return max(pool, key=lambda run: len(run["keys"]))


def _build(root):
    anchors, instances = _load(root)
    runs = _runs(anchors, instances)
    cards = {}
    for primitive in PRIMITIVES:
        group = [run for run in runs if run["primitive"] == primitive]
        representative = _representative(primitive, runs)
        middle_key = representative["keys"][len(representative["keys"]) // 2]
        instance = instances[middle_key]
        decision = instance["decision_evidence"]["counterfactual_profile"]
        physical = instance["outcome_evidence"]["physical_profile"]
        verification = instance["verification_evidence"]["verification_profile"]
        cards[primitive] = {
            "primitive": primitive,
            "instances": len(group),
            "decisions": sum(len(run["keys"]) for run in group),
            "solver_instances": dict(sorted(Counter(
                run["solver"] for run in group
            ).items())),
            "mean_length": mean(len(run["keys"]) for run in group),
            "representative": {
                "instance_id": instance["instance_id"],
                "solver": representative["solver"],
                "seed": representative["seed"],
                "episode_id": representative["episode_id"],
                "start_step": representative["start_step"],
                "middle_step": middle_key[2],
                "end_step": representative["end_step"],
            },
            "why": _why(decision),
            "what_if": _what_if(primitive, physical),
            "verification": _verification(verification),
            "temporal": _temporal(primitive, representative["states"]),
            "raw": {
                "counterfactual_profile": decision,
                "physical_profile": physical,
                "verification_profile": verification,
            },
        }
    return {
        "source": str(root),
        "method": (
            "Four macro families grouped from contiguous full-trajectory "
            "M1-M13 explanation instances."
        ),
        "primitives": cards,
        "total_instances": sum(card["instances"] for card in cards.values()),
        "total_decisions": sum(card["decisions"] for card in cards.values()),
    }


def _markdown(payload):
    lines = [
        "# Real Data Behind the Explanation-Derived Driving Primitives",
        "",
        "This document is generated from the frozen four-policy artefacts in",
        "`runs/explanations/cedp_v2_4policy`. The four names summarize",
        "contiguous temporal M1–M13 explanations; they are not assigned directly",
        "from one isolated `(state, action)` pair.",
        "",
        "## Pipeline",
        "",
        "```text",
        "real rollout state/action",
        "  -> Why: state counterfactual decision boundary",
        "  -> What-if: paired factual/foil simulator outcomes",
        "  -> Verification: metamorphic relation results",
        "  -> Temporal: contiguous explanation arc",
        "  -> human-readable driving primitive",
        "```",
        "",
        "## Real distribution",
        "",
        "| Primitive | Temporal instances | Decisions | Solver coverage |",
        "| --- | ---: | ---: | --- |",
    ]
    for primitive in PRIMITIVES:
        card = payload["primitives"][primitive]
        solvers = ", ".join(
            "%s=%d" % item for item in card["solver_instances"].items()
        )
        lines.append(
            "| %s | %d | %d | %s |"
            % (primitive, card["instances"], card["decisions"], solvers)
        )
    lines.extend([
        "",
        "## Evidence cards",
        "",
    ])
    for primitive in PRIMITIVES:
        card = payload["primitives"][primitive]
        source = card["representative"]
        lines.extend([
            "### " + primitive,
            "",
            "**Real source:** `%s`, seed `%s`, episode `%s`, decisions `%d–%d`, "
            "representative explanation `%s` at decision `%d`."
            % (
                source["solver"],
                source["seed"],
                source["episode_id"],
                source["start_step"],
                source["end_step"],
                source["instance_id"],
                source["middle_step"],
            ),
            "",
            "- **Why:** " + card["why"],
            "- **What-if:** " + card["what_if"],
            "- **Verification:** " + card["verification"],
            "- **Temporal arc:** " + card["temporal"],
            "",
        ])
    lines.extend([
        "## Interpretation boundary",
        "",
        "The primitive carries all three explanation pillars. A verification",
        "field may contain PASS, FAIL, ABSTAIN, or no applicable relation. In",
        "this descriptive result, `Verification` means that the measured result",
        "is preserved—not that every relation must pass.",
        "",
        "The machine-readable values, including raw counterfactual, physical,",
        "and verification profiles, are stored next to this document in",
        "`runs/explanations/cedp_v2_4policy/primitive_real_evidence.json`.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/explanations/cedp_v2_4policy"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "runs/explanations/cedp_v2_4policy/primitive_real_evidence.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/explanation_derived_primitive_real_data.md"),
    )
    args = parser.parse_args()
    payload = _build(args.input_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        _markdown(payload) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "json": str(args.json_output),
        "markdown": str(args.markdown_output),
        "primitives": {
            name: {
                "instances": card["instances"],
                "decisions": card["decisions"],
                "representative": card["representative"]["instance_id"],
            }
            for name, card in payload["primitives"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
