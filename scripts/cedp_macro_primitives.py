"""Macro driving primitives: full temporal arcs, not per-decision phases.

Contiguous certified decisions of the same behaviour family are merged into one
macro-primitive instance spanning the whole arc — a stop sign is the single
approach -> brake -> hold -> resume event, a curve is see -> slow -> steer ->
recover, and so on. Each macro is named from how its state evolves across the
arc. Reads one or more C-EDDP run directories (so all four policies combine).

Usage: cedp_macro_primitives.py OUT.json DIR1 [DIR2 ...]
"""

import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
# Each source is "dir" or "dir:solver1,solver2" to keep only those solvers
# (so the ungated SAC in a shared dir does not collide with the gated one).
SOURCES = []
for arg in sys.argv[2:]:
    if ":" in arg:
        path, solvers = arg.split(":", 1)
        SOURCES.append((Path(path), set(solvers.split(","))))
    else:
        SOURCES.append((Path(arg), None))

CURVE_KAPPA = 0.8
LANE_ERR = 0.05
CONVERGE = 0.01
OSC_CHANGES = 3
CRUISE_ERR = 0.05


def macro_family(s):
    if s.get("stop_present") and s.get("stop_distance") is not None:
        return "stop"
    if s.get("duck_present") or s.get("duck_active"):
        return "duck"
    kappa = float(s.get("curvature") or 0.0)
    if s.get("curvature_class") in ("curve_left", "curve_right") or abs(kappa) >= CURVE_KAPPA:
        return "curve"
    return "lane"


def sign_changes(xs):
    s = [1 if x > 1e-3 else -1 if x < -1e-3 else 0 for x in xs]
    s = [x for x in s if x]
    return sum(1 for a, b in zip(s, s[1:]) if a != b)


def name_macro(family, states, actions):
    d = [float(s["d"]) for s in states]
    phi = [float(s["phi"]) for s in states]
    v = [float(s["v"]) for s in states]
    omega = [float(a["omega_cmd"]) for a in actions]
    if family == "stop":
        return "StopSign", "stop"
    if family == "duck":
        return "PedestrianCrossing", "duck"
    if family == "curve":
        left = sum(1 for s in states if s.get("curvature_class") == "curve_left")
        right = sum(1 for s in states if s.get("curvature_class") == "curve_right")
        if left == right:
            left = statistics.mean([float(s.get("curvature") or 0.0) for s in states]) >= 0
        else:
            left = left >= right
        return "CurveNegotiation" + ("Left" if left else "Right"), "curvature"
    # lane family -> cruise / lane-correction / oscillatory, from the arc
    dabs0, dabsN = abs(d[0]), abs(d[-1])
    pabs0, pabsN = abs(phi[0]), abs(phi[-1])
    converging = (dabsN < dabs0 - CONVERGE) or (pabsN < pabs0 - CONVERGE)
    if sign_changes(phi) >= OSC_CHANGES and sign_changes(omega) >= OSC_CHANGES and not converging:
        return "OscillatorySteering", "lane"
    if max(abs(x) for x in d) < CRUISE_ERR and max(abs(x) for x in phi) < CRUISE_ERR and sign_changes(phi) < OSC_CHANGES:
        return "Cruise", "cruise"
    direction = "Left" if statistics.mean(omega) > 0 else "Right"
    return "LaneCorrection" + direction, "lane"


def load_dir(d, keep):
    anchors = {}
    for line in open(d / "full_decision_anchors.jsonl", encoding="utf-8"):
        a = json.loads(line)
        if keep and a["solver"] not in keep:
            continue
        anchors[(a["solver"], a["episode_id"], a["step_index"])] = a
    certified = {}
    for path in glob.glob(str(d / "instance_shards" / "*.json")):
        it = json.load(open(path, encoding="utf-8"))
        if it["status"] != "CERTIFIED":
            continue
        if keep and it["solver"] not in keep:
            continue
        certified[(it["solver"], it["episode_id"], it["step_index"])] = it
    return anchors, certified


def main():
    anchors, certified = {}, {}
    for d, keep in SOURCES:
        a, c = load_dir(d, keep)
        anchors.update(a)
        certified.update(c)

    # per (solver, episode) ordered certified decisions
    by_episode = defaultdict(list)
    for key in certified:
        solver, episode, step = key
        by_episode[(solver, episode)].append(step)

    macros = []
    for (solver, episode), steps in by_episode.items():
        steps.sort()
        # families per certified step
        seq = []
        for st in steps:
            a = anchors.get((solver, episode, st))
            if a is None:
                continue
            seq.append((st, macro_family(a["state"]), a))
        # group contiguous same-family (contiguity by consecutive certified steps)
        i = 0
        while i < len(seq):
            fam = seq[i][1]
            j = i
            while j + 1 < len(seq) and seq[j + 1][1] == fam and seq[j + 1][0] == seq[j][0] + 1:
                j += 1
            run = seq[i:j + 1]
            states = [r[2]["state"] for r in run]
            actions = [r[2]["selected_action"] for r in run]
            name, family = name_macro(fam, states, actions)
            v = [float(s["v"]) for s in states]
            d = [float(s["d"]) for s in states]
            phi = [float(s["phi"]) for s in states]
            mid = certified.get((solver, episode, run[len(run) // 2][0]))
            cp = mid["decision_evidence"]["counterfactual_profile"] if mid else {}
            ve = mid["verification_evidence"]["verification_profile"] if mid else {}
            n = len(states)
            idx = list(range(0, n, max(1, n // 9)))[:10] if n > 1 else [0]
            macros.append({
                "name": name, "family": family, "solver": solver,
                "seed": run[0][2]["seed"], "start": run[0][0], "end": run[-1][0], "length": n,
                "series": {
                    "v": [round(v[k], 3) for k in idx],
                    "d": [round(d[k], 3) for k in idx],
                    "phi": [round(phi[k], 3) for k in idx],
                },
                "arc": {
                    "steps": n,
                    "v_start": round(v[0], 3), "v_min": round(min(v), 3), "v_end": round(v[-1], 3),
                    "d_start": round(d[0], 3), "d_end": round(d[-1], 3),
                    "phi_start": round(phi[0], 3), "phi_end": round(phi[-1], 3),
                    "phi_sign_changes": sign_changes(phi),
                },
                "m5_m6": {
                    "minimum_flip_concept": cp.get("minimum_flip_concept"),
                    "minimum_flip_distance": round(float(cp.get("minimum_flip_distance", 0)), 4),
                },
                "m7": {
                    rel: ("PASS" if ve.get(rel + "_pass") else "FAIL" if ve.get(rel + "_fail") else "n/a")
                    for rel in ("stop", "pedestrian", "curvature", "lane_symmetry")
                },
            })
            i = j + 1

    by_name = defaultdict(list)
    for m in macros:
        by_name[m["name"]].append(m)

    summary = {
        "total_certified_decisions": len(certified),
        "total_macro_instances": len(macros),
        "solvers": dict(Counter(k[0] for k in certified)),
        "distribution": {}, "primitives": [],
    }
    for name in sorted(by_name, key=lambda n: -len(by_name[n])):
        g = by_name[name]
        solvers = Counter(m["solver"] for m in g)
        lengths = [m["length"] for m in g]
        summary["distribution"][name] = {
            "instances": len(g), "solvers": dict(solvers),
            "mean_length": round(statistics.mean(lengths), 1),
            "decisions": sum(lengths),
        }
        reps = sorted(g, key=lambda m: -m["length"])[:2]
        summary["primitives"].append({
            "name": name, "family": g[0]["family"], "instances": len(g),
            "decisions": sum(lengths), "mean_length": round(statistics.mean(lengths), 1),
            "solvers": dict(solvers), "representatives": reps,
        })

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "certified_decisions": len(certified),
        "macro_instances": len(macros),
        "solvers": summary["solvers"],
        "distribution": {k: v["instances"] for k, v in summary["distribution"].items()},
        "solver_split": {k: v["solvers"] for k, v in summary["distribution"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
