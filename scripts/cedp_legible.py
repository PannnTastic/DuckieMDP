"""Legible three-question explanation per macro driving primitive.

For each primitive it answers, in plain terms grounded in the aggregated
evidence: (1) why this action, (2) what if the state or action were different
(counterfactual), (3) does the choice hold across instances and under
metamorphic perturbation. Emits legible.json for the walkthrough artifact.

Usage: cedp_legible.py OUT.json "dir[:solvers]" ...
"""

import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
SOURCES = []
for arg in sys.argv[2:]:
    if ":" in arg:
        p, s = arg.split(":", 1)
        SOURCES.append((Path(p), set(s.split(","))))
    else:
        SOURCES.append((Path(arg), None))

CURVE_KAPPA = 0.8
LANE_ERR = 0.05
CONVERGE = 0.01
OSC = 3
CRUISE_ERR = 0.05
SLABEL = {"q_learning": "Q-learning", "sarsa": "SARSA", "sac": "SAC-gated", "td3": "TD3"}
REL_OF = {"StopSign": "stop", "PedestrianCrossing": "pedestrian",
          "CurveNegotiation": "curvature", "LaneCorrection": "lane_symmetry"}


def base_family(s):
    if s.get("stop_present") and s.get("stop_distance") is not None:
        return "stop"
    if s.get("duck_present") or s.get("duck_active"):
        return "duck"
    k = float(s.get("curvature") or 0.0)
    if s.get("curvature_class") in ("curve_left", "curve_right") or abs(k) >= CURVE_KAPPA:
        return "curve"
    return "lane"


def sign_changes(xs):
    s = [1 if x > 1e-3 else -1 if x < -1e-3 else 0 for x in xs]
    s = [x for x in s if x]
    return sum(1 for a, b in zip(s, s[1:]) if a != b)


def load():
    anchors, certified = {}, {}
    for d, keep in SOURCES:
        for line in open(d / "full_decision_anchors.jsonl", encoding="utf-8"):
            a = json.loads(line)
            if keep and a["solver"] not in keep:
                continue
            anchors[(a["solver"], a["episode_id"], a["step_index"])] = a
        for p in glob.glob(str(d / "instance_shards" / "*.json")):
            x = json.load(open(p, encoding="utf-8"))
            if x["status"] != "CERTIFIED" or (keep and x["solver"] not in keep):
                continue
            certified[(x["solver"], x["episode_id"], x["step_index"])] = x
    return anchors, certified


def macro_name(fam, states, actions):
    if fam == "stop":
        return "StopSign", "stop"
    if fam == "duck":
        return "PedestrianCrossing", "duck"
    d = [float(s["d"]) for s in states]
    phi = [float(s["phi"]) for s in states]
    om = [float(a["omega_cmd"]) for a in actions]
    if fam == "curve":
        left = sum(1 for s in states if s.get("curvature_class") == "curve_left") >= \
               sum(1 for s in states if s.get("curvature_class") == "curve_right")
        return "CurveNegotiation" + ("Left" if left else "Right"), "curvature"
    conv = (abs(d[-1]) < abs(d[0]) - CONVERGE) or (abs(phi[-1]) < abs(phi[0]) - CONVERGE)
    if sign_changes(phi) >= OSC and sign_changes(om) >= OSC and not conv:
        return "OscillatorySteering", "lane"
    if max(abs(x) for x in d) < CRUISE_ERR and max(abs(x) for x in phi) < CRUISE_ERR and sign_changes(phi) < OSC:
        return "Cruise", "cruise"
    return "LaneCorrection" + ("Left" if statistics.mean(om) > 0 else "Right"), "lane"


def main():
    anchors, certified = load()
    by_ep = defaultdict(list)
    for (solver, ep, st) in certified:
        by_ep[(solver, ep)].append(st)

    # assign each certified decision a macro name; collect per-name aggregates
    agg = defaultdict(lambda: {
        "family": None, "instances": 0, "decisions": 0,
        "solvers": Counter(), "minflip": Counter(), "flipdist": [],
        "v_start": [], "v_min": [], "v_end": [], "brake_frac": [],
        "rel": defaultdict(list), "dsv": [], "foil_ld": [], "dclr": [],
    })
    for (solver, ep), steps in by_ep.items():
        steps.sort()
        seq = [(st, base_family(anchors[(solver, ep, st)]["state"]), anchors[(solver, ep, st)])
               for st in steps if (solver, ep, st) in anchors]
        i = 0
        while i < len(seq):
            fam = seq[i][1]
            j = i
            while j + 1 < len(seq) and seq[j + 1][1] == fam and seq[j + 1][0] == seq[j][0] + 1:
                j += 1
            run = seq[i:j + 1]
            states = [r[2]["state"] for r in run]
            actions = [r[2]["selected_action"] for r in run]
            name, family = macro_name(fam, states, actions)
            a = agg[name]
            a["family"] = family
            a["instances"] += 1
            a["solvers"][solver] += 1
            v = [float(s["v"]) for s in states]
            a["v_start"].append(v[0]); a["v_min"].append(min(v)); a["v_end"].append(v[-1])
            a["brake_frac"].append(sum(1 for x in v if x < 0.05) / len(v))
            for r in run:
                a["decisions"] += 1
                inst = certified[(solver, ep, r[0])]
                cp = inst["decision_evidence"]["counterfactual_profile"]
                ve = inst["verification_evidence"]["verification_profile"]
                oe = inst["outcome_evidence"]["physical_profile"]
                a["minflip"][cp.get("minimum_flip_concept")] += 1
                a["flipdist"].append(float(cp.get("minimum_flip_distance", 0)))
                a["dsv"].append(float(oe.get("delta_stop_violations", 0)))
                a["foil_ld"].append(float(oe.get("foil_lane_departure", 0)))
                a["dclr"].append(float(oe.get("delta_minimum_duck_clearance_m", 0)))
                for rel in ("stop", "pedestrian", "curvature", "lane_symmetry"):
                    if ve.get(rel + "_applicable"):
                        a["rel"][rel].append(1 if ve.get(rel + "_pass") else 0)
            i = j + 1

    def mean(xs):
        return round(statistics.mean(xs), 3) if xs else None

    prims = []
    for name in sorted(agg, key=lambda n: -agg[n]["decisions"]):
        a = agg[name]
        if name == "StopSign":
            rel = "stop"
        elif name == "PedestrianCrossing":
            rel = "pedestrian"
        elif name.startswith("Curve"):
            rel = "curvature"
        else:
            rel = "lane_symmetry"
        rel_pass = mean(a["rel"].get(rel, []))
        prims.append({
            "name": name, "family": a["family"],
            "instances": a["instances"], "decisions": a["decisions"],
            "policies": [SLABEL[s] for s in ("q_learning", "sarsa", "sac", "td3") if a["solvers"].get(s)],
            "solvers": dict(a["solvers"]),
            "action": {
                "v_start": mean(a["v_start"]), "v_min": mean(a["v_min"]), "v_end": mean(a["v_end"]),
                "brake_frac": mean(a["brake_frac"]),
            },
            "why_trigger": a["minflip"].most_common(1)[0][0] if a["minflip"] else None,
            "state_cf": {
                "concept": a["minflip"].most_common(1)[0][0] if a["minflip"] else None,
                "min_flip_distance": mean(a["flipdist"]),
                "top": a["minflip"].most_common(3),
            },
            "action_cf": {
                "delta_stop_violations": mean(a["dsv"]),
                "foil_lane_departure_rate": mean(a["foil_ld"]),
                "delta_duck_clearance": mean(a["dclr"]),
            },
            "holds": {
                "relation": rel, "pass_rate": rel_pass,
                "applicable_decisions": len(a["rel"].get(rel, [])),
                "n_policies": len(a["solvers"]),
            },
        })

    summary = {"total_decisions": len(certified), "total_instances": sum(p["instances"] for p in prims),
               "primitives": prims}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for p in prims:
        print("%-22s inst=%-4d dec=%-5d pol=%d | trigger=%-8s Δflip=%.3f | %s holds=%s (%d dec)" % (
            p["name"], p["instances"], p["decisions"], p["holds"]["n_policies"],
            p["why_trigger"], p["state_cf"]["min_flip_distance"] or 0,
            p["holds"]["relation"], p["holds"]["pass_rate"], p["holds"]["applicable_decisions"]))


if __name__ == "__main__":
    main()
