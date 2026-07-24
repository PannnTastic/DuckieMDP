"""Trajectory-based naming candidates from the M5 paired-outcome rollout.

Instead of a single decision snapshot, each certified explanation is named from
how its state evolves across the M4/M5 factual trajectory: whether the lane
error converges (LaneCorrection) or oscillates (OscillatorySteering), whether
speed ramps down and is held at a stop (StopApproach/Hold/Resume), whether a
Duckie is being yielded to, or whether steering tracks a curve. The M5
counterfactual trigger and M7 metamorphic result are carried through for display.
"""

import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/explanations/cedp_v2_gated")

STOP_V = 0.045
YIELD_V = 0.05
CURVE_KAPPA = 0.8
LANE_ERR = 0.045
CONVERGE = 0.008
OSC_CHANGES = 3


def sign_changes(xs):
    s = [1 if x > 1e-4 else -1 if x < -1e-4 else 0 for x in xs]
    s = [x for x in s if x != 0]
    return sum(1 for a, b in zip(s, s[1:]) if a != b)


def traj_series(steps):
    st = [s["decision"]["state"] for s in steps]
    ac = [s["decision"]["action"] for s in steps]
    return {
        "d": [float(x["d"]) for x in st],
        "phi": [float(x["phi"]) for x in st],
        "v": [float(x["v"]) for x in st],
        "kappa": [float(x.get("curvature") or 0.0) for x in st],
        "omega": [float(a["omega_cmd"]) for a in ac],
        "cclass": [x.get("curvature_class") for x in st],
        "stop_present": [bool(x.get("stop_present")) for x in st],
        "stop_prog": [float(x.get("stop_hold_progress") or 0.0) for x in st],
        "stop_sat": [bool(x.get("stop_satisfied")) for x in st],
        "duck_present": [bool(x.get("duck_present")) for x in st],
        "duck_active": [bool(x.get("duck_active")) for x in st],
    }


def name_trajectory(t):
    v0, vN, vmin = t["v"][0], t["v"][-1], min(t["v"])
    dabs0, dabsN = abs(t["d"][0]), abs(t["d"][-1])
    pabs0, pabsN = abs(t["phi"][0]), abs(t["phi"][-1])
    mean_omega = statistics.mean(t["omega"])

    # 1. Stop sign present anywhere on the rollout.
    if any(t["stop_present"]):
        if any(t["stop_sat"]) and vN > v0 + 0.03:
            return "ResumeAfterStop", "stop", "stop satisfied then speed rises"
        if vmin < STOP_V and (max(t["stop_prog"]) > 0.25 or vN < STOP_V):
            return "StopHold", "stop", "speed held near zero at the line"
        if vN < v0 - 0.02:
            return "StopApproach", "stop", "speed ramps down toward the sign"
        return "StopApproach", "stop", "approaching an unsatisfied stop"

    # 2. Pedestrian in the corridor.
    if any(t["duck_active"]) or any(t["duck_present"]):
        if vN > v0 + 0.03 and not t["duck_active"][-1]:
            return "ResumeAfterYield", "duck", "Duckie clears then speed rises"
        if vmin < YIELD_V:
            return "PedestrianYield", "duck", "speed dropped to yield"
        return "PedestrianApproach", "duck", "Duckie ahead, slowing"

    # 3. Curve — the road bends across the rollout.
    curve_frac = sum(1 for c in t["cclass"] if c in ("curve_left", "curve_right")) / len(t["cclass"])
    mean_abs_k = statistics.mean(abs(k) for k in t["kappa"])
    if curve_frac >= 0.5 or mean_abs_k >= CURVE_KAPPA:
        left = sum(1 for c in t["cclass"] if c == "curve_left") >= sum(1 for c in t["cclass"] if c == "curve_right")
        if mean_abs_k >= CURVE_KAPPA:
            left = statistics.mean(t["kappa"]) > 0
        return "CurveNegotiation" + ("Left" if left else "Right"), "curvature", "steering tracks a sustained curve"

    # 4. Oscillatory steering — heading and steer reverse without converging.
    phi_osc = sign_changes(t["phi"])
    omega_osc = sign_changes(t["omega"])
    converging = (dabsN < dabs0 - CONVERGE) or (pabsN < pabs0 - CONVERGE)
    if phi_osc >= OSC_CHANGES and omega_osc >= OSC_CHANGES and not converging:
        return "OscillatorySteering", "lane", "heading and steer reverse %d/%d times without converging" % (phi_osc, omega_osc)

    # 5. Lane correction — a meaningful, converging heading/lateral fix.
    if (dabs0 >= LANE_ERR or pabs0 >= LANE_ERR) and converging:
        direction = "Left" if mean_omega > 0 else "Right"
        return "LaneCorrection" + direction, "lane", "lane error converges under corrective steer"

    # 6. Cruise — stable straight travel.
    if dabs0 < LANE_ERR and pabs0 < LANE_ERR and phi_osc < OSC_CHANGES:
        return "Cruise", "cruise", "stable straight travel"

    direction = "Left" if mean_omega > 0 else "Right"
    return "LaneCorrection" + direction, "lane", "residual heading/lateral correction"


def main():
    records = []
    for path in glob.glob(str(RUN / "instance_shards" / "*.json")):
        d = json.load(open(path, encoding="utf-8"))
        if d["status"] != "CERTIFIED":
            continue
        prp = d["provenance"].get("paired_report_path")
        if not prp or not Path(prp).exists():
            continue
        report = json.load(open(prp, encoding="utf-8"))
        steps = report.get("factual", {}).get("trajectory", {}).get("steps")
        if not steps:
            continue
        t = traj_series(steps)
        name, family, reason = name_trajectory(t)
        cp = d["decision_evidence"]["counterfactual_profile"]
        ve = d["verification_evidence"]["verification_profile"]
        n = len(t["v"])
        idx = list(range(0, n, max(1, n // 8)))[:9]
        records.append({
            "name": name, "family": family, "reason": reason,
            "solver": d["solver"], "seed": d["seed"], "step": d["step_index"],
            "series": {
                "d": [round(t["d"][i], 3) for i in idx],
                "phi": [round(t["phi"][i], 3) for i in idx],
                "v": [round(t["v"][i], 3) for i in idx],
                "omega": [round(t["omega"][i], 3) for i in idx],
            },
            "traj_summary": {
                "steps": n,
                "d_start": round(t["d"][0], 3), "d_end": round(t["d"][-1], 3),
                "phi_start": round(t["phi"][0], 3), "phi_end": round(t["phi"][-1], 3),
                "v_start": round(t["v"][0], 3), "v_end": round(t["v"][-1], 3), "v_min": round(min(t["v"]), 3),
                "phi_sign_changes": sign_changes(t["phi"]),
                "curve_frac": round(sum(1 for c in t["cclass"] if c and c.startswith("curve")) / n, 2),
            },
            "m5_m6_counterfactual": {
                "minimum_flip_concept": cp.get("minimum_flip_concept"),
                "minimum_flip_distance": round(float(cp.get("minimum_flip_distance", 0)), 4),
                "attempts": cp.get("attempts"), "valid_attempts": cp.get("valid_attempts"),
            },
            "m7_metamorphic": {
                rel: ("PASS" if ve.get(rel + "_pass") else "FAIL" if ve.get(rel + "_fail") else "n/a")
                for rel in ("stop", "pedestrian", "curvature", "lane_symmetry")
            },
        })

    by_name = defaultdict(list)
    for r in records:
        by_name[r["name"]].append(r)

    summary = {"total_certified": len(records), "method": "trajectory (M4/M5 rollout)",
               "distribution": {}, "primitives": []}
    for name in sorted(by_name, key=lambda n: -len(by_name[n])):
        group = by_name[name]
        solvers = Counter(r["solver"] for r in group)
        summary["distribution"][name] = {"count": len(group), "solvers": dict(solvers)}
        reps = sorted(group, key=lambda r: r["m5_m6_counterfactual"]["minimum_flip_distance"])[:2]
        summary["primitives"].append({
            "name": name, "family": group[0]["family"], "count": len(group),
            "solvers": dict(solvers), "representatives": reps,
        })

    (RUN / "naming_trajectory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(records),
                      "distribution": {k: v["count"] for k, v in summary["distribution"].items()},
                      "solver_split": {k: v["solvers"] for k, v in summary["distribution"].items()}}, indent=2))


if __name__ == "__main__":
    main()
