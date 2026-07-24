# Real Data Behind the Explanation-Derived Driving Primitives

This document is generated from the frozen four-policy artefacts in
`runs/explanations/cedp_v2_4policy`. The four names summarize
contiguous temporal M1–M13 explanations; they are not assigned directly
from one isolated `(state, action)` pair.

## Pipeline

```text
real rollout state/action
  -> Why: state counterfactual decision boundary
  -> What-if: paired factual/foil simulator outcomes
  -> Verification: metamorphic relation results
  -> Temporal: contiguous explanation arc
  -> human-readable driving primitive
```

## Real distribution

| Primitive | Temporal instances | Decisions | Solver coverage |
| --- | ---: | ---: | --- |
| LaneKeeping | 91 | 879 | q_learning=28, sac=19, sarsa=28, td3=16 |
| CurveNegotiation | 95 | 2038 | q_learning=30, sac=20, sarsa=30, td3=15 |
| StopCompliance | 30 | 631 | q_learning=10, sac=5, sarsa=10, td3=5 |
| PedestrianYield | 22 | 452 | q_learning=6, sac=5, sarsa=6, td3=5 |

## Evidence cards

### LaneKeeping

**Real source:** `sac`, seed `20202`, episode `sac_20202`, decisions `152–179`, representative explanation `cedp-instance-5fef7136e145216feac4` at decision `166`.

- **Why:** The selected action changes under lateral, heading, speed, curvature, stop_distance, duck_risk interventions; the nearest recorded decision boundary is heading at distance 0.0151.
- **What-if:** Selected vs foil: max |d| 0.032 m vs 0.033 m; max |phi| 0.032 rad vs 0.034 rad; lane departure no vs no; steering jerk 10.229 vs 11.639.
- **Verification:** pedestrian=PASS; curvature=FAIL; lane_symmetry=FAIL
- **Temporal arc:** deviation -> corrective steering -> recenter; 28 decisions; v 0.095 -> min 0.093 -> 0.096; d -0.054 -> -0.030; phi 0.142 -> 0.004.

### CurveNegotiation

**Real source:** `sac`, seed `20101`, episode `sac_20101`, decisions `75–101`, representative explanation `cedp-instance-a806b5816ae1a3d3e11f` at decision `88`.

- **Why:** The selected action changes under lateral, heading, speed, curvature, stop_distance, duck_risk interventions; the nearest recorded decision boundary is heading at distance 0.0186.
- **What-if:** Selected vs foil: max |d| 0.063 m vs 0.063 m; max |phi| 0.146 rad vs 0.139 rad; lane departure no vs no; steering jerk 8.584 vs 12.525.
- **Verification:** pedestrian=PASS
- **Temporal arc:** curve detected -> decelerate -> steer -> recover; 27 decisions; v 0.100 -> min 0.095 -> 0.101; d -0.025 -> -0.045; phi -0.088 -> 0.146.

### StopCompliance

**Real source:** `sac`, seed `20202`, episode `sac_20202`, decisions `25–50`, representative explanation `cedp-instance-ba2c99b9d9ec7bbb479a` at decision `38`.

- **Why:** The selected action changes under lateral, heading, speed, curvature, stop_distance, stop_satisfied, duck_risk interventions; the nearest recorded decision boundary is heading at distance 0.0464.
- **What-if:** Selected vs foil: full stops 0 vs 0; stop violations 0 vs 0; brake ratio 0.000 vs 0.000.
- **Verification:** pedestrian=PASS
- **Temporal arc:** approach -> decelerate -> stop -> hold -> resume; 26 decisions; v 0.108 -> min 0.008 -> 0.124; d -0.034 -> 0.015; phi 0.105 -> -0.246.

### PedestrianYield

**Real source:** `sac`, seed `20202`, episode `sac_20202`, decisions `93–124`, representative explanation `cedp-instance-47a0ce6962bcf654e1cb` at decision `109`.

- **Why:** The selected action changes under lateral, heading, speed, curvature, stop_distance, duck_risk interventions; the nearest recorded decision boundary is heading at distance 0.0474.
- **What-if:** Selected vs foil: minimum Duckie clearance 0.311 m vs 0.311 m; Duckie collision no vs no; brake ratio 0.000 vs 0.000.
- **Verification:** pedestrian=PASS
- **Temporal arc:** detect crossing -> yield/decelerate -> hold -> resume; 32 decisions; v 0.111 -> min 0.002 -> 0.103; d -0.029 -> -0.043; phi 0.058 -> -0.074.

## Interpretation boundary

The primitive carries all three explanation pillars. A verification
field may contain PASS, FAIL, ABSTAIN, or no applicable relation. In
this descriptive result, `Verification` means that the measured result
is preserved—not that every relation must pass.

The machine-readable values, including raw counterfactual, physical,
and verification profiles, are stored next to this document in
`runs/explanations/cedp_v2_4policy/primitive_real_evidence.json`.
