# M9 — SAC Internal Diagnostics

## Outcome

M9 is complete and accepted for the frozen five-minute SAC checkpoint. The
stage diagnoses the deterministic actor and both learned critics; it does not
retrain the policy and does not activate a teacher.

Checkpoint:

```text
artifacts/sac/full_repeat_duck_5min/sac_best.pt
SHA-256: 0b01447edd85e539de57f9a304fc287d26d5d7c4e73a5a51cd493fea2f4c4f2b
```

The implementation is in `src/explainability/explain_sac.py`, with the
reproducible runner in `scripts/run_m9_sac_diagnostics.py`.

## What M9 explains

M9 adds solver-specific internal evidence to the shared per-state explanation:

- Integrated Gradients (IG) separately for `v_cmd` and `omega_cmd`;
- aggregation of the 15 input features into lane, heading, speed, road, stop,
  and pedestrian concepts;
- sensitivity to the selected IG baseline;
- double-critic comparisons at the actor action and five canonical reference
  actions;
- a bounded local search around real states for nearby action/primitive
  boundaries;
- attribution stability away from detected boundaries.

This evidence is diagnostic. It does not replace the trigger, state
counterfactual, paired simulator outcome, or metamorphic verification built in
M2–M7.

## Data and policy mode

The deterministic actor mean was rolled out on the five frozen development
seeds:

```text
2101, 2202, 2303, 2404, 2505
```

The audit collected 2,000 real decision states, capped at 400 decisions per
seed. None of these truncated audit rollouts terminated early. Fifty-seven
states were labelled `CruiseStraight`; only those real states were used to form
the empirical alternative IG baseline.

The primary baseline is the frozen neutral canonical state. The alternative is
the empirical centroid of the 57 real `CruiseStraight` observations. The
straight-line IG integration path may contain fractional mask values between
its endpoints; therefore IG remains a differentiable actor diagnostic rather
than a valid simulator trajectory.

## Integrated Gradients validation

The main IG calculation uses 1,024 trapezoidal integration steps. This number
was selected by convergence testing, not by weakening the gate:

| Integration steps | Maximum absolute completeness residual |
|---:|---:|
| 128 | 0.035753 |
| 256 | 0.011039 |
| 512 | 0.007474 |
| 1,024 | **0.002054** |

The frozen acceptance limit is 0.005, so the 1,024-step result passes. The
attributions sum to the actor-output difference within that numerical error.

## Baseline sensitivity and robust concept claims

| Anchor | Output | Neutral dominant concept | Empirical dominant concept | Stable? | Cosine similarity | Rank correlation |
|---|---|---|---|---:|---:|---:|
| Lane | `v_cmd` | speed | pedestrian | No | 0.373 | 0.414 |
| Lane | `omega_cmd` | heading | pedestrian | No | 0.694 | 0.511 |
| Stop | `v_cmd` | pedestrian | pedestrian | Yes | 0.475 | 0.718 |
| Stop | `omega_cmd` | pedestrian | stop | No | 0.314 | 0.593 |
| Duck | `v_cmd` | pedestrian | pedestrian | Yes | 0.919 | 0.797 |
| Duck | `omega_cmd` | pedestrian | pedestrian | Yes | 0.414 | 0.610 |

Consequently, the main result may claim pedestrian dominance for both outputs
at the Duckie anchor and for speed at the stop anchor. It must not claim a
baseline-independent dominant concept for either lane output or stop-anchor
steering.

The stop anchor also contains a visible but inactive Duckie. Pedestrian
attribution there is an observed actor dependency, not proof that the Duckie
caused the stop decision. That distinction requires the M6 state intervention
and M5 paired-outcome evidence.

## Actor outputs at the three real anchors

| Anchor | `v_cmd` | `omega_cmd` |
|---|---:|---:|
| Lane | 0.0470 | -0.7515 |
| Stop | 0.1104 | -1.1350 |
| Duck crossing | 0.0115 | +0.0587 |

The Duckie action is a near-full hold. The stop anchor is an approach/deceleration
state rather than the completed hold itself.

## Critic probe comparisons

Each state compares the actor action with brake, slow-straight,
cruise-straight, corrective-left, and corrective-right. Every non-actor probe
is labelled:

```text
LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT
```

No replay snapshot exists in the checkpoint, so the audit never calls an
action definitely out-of-distribution. It reports normalized action distance,
actor log-probability, both critic values, their minimum, and critic
disagreement.

Important findings:

- lane-anchor actor probe: `min(Q1,Q2)=-91.743`, critic disagreement `32.599`;
- stop-anchor actor probe: `min(Q1,Q2)=-76.421`, disagreement `0.395`;
- Duckie-anchor actor probe: `min(Q1,Q2)=-57.625`, disagreement `0.894`;
- at the lane anchor, several reference actions receive a larger minimum-Q than
  the actor action, but the enormous critic disagreement and low-support labels
  prohibit interpreting this as an advantage or confidence score;
- at the Duckie anchor, actor and brake probes have nearly equal minimum-Q
  (`-57.625` versus `-57.637`), consistent with a holding action, but still not
  a safety proof.

These values are called **critic probe comparisons**, never critic advantage or
confidence. Counterfactual simulator outcomes remain the primary consequence
evidence.

## Boundary and stability audit

The local search perturbs only valid canonical fields using the frozen scales:

```text
d:                 ±0.005 m
phi:               ±0.01 rad
v:                 ±0.01
object distances:  ±0.02 m
```

Candidates violating schema invariants are not constructed. Candidates outside
the semantic manifold are retained as rejected records and are never passed to
the actor.

No boundary was detected around the three named M6 anchors at these local
scales. Across 12 evenly spaced real development anchors, nine were near a
primitive/action boundary and were separated from the non-boundary stability
claim. The remaining comparisons produced:

| Metric | Value |
|---|---:|
| Non-boundary comparisons | 64 |
| Median attribution distance | 0.000756 |
| p95 attribution distance | **0.004655** |
| Frozen acceptance threshold | 0.10 |
| Result | **PASS** |

A boundary label is not a failure; it marks a region where a small state change
can legitimately change the policy output or primitive.

## Acceptance

All M9 gates passed:

- actor and both critics load from the exact checkpoint;
- deterministic actor mean is the only policy mode explained;
- the alternative baseline is supported by real development states;
- maximum absolute IG completeness residual is at most 0.005;
- baseline sensitivity is reported for every anchor and output;
- all critic values are finite and every reference carries its support caveat;
- invalid boundary candidates are never queried;
- non-boundary attribution stability p95 is at most 0.10;
- full regression suite: **132 tests passed**.

## Reproduce and inspect

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m scripts.run_m9_sac_diagnostics
```

Generated audit artefacts:

- `runs/explanations/m9_sac_internal/m9_summary.json`;
- `runs/explanations/m9_sac_internal/ig_feature_attributions.csv`;
- `runs/explanations/m9_sac_internal/critic_probe_comparisons.csv`;
- `runs/explanations/m9_sac_internal/local_boundary_points.csv`;
- `runs/explanations/m9_sac_internal/attribution_stability.csv`.

The compact acceptance record is `docs/m9_sac_internal_acceptance.json`.

## Scientific disposition

M9 supports three strong conclusions:

1. the Duckie hold decision has a baseline-stable pedestrian attribution;
2. the actor attribution is locally stable away from detected boundaries;
3. critic probes are useful diagnostics but cannot carry the explanation claim,
   especially at the lane anchor where critic disagreement is large.

The next stage is M10 solver-aware rule extraction. It will summarize the exact
Q policy and the SAC primitive mapping globally while preserving fidelity,
support, and surrogate caveats.
