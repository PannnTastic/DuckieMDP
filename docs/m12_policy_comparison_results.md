# M12  Unified Q-learning and SAC Explanation Report

## Outcome

The machine-readable part of M12 is complete and accepted. The comparison uses
the original frozen policies, not the M10 surrogate trees:

- Q-learning: greedy, teacher-free, lowest-action-id tie break;
- SAC: deterministic actor mean;
- map: `small_loop`;
- five matched seeds: 20101, 20202, 20303, 20404, 20505;
- identical initial poses to absolute tolerance `1e-7`;
- `frame_skip=6`;
- 1,500 physics ticks or 250 policy decisions (50 seconds) per episode.

All ten episodes reached timeout. Neither solver had off-road, collision,
stop-violation, or unsafe-proceed events in this matched ensemble.

This is a behavioral comparison of two trained controllers, not a claim that
the solvers differ only in optimization algorithm. Their state encodings,
action spaces, exploration histories, and training regimes remain different.

## Evaluation contract

The shared comparison config is:

`runs/explanations/m12_policy_comparison/shared_comparison_config.yaml`.

The straight-steering reward term is neutralized for this comparison because
the legacy Q wrapper does not pass continuous `action_omega` into the reward
function. This prevents the reported returns from silently using different
reward formulas. The policy actions themselves are unchanged.

Checkpoint SHA-256:

| Policy | Checkpoint hash |
|---|---|
| Q-learning | `59929f4e2c1968d274e9d3aa6c83ccb2cd7c915c632093cfaba6b544340906dd` |
| SAC | `0b01447edd85e539de57f9a304fc287d26d5d7c4e73a5a51cd493fea2f4c4f2b` |

## Matched rollout results

| Metric | Q-learning | SAC |
|---|---:|---:|
| Episodes | 5 | 5 |
| Decision steps | 1,250 | 1,250 |
| Timeout rate | 100% | 100% |
| Mean return | 44.63 | 17.82 |
| Stop compliance | 100% (10/10) | 100% (5/5) |
| Pedestrian yield-command rate | 100% (35/35 active steps) | 100% (35/35 active steps) |
| Unsafe-proceed rate | 0% | 0% |
| Unnecessary-brake rate | 0% | 0% |
| Undesirable primitive rate | 6.96% | 0% |
| First brake distance, mean | 0.326 m | 0.251 m |
| Straight-road mean absolute omega | 0.480 | 0.540 |
| Straight-road lane-error/omega correlation | 0.759 | 0.655 |

The Q policy completed approximately two loop passes per episode and therefore
encountered ten stop obligations; SAC encountered five. This is an observed
trajectory difference, not an environment mismatch.

The first-brake metric indicates that Q begins the discrete brake action
earlier. SAC decelerates continuously and first crosses the common hold
threshold closer to the stop line. Both satisfy every observed stop.

## Steering interpretation

Q-learning receives seven macro-actions. It can only jump between fixed
steering commands, and 87 of 1,250 steps are labelled
`OscillatorySteering` (6.96%). Most such segments last one decision step.

SAC has no oscillatory-primitive labels in this ensemble, but its mean absolute
steering magnitude on straight road is slightly larger. Therefore the evidence
supports the narrower statement:

> SAC changes steering direction less abruptly in these rollouts, while Q uses
> a slightly smaller mean steering magnitude on states classified as straight.

It does not support the blanket claim that every notion of smoothness is better
for SAC.

The Q straight-road statistic uses `curvature_class == "straight"` when
continuous curvature is unavailable. Curve tiles are explicitly excluded.

## Primitive profile

The most frequent Q primitives are:

| Primitive | Rate |
|---|---:|
| CruiseCurveLeft | 21.60% |
| LaneCorrectLeft | 14.80% |
| CruiseStraight | 14.08% |
| ApproachCrossing | 12.72% |
| ResumeAfterStop | 11.52% |

The most frequent SAC primitives are:

| Primitive | Rate |
|---|---:|
| ApproachCrossing | 45.92% |
| LaneCorrectLeft | 20.16% |
| CruiseCurveLeft | 14.64% |
| LaneCorrectRight | 7.92% |
| CruiseStraight | 4.64% |

`ApproachCrossing` means that a relevant Duckie is visible but is not
currently an active crossing threat. It does not mean the ego is braking.
Because the repeated Duckie remains spatially relevant for long intervals, this
label dominates SAC's timeline. This is a lexicon semantic effect and must not
be reported as 45.92% yielding.

## Feature-influence signatures

The comparison keeps solver-specific explanation methods honest:

- Q-learning: supported one-bin action-flip rate;
- SAC: absolute Integrated Gradients over the three frozen critical anchors.

They are normalized within solver. They are qualitative signatures, not values
on a common causal or probabilistic scale.

Q's `tracking_error_bin = phi + d` is lane-heading entangled. Pure lane and
heading dimensions are therefore excluded from the common subset.

| Comparable concept | Q normalized flip signature | SAC normalized IG signature |
|---|---:|---:|
| Speed | 0.000 | 0.155 |
| Road geometry | 0.445 | 0.031 |
| Stop | 0.456 | 0.093 |
| Pedestrian | 0.099 | 0.721 |

The Q policy changes macro-action most often when supported road or stop bins
change. The SAC actor's three-anchor IG signature is dominated by pedestrian
features. These are local/global-support diagnostics from different estimators;
the table is useful for forming hypotheses, not declaring absolute feature
importance across solvers.

## Unified explanation artefact

`src/explainability/explanation_report.py` assembles the complete accepted
pipeline into:

- `runs/explanations/m12_unified_report/unified_explanation_report.json`;
- `runs/explanations/m12_unified_report/local_explanation_index.csv`.

The JSON contains:

1. six local explanations: lane correction, stop hold, and pedestrian yield
   for both policies;
2. selected action, foil action, primitive contrast, trigger state, Q-margin
   where available, and natural-language template output;
3. factual and counterfactual physical outcomes and reward profiles;
4. M6 valid-manifold response curves;
5. M7 metamorphic verification;
6. M8 exhaustive Q-table characterization and safety checks;
7. M9 SAC IG, stability, local boundaries, and critic probes;
8. M10 solver-aware rule extraction;
9. M12 matched rollout comparison;
10. checkpoint and source-artifact hashes.

The report passes all binding gates. A false `teacher_active` invariant is
handled as an expected condition, not mistakenly treated as a failed Boolean.

The integrity gate also compares checkpoint paths and SHA-256 values. All six
local cases and stages M6--M10 use the same canonical full-task Q table and the
same canonical SAC actor. The prior Q lane-only mismatch was regenerated with
the full-task Q checkpoint. The exact policy/config matrix is recorded in
`docs/explanation_target_audit.md`.

## Reproduce

```bash
PYTHONPATH=. PYTHONWARNINGS=ignore .venv-sac/bin/python \
  scripts/run_m12_policy_comparison.py

PYTHONPATH=. .venv-sac/bin/python scripts/run_m12_unified_report.py

PYTHONWARNINGS=ignore .venv-sac/bin/python -m pytest -q -p no:warnings
```

Final regression result:

```text
147 passed in 65.55s
```

## Acceptance disposition

All predeclared fairness and report-integrity checks pass. Primitive coverage is
100% for both policies. The six paired branches satisfy the deterministic
branch invariants, teacher is inactive, and local explanations use the original
policies.

The five-seed ensemble is adequate for this M12 integration validation and
descriptive comparison. It is not a confidence interval for population-level
performance. Larger held-out ensembles remain necessary for thesis inferential
claims.

M11 was subsequently executed after this M12 comparison. The lexicon freeze
predates clustering, and the result provides partial solver-dependent support;
see `docs/m11_bottom_up_clustering_results.md`. The original M12 comparison
metrics and policies remain unchanged.

## Multiview explanation overlay

Both existing renderers now expose the stored explanation vocabulary without
changing either policy:

- decision primitive and frozen trigger rule;
- Q second-best foil plus exact Q-margin;
- SAC semantic canonical-action foil plus critic-probe delta;
- explicit critic-probe caveat in the underlying overlay record.

Smoke renders passed at 1920x1080, 20 FPS, and 30 physics ticks for each policy.
The ignored M12 run directory contains the two short validation videos.

Video remains a secondary projection of the versioned JSON/CSV evidence.
