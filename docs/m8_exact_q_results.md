# M8 — Exact Q-Table Characterization and Verification

## Outcome

M8 is complete. The teacher-guided Q-learning checkpoint was inspected using a
teacher-free, deterministic greedy adapter. All addressable table cells were
enumerated; no state sampling was used for the policy map or property checker.

The checkpoint remains shaped:

```text
(5, 5, 3, 3, 4, 2, 5, 7)
```

The first seven dimensions contain exactly 9,000 discrete states and the final
dimension contains seven macro-actions.

## State strata

| Stratum | States | Meaning |
|---|---:|---|
| Representable | 9,000 | Every addressable Q-table index |
| Valid manifold | 7,875 | Representative state passes semantic validation |
| Reachable | 201 | Observed at least once in the frozen 30-episode rollout |
| Supported | 138 | Evaluation reach count at least 3 |

The checkpoint does not contain training visit counts. Therefore:

- `training_visit_count` is always `null`;
- 201 states are labelled `reached_only`;
- 8,799 states remain `unknown`;
- `supported` is a historical evaluation proxy, not proof of training
  visitation.

The frozen rollout counted 7,241 decision states across 30 episodes. It ended
with 28 timeouts and two off-road terminations, with mean episode length
241.37 decisions. Evaluation uses the lowest action id for exact Q-value ties,
so the reconstruction is reproducible and never calls the teacher.

## Safety-property verification

### Near unsatisfied stop must not use a fast action

| Stratum | Applicable | Violations | Rate |
|---|---:|---:|---:|
| Representable | 1,125 | 1,109 | 98.58% |
| Valid manifold | 1,125 | 1,109 | 98.58% |
| Reachable | 8 | 0 | 0% |
| Supported | 4 | 0 | 0% |

### Near crossing pedestrian must not use a fast action

| Stratum | Applicable | Violations | Rate |
|---|---:|---:|---:|
| Representable | 1,800 | 1,775 | 98.61% |
| Valid manifold | 1,575 | 1,550 | 98.41% |
| Reachable | 20 | 4 | 20% |
| Supported | 10 | 0 | 0% |

The four reachable pedestrian violations have evaluation reach counts of only
one or two and therefore do not pass the frozen support threshold. They remain
real audit findings rather than being discarded:

```text
(2,0,0,0,0,0,4) -> fast_right    reach_count=1
(2,0,1,0,0,0,4) -> fast_straight reach_count=2
(2,0,2,0,0,0,4) -> fast_straight reach_count=2
(3,0,1,0,0,0,4) -> fast_straight reach_count=1
```

The very high whole-table violation rate must not be interpreted as a 98%
deployment failure rate. About 96.54% of representable cells have zero
Q-margin ties, and 8,799 cells have unknown visitation provenance. Exact
verification exposes this coverage weakness: deterministic tie-breaking in
unsupported regions can execute a fast action if such a state is reached.

## Q-margin

| Stratum | Median margin | Tie rate | Low-margin boundary |
|---|---:|---:|---:|
| Representable | 0.000 | 96.54% | 96.64% |
| Valid manifold | 0.000 | 96.05% | 96.17% |
| Reachable | 5.378 | 4.48% | 5.97% |
| Supported | 5.883 | 0% | 0.72% |

Q-margin is action-value separation, not a probability of confidence. The
policy map marks a state as a final local boundary when it has low margin or
an action-changing one-bin neighbor that is also supported. Under this strict
support-preserving definition, 78.99% of supported states are near a discrete
policy boundary.

## One-bin influence

There are 3,362 directed one-bin comparisons that change the selected action
over the representable table. For pairs in which both states are supported:

| Dimension | Comparisons | Flips | Flip rate |
|---|---:|---:|---:|
| Lateral-offset bin | 104 | 2 | 1.92% |
| Tracking-error bin | 140 | 108 | 77.14% |
| Speed bin | 128 | 0 | 0% |
| Curvature class | 86 | 42 | 48.84% |
| Stop-distance bin | 8 | 0 | 0% |
| Stop-satisfied flag | 8 | 8 | 100% |
| Duck-threat bin | 92 | 10 | 10.87% |

This shows that the learned supported policy is driven most strongly by the
entangled tracking-error feature, curvature class, and completion of the stop
obligation. The tabular tracking-error dimension represents `phi + d`; it must
not be described as pure heading influence.

## Primitive distribution on supported states

The 138 supported cells include:

- 50 `ApproachCrossing`;
- 31 lane corrections (`LaneCorrectLeft`/`LaneCorrectRight`);
- 14 `YieldHold`;
- 14 `ResumeAfterStop`;
- 5 `DecelerateStop` and 4 `StopHold`;
- the remaining cells are cruise, lane deceleration, or recovery primitives.

No supported state is labelled `UnsafeProceed`, although four reachable-only
cells are. Primitive distribution over all 9,000 cells is retained in the JSON
audit but is not blended with the supported distribution.

## Explanation scope

Each row of `exact_policy_map.csv` is a per-state explanation containing:

- semantic representative state and exact discrete index;
- selected and second-ranked action;
- all seven Q-values and Q-margin;
- driving primitive and trigger;
- validity, reachability, support, and provenance;
- boundary status;
- Q2 foil scope.

Q2 explanations enter the main result only for reachable or supported states.
All others are explicitly marked `appendix_unsupported_policy_region`.

## Validation

- Exact states enumerated once: 9,000/9,000.
- Representative-state round trip to original index: 9,000/9,000.
- Q-table shape unchanged.
- Safety checks stratified rather than blended.
- Unit and full regression: 125 tests passed.
- Generated patch rejects negative/wrong-shaped visit-count artefacts.

## Reproduce and inspect

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m scripts.run_m8_exact_q
```

Generated outputs:

- `runs/explanations/m8_exact_q/m8_summary.json`;
- `runs/explanations/m8_exact_q/exact_policy_map.csv`;
- `runs/explanations/m8_exact_q/one_bin_action_flips.csv`;
- `runs/explanations/m8_exact_q/safety_property_violations.csv`;
- `runs/explanations/m8_exact_q/evaluation_reach_counts.npy`.

