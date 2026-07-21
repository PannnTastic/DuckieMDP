# M10 — Solver-Aware Rule Extraction

## Outcome

M10 is complete. The frozen Q-learning and SAC policies were distilled with
scikit-learn decision trees without retraining either policy.

The result has two different scientific dispositions:

- the Q-learning trees are exact on the entire 9,000-state representable
  domain;
- the SAC trees are useful global summaries, but the continuous-action tree is
  **not** a safe replacement for the actor because closed-loop stop and
  pedestrian compliance degrade despite high held-out fidelity.

This distinction is a required result, not a failed demo hidden by a global
accuracy score.

## Frozen software and policy contract

```text
scikit-learn  1.6.1
joblib        1.4.2
NumPy         1.20.0
random_state  0
```

These packages are pinned in `requirements-sac.txt`. They are used only for
post-hoc explanation. Q-learning and SAC checkpoint hashes remain unchanged,
and neither solver is trained during M10.

The source is `src/explainability/rule_extraction.py`; the complete experiment
is `scripts/run_m10_rule_extraction.py`.

## Q-learning: exact rule extraction

The input consists of all seven Q state-bin coordinates and all 9,000 rows from
M8. The action tree predicts the exact greedy action; the primitive tree
predicts the frozen static driving primitive.

| Tree | Depth | Leaves | Nodes | Fidelity |
|---|---:|---:|---:|---:|
| Q action | 15 | 257 | 513 | **100%** |
| Q primitive | 15 | 138 | 275 | **100%** |

Fidelity is 100% independently on representable, valid-manifold, reachable, and
supported strata. Therefore the Q action tree may be called exact on the
explicitly named representable table domain. It is not a proof about continuous
states between bins.

All seven dimensions are used. The `tracking_error_bin` remains the entangled
quantity `phi + d`; its rules must not be described as pure heading rules.

### Q closed-loop equivalence

The action tree was run in the environment for the five frozen Q evaluation
seeds. Every selected tree action was compared online with the frozen greedy
Q adapter:

```text
decisions compared:  1,250
action mismatches:   0
```

All five episodes reached timeout, with zero collision/off-road events, 100%
stop compliance, and mean return 49.60. Because the tree is exact on every
addressable table state, the observed rollout equivalence is expected rather
than accidental.

## SAC datasets

The SAC surrogate is trained only on deterministic development rollouts and is
tested on disjoint final seeds.

| Split | Seeds | Decision states |
|---|---|---:|
| Development | 2101, 2202, 2303, 2404, 2505 | 2,000 |
| Held-out | 20101, 20202, 20303, 20404, 20505 | 2,000 |

Primitive labels use the static mapping `P(s,a)`. Temporal transition labels,
such as resume and oscillatory behavior requiring prior context, remain
trajectory-level explanations and are not silently forced into a memoryless
surrogate.

## SAC primitive tree

The primitive classification tree has depth 10, 54 leaves, and 107 nodes. It
uses ten of the fifteen continuous features.

| Held-out stratum | Samples | Fidelity | Balanced fidelity | Macro F1 |
|---|---:|---:|---:|---:|
| All | 2,000 | **94.90%** | 82.33% | 82.90% |
| Lane context | 1,907 | 95.54% | 84.88% | 74.27% |
| Pedestrian context | 35 | 94.29% | 96.67% | 89.94% |
| Stop context | 58 | **74.14%** | 67.93% | 38.74% |

The global threshold of 85% is passed. However, the stop-context fidelity is
only 74.14%. Therefore:

- the primitive tree is eligible as a global policy summary;
- it is not the primary explanation for a specific stop decision;
- stop explanations continue to use the original actor, M6 counterfactuals,
  M5 paired outcomes, and M7 verification.

This is why global fidelity alone is not sufficient for safety-critical claims.

## SAC continuous-action tree

The multi-output regression tree predicts `(v_cmd, omega_cmd)`. It has depth 14,
495 leaves, and 989 nodes. This size already shows that the continuous actor is
not compressible into a tiny rule list without losing behavior.

### Held-out action fidelity

| Context | MAE `v_cmd` | MAE `omega_cmd` |
|---|---:|---:|
| All | **0.01029** | **0.07967** |
| Lane | 0.01019 | 0.07649 |
| Pedestrian | 0.00177 | 0.04718 |
| Stop | 0.01881 | **0.20361** |

The global frozen limits (`0.03`, `0.20`) pass, although stop-context steering
slightly exceeds the global omega limit. When the predicted continuous action
is converted back into a primitive, held-out fidelity is:

| Context | Action-induced primitive fidelity |
|---|---:|
| All | **97.45%** |
| Lane | 97.80% |
| Pedestrian | 100% |
| Stop | **84.48%** |

Again, rare stop states are where the global score is least representative.

## Tree-only closed-loop validation

The actor and action tree were each evaluated from the same five held-out
initial seeds for the full five-minute horizon. The surrogate rollout encodes
the state directly and invokes the SAC actor **zero times**.

| Metric | Original actor | Action tree |
|---|---:|---:|
| Timeout rate | 100% | 100% |
| Total failure rate | 0% | 0% |
| Collision rate | 0% | 0% |
| Mean return | **106.74** | **20.73** |
| Mean progress | 28.12 m | 28.90 m |
| Mean absolute `d` | 0.0634 m | 0.0595 m |
| p95 absolute `d` | 0.1054 m | 0.1248 m |
| Stop compliance | **100%** | **85.71%** |
| Duckie yield-step rate | **100%** | **37.14%** |

The surrogate had five stop violations across the five runs. It also failed to
hold for the active Duckie on many crossing steps, even though no collision
occurred in this small seed ensemble. Mean return fell to 19.42% of actor
return.

Therefore the action tree is useful for inspecting approximate decision
regions, but it is explicitly **not eligible as a replacement controller**.
The original actor remains the policy being explained.

This result demonstrates why M10 measures both open-loop fidelity and
closed-loop outcomes: small action errors can change future state visitation
and accumulate into safety-rule violations.

## Rule artefacts

Human-readable rule files are exported for audit:

- Q action rules: 466 text lines;
- Q primitive rules: 280 lines;
- SAC primitive rules: 160 lines;
- SAC action rules: 1,087 lines.

Every root-to-leaf rule is also stored as structured JSON with its conditions,
prediction, sample count, and impurity. The model binaries carry SHA-256 hashes
in the summary manifest.

## Acceptance disposition

The predeclared structural/fidelity gates all pass:

- Q action and primitive trees are exact on 9,000 states;
- Q rollout has zero action mismatches;
- SAC primitive fidelity is at least 85%;
- SAC action-induced primitive fidelity is at least 80%;
- global action MAEs pass their thresholds;
- SAC surrogate has no collision and no terminal failure in five runs.
- full regression suite passes all **137 tests** after adding scikit-learn.

`main_result_eligible=true` means **eligible as a global rule summary**. It does
not mean policy replacement. The policy-replacement claim is denied by design
and reinforced by the observed stop/yield safety gap.

## Reproduce and inspect

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m scripts.run_m10_rule_extraction
```

Generated files:

- `runs/explanations/m10_rule_extraction/m10_summary.json`;
- `runs/explanations/m10_rule_extraction/leaf_rules.json`;
- four `.joblib` models;
- four human-readable `*_rules.txt` files.

Compact acceptance evidence is stored in
`docs/m10_rule_extraction_acceptance.json`.

## Scientific conclusion

For Q-learning, rule extraction is exact inspection of the finite policy. For
SAC, rule extraction is an approximate global summary whose fidelity must be
reported by context. The experiment provides direct evidence that a high
per-state fidelity score does not guarantee closed-loop safety equivalence.
Local SAC explanations must continue to query the original actor and simulator,
not the surrogate tree.
