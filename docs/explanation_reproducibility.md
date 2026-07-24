# Reproducing the Four-Policy Explanation Pipeline

This is the canonical reproduction guide for the MDP explanation experiment.
It covers the four frozen policies—Q-learning, SARSA, SAC, and TD3—and the
mapping from per-decision explanation evidence to four temporal driving
primitive families.

## 1. What is reproduced

The main result is descriptive:

```text
real deterministic policy decision
  -> Why: state counterfactual / decision boundary
  -> What-if: selected action versus foil paired rollout
  -> Verification: metamorphic and safety result
  -> Temporal: contiguous sequence of explanations
  -> LaneKeeping | CurveNegotiation | StopCompliance | PedestrianYield
```

`Verification` preserves the measured status. A primitive may carry a `PASS`,
`FAIL`, `ABSTAIN`, or non-applicable relation. The main result does not claim
that every cluster or every primitive passes a universal certificate.
Support-aware cluster certification remains an optional audit documented in
[`support_aware_cedp_correction.md`](support_aware_cedp_correction.md).

## 2. Frozen inputs

The canonical configuration is
[`four_policy_reproducible.yaml`](../configs/explainability/four_policy_reproducible.yaml).
It references only files available in a Git clone:

| Solver | Frozen policy | Explanation mode |
| --- | --- | --- |
| Q-learning | `artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy` | greedy, teacher-free |
| SARSA | `artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy` | greedy, teacher-free |
| SAC | `artifacts/policies/sac_gated_duck/sac_final.pt` | deterministic actor mean |
| TD3 | `artifacts/policies/td3_gated_duck/td3_final.pt` | deterministic actor mean |

Teacher guidance describes how the two tabular checkpoints were trained.
Teacher code is never called while collecting explanations.

The exact shared environment is
[`shared_gated_duck_config.yaml`](../configs/explainability/shared_gated_duck_config.yaml).
All policies use the same map, seed list, spawn contract, stop/Duckie scene,
frame skip, action physics, and reward wrapper during explanation collection.

## 3. Environment

The frozen environment used Python 3.9.15. A CUDA wheel is pinned because the
continuous policies were trained with CUDA; explanation inference and tests can
run on CPU.

```bash
cd duckie-mdp
python3.9 -m venv .venv-sac
.venv-sac/bin/python -m pip install --upgrade pip
.venv-sac/bin/python -m pip install -r requirements-explainability-lock.txt
```

On a headless Linux machine, ensure the normal OpenGL/Mesa system libraries
required by gym-duckietown are installed. The runner sets
`LIBGL_ALWAYS_SOFTWARE=1` by default.

## 4. Reproduction commands

### Verify the release bundle

```bash
./scripts/reproduce_explanation_pipeline.sh verify
```

This checks:

- hashes and byte sizes for checkpoints, configs, evidence, and critical code;
- Q-table shape `(5,5,3,3,4,2,5,7)` for Q-learning and SARSA;
- all four policy adapters load the frozen checkpoint expected by the manifest;
- evaluation modes are greedy/deterministic and teacher-free;
- six paired local outcomes satisfy branch invariants;
- four primitive families cover exactly 238 temporal instances and 4,000
  explained decisions.

Expected final field:

```json
{"passed": true}
```

### Run explanation regression tests

```bash
./scripts/reproduce_explanation_pipeline.sh test
```

The focused suite covers schema/adapters, primitive labeling, deterministic
replay, paired temporal outcomes, state counterfactual validity, metamorphic
relations, exact Q enumeration, SAC diagnostics, clustering, SARSA, EDDP/C-EDDP,
and support-aware audit behavior.

### Run a four-policy smoke branch

```bash
./scripts/reproduce_explanation_pipeline.sh smoke
```

This records five real decisions per policy and computes one paired explanation
for each solver. Output goes to
`runs/explanations/four_policy_reproduction_smoke/`, which is ignored by Git.

### Recompute the complete result

```bash
./scripts/reproduce_explanation_pipeline.sh full
```

The full budget is:

```text
4 solvers × 5 seeds × 1 episode × at most 200 decisions = 4,000 anchors
```

Every anchor runs valid state counterfactuals, a paired factual/foil simulator
rollout, and verification. This is computationally expensive. Collection is
resumable: each completed result is atomically stored in `instance_shards/`.
Rerunning the same command reuses those shards.

If collection is complete but postprocessing was interrupted:

```bash
./scripts/reproduce_explanation_pipeline.sh postprocess
```

## 5. Frozen result bundle

[`artifacts/explainability/four_policy`](../artifacts/explainability/four_policy/README.md)
contains the compact paper-facing result:

- M1–M13 summary JSON;
- six local paired action outcomes;
- the four-family real-evidence JSON;
- segmentation, discovery, runtime, and aggregate reports;
- a deterministic reproducibility manifest and `SHA256SUMS`.

The original 50+ MB decision-level JSONL and generated videos are deliberately
not in Git. They are reproducible outputs under `runs/` and `videos/`.

## 6. Refreshing a release after intentional changes

Only refresh the release after an intentional code/config/checkpoint change:

```bash
.venv-sac/bin/python scripts/freeze_explanation_release.py
./scripts/reproduce_explanation_pipeline.sh verify
```

Commit the changed manifest together with every file whose hash changed. Never
refresh the manifest merely to hide an unexplained mismatch.

## 7. Interpretation boundary

- A paired rollout is a simulator-based interventional counterfactual, not an
  empirical probability and not a real-world causal guarantee.
- Reactive Duckie behavior is endogenous. Initial conditions, RNG streams,
  controller parameters, clock, and action prefix are controlled; the Duckie
  may react differently after the first action because that is part of the
  consequence being measured.
- The live dashboard video displays a real precomputed representative M1–M13
  evidence card for the active primitive family. It does not recompute a paired
  counterfactual at every video frame.
- The optional support-aware certification audit and the descriptive
  explanation-derived primitive result must not be reported as the same claim.
