# Support-Aware C-EDDP Certification Correction

## Status

This document records the support-stratification correction applied after the
first four-policy C-EDDP run. The corrected artefacts are in:

```text
runs/explanations/cedp_v2_4policy_support_aware/
```

The original directory is preserved for audit comparison:

```text
runs/explanations/cedp_v2_4policy/
```

## Why the correction was necessary

M7 and M8 already distinguish supported, reachable, valid-manifold, and unseen
regions. The first C-EDDP collection path did not carry that distinction into
its local metamorphic verifier. In addition, the local instance payload set:

```text
supported_or_reachable_state = true
```

without measured support evidence.

That meant an applicable metamorphic relation could be evaluated on a
semantically valid but empirically unsupported target, and the result was
aggregated as an ordinary pass or failure.

The corrected contract is:

```text
source supported + target supported
    -> relation is eligible for a PASS/FAIL claim

source/target outside empirical support
    -> ABSTAIN

invalid counterfactual target
    -> ABSTAIN
```

An abstention is neither a pass nor a failure. It says that this run does not
contain enough empirical support to make the relation claim.

## Meaning of support

### Tabular Q-learning and SARSA

Support is counted exactly in the discrete Q-table cell representation using
the frozen 4,000-anchor C-EDDP evaluation population.

```text
count >= 3  -> evaluation_supported
count 1..2  -> reachable
count = 0   -> unseen
```

This is **evaluation support**, not training visitation. No training-visit
artefact was available, so the code and reports must not call it training
support.

### Continuous SAC and TD3

Support is measured in the actor observation space:

1. states are first grouped by semantic mode flags such as stop presence,
   Duckie presence, Duckie activity, and crossing availability;
2. within the same semantic group, nearest-neighbour distance is computed;
3. the support radius is frozen from the anchor population;
4. the target is supported only when it lies inside that radius and the group
   contains enough observations.

This prevents an absent-Duckie state from supporting an intervention that
creates an active crossing Duckie.

## Code map

| File | Responsibility |
| --- | --- |
| `src/explainability/eddp/support.py` | Builds the frozen support oracle and classifies source/target states. |
| `src/explainability/eddp/verification.py` | Produces `PASS`, `FAIL`, `ABSTAIN`, or `NOT_APPLICABLE` per relation. |
| `src/explainability/certified_primitives/collection.py` | Measures source support; no hardcoded local support gate. |
| `src/explainability/certified_primitives/certificate_adapter.py` | Converts measured support into the local binding certificate and preserves its evidence. |
| `src/explainability/certified_primitives/certificate_checker.py` | Aggregates only eligible relation pairs and requires nonzero property evidence. |
| `scripts/run_cedp_reprofile.py` | Recomputes support-aware profiles while reusing expensive paired outcomes. |
| `scripts/run_cedp_support_audit.py` | Produces the reproducible support/status audit JSON. |
| `configs/explainability/cedp_v2_4policy.yaml` | Freezes support thresholds and the minimum property-evidence gate. |

## Reprofile procedure

The correction does not retrain policies and does not rerun simulator
counterfactual branches. It reuses the stored factual/foil outcomes, loads all
4,000 real anchors, then recomputes state counterfactual and metamorphic
profiles.

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_reprofile.py \
  --config configs/explainability/cedp_v2_4policy.yaml \
  --input-dir runs/explanations/cedp_v2_4policy \
  --output-dir runs/explanations/cedp_v2_4policy_support_aware
```

The downstream stages are then rerun on the corrected instances:

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_segment.py \
  --config configs/explainability/cedp_v2_4policy.yaml \
  --output-dir runs/explanations/cedp_v2_4policy_support_aware

PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_discovery.py \
  --config configs/explainability/cedp_v2_4policy.yaml \
  --output-dir runs/explanations/cedp_v2_4policy_support_aware

PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_certify.py \
  --config configs/explainability/cedp_v2_4policy.yaml \
  --output-dir runs/explanations/cedp_v2_4policy_support_aware
```

## Corrected local-verification results

Each solver contributes 1,000 anchors. All 4,000 source anchors remain locally
certified because they are real rollout states and retain valid paired-outcome
evidence.

### Q-learning

| Relation | PASS | FAIL | ABSTAIN | NOT_APPLICABLE |
| --- | ---: | ---: | ---: | ---: |
| Stop | 6 | 0 | 3 | 991 |
| Pedestrian | 189 | 0 | 731 | 80 |
| Curvature | 225 | 0 | 15 | 760 |
| Lane symmetry | 197 | 0 | 43 | 760 |

### SARSA

The observed status counts are identical to Q-learning for this frozen
evaluation population:

| Relation | PASS | FAIL | ABSTAIN | NOT_APPLICABLE |
| --- | ---: | ---: | ---: | ---: |
| Stop | 6 | 0 | 3 | 991 |
| Pedestrian | 189 | 0 | 731 | 80 |
| Curvature | 225 | 0 | 15 | 760 |
| Lane symmetry | 197 | 0 | 43 | 760 |

This is the important correction: supported pedestrian pairs for both tabular
policies have 189 passes and zero failures. The 731 abstentions are retained
instead of being collapsed into an unsupported verdict.

### SAC

| Relation | PASS | FAIL | ABSTAIN | NOT_APPLICABLE |
| --- | ---: | ---: | ---: | ---: |
| Stop | 5 | 0 | 0 | 995 |
| Pedestrian | 0 | 0 | 975 | 25 |
| Curvature | 23 | 136 | 90 | 751 |
| Lane symmetry | 0 | 0 | 252 | 748 |

### TD3

| Relation | PASS | FAIL | ABSTAIN | NOT_APPLICABLE |
| --- | ---: | ---: | ---: | ---: |
| Stop | 0 | 0 | 0 | 1000 |
| Pedestrian | 0 | 0 | 978 | 22 |
| Curvature | 61 | 44 | 42 | 853 |
| Lane symmetry | 0 | 2 | 145 | 853 |

For SAC and TD3, most pedestrian targets are outside local empirical support.
The correct result is therefore abstention, not a claim that pedestrian
monotonicity passed or failed.

The machine-readable source for these tables is:

```text
runs/explanations/cedp_v2_4policy_support_aware/support_aware_audit.json
```

## Primitive-level result

After rebuilding the temporal pipeline:

```text
instances                  4000
trajectories                 20
temporal segments           416
discovered clusters          16
cluster coverage         0.6058
bootstrap stability      0.3242
outcome coherence ratio  0.4367
```

All 16 clusters are currently:

```text
PRIMITIVE_CANDIDATE
```

The failed-gate counts are:

| Gate | Number of clusters failing |
| --- | ---: |
| Bootstrap stability | 16 |
| Claimed properties pass | 8 |
| Minimum support | 7 |
| Minimum property evidence | 5 |

The binding bootstrap threshold remains `0.70`. It was not lowered to recover
the previous certificate count. Therefore the support-aware run currently has
**no certified primitive cluster**.

The former result of eight certified primitive clusters is retained only as a
pre-correction comparison. It must not be presented as the final
support-stratified certification result.

## Interpretation

The correction fixed the original pedestrian-verification problem for the
supported tabular region: Q-learning and SARSA have no supported pedestrian
violations in this population.

It also exposed a different blocker: once abstention is represented honestly,
the explanation signatures and discovered clusters are not stable enough
under bootstrap resampling. That is a discovery/data-support problem, not a
reason to weaken the certification gate.

The defensible next experiment is to increase balanced support for stop and
pedestrian modes—especially continuous-policy active-Duckie targets—then rerun
discovery with the thresholds still frozen.

## Validation

Focused explainability tests:

```text
106 passed
```

The new regression tests cover:

- both source and target supported -> claimable PASS/FAIL;
- supported source with unsupported target -> ABSTAIN;
- missing support oracle -> ABSTAIN;
- local source support is measured rather than hardcoded.
