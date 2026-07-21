# M7 — Metamorphic Policy Testing Results

## Outcome

M7 is complete. The implementation evaluates four frozen domain relations on
the teacher-free greedy Q policy and deterministic SAC actor mean:

- `MR-STOP`: a closer unsatisfied stop line must not increase speed;
- `MR-PEDESTRIAN`: greater pedestrian risk must not increase speed or produce
  an unsafe proceed primitive;
- `MR-CURVATURE`: greater absolute curvature must not increase speed;
- `MR-LANE-SYMMETRY`: mirrored lane errors should produce mirrored steering.

This is **LEGIBLE-inspired metamorphic policy testing**, not a claim that the
full LEGIBLE algorithm was reproduced. Each result is an explanation over a
pair of states: a valid source state and one valid counterfactual target state.
The aggregated pass/fail rates characterize a wider policy behavior.

## Validation gates

- Real rollout anchor: M6 lane anchor, seed 701, decision step 5.
- Generated source/target pairs: 48.
- Semantically and numerically valid pairs: 48/48.
- Pairs satisfying their formal preconditions: 48/48.
- `NOT_APPLICABLE`: 0.
- Rejected state policy queries: 0 by construction and unit test.
- Full regression: 120 tests passed.
- Q-table shape remains `(5,5,3,3,4,2,5,7)` through the existing regression.

A policy-level `FAIL` is retained as a scientific finding. It does not make the
M7 pipeline fail acceptance, provided the pair is valid, applicable, and the
comparison follows the frozen solver-specific rule.

## Results

| Solver | Relation | PASS | FAIL | Pass rate |
|---|---:|---:|---:|---:|
| Q-learning | MR-STOP | 6 | 0 | 100.0% |
| Q-learning | MR-PEDESTRIAN | 6 | 0 | 100.0% |
| Q-learning | MR-CURVATURE | 6 | 0 | 100.0% |
| Q-learning | MR-LANE-SYMMETRY | 3 | 3 | 50.0% |
| SAC | MR-STOP | 6 | 0 | 100.0% |
| SAC | MR-PEDESTRIAN | 5 | 1 | 83.3% |
| SAC | MR-CURVATURE | 3 | 3 | 50.0% |
| SAC | MR-LANE-SYMMETRY | 1 | 5 | 16.7% |

Overall, Q-learning passes 21/24 pairs (87.5%) and SAC passes 15/24 pairs
(62.5%). These numbers are not task success rates; they are consistency rates
for the selected metamorphic test suite.

## Notable findings

1. Both policies satisfy all tested stop-distance monotonicity pairs.
2. Q-learning satisfies all tested pedestrian-risk and curvature pairs.
3. SAC increases speed by about 0.051 command units for the intervention
   `side_far -> side_near`, producing the only SAC pedestrian failure.
4. SAC increases speed for positive curvature interventions `0 -> +1`,
   `0 -> +2`, and `0 -> +4`; the corresponding negative-curvature tests pass.
   This is directional policy asymmetry, not evidence that the curvature
   feature is unused.
5. Lane symmetry is the weakest property for both policies. SAC passes only
   one of six mirrored pairs, with several pairs retaining positive steering
   after the lane error is mirrored. Q-learning passes three of six; one
   failure mirrors steering direction but changes the speed macro-action.

## Scope and support caveat

The anchor was observed in a real rollout, but the intervened states are valid
synthetic states and are not automatically proven reachable. Q-learning visit
counts are unavailable for this checkpoint, so its pairs retain
`unknown_no_visit_count_artifact`. SAC pairs retain
`real_anchor_synthetic_pairs_not_reachability_proven`. Consequently, these
results are controlled policy probes, not exhaustive deployment guarantees.

M8 will add exact Q-table characterization and the
representable/valid/reachable/supported strata needed for stronger global
claims.

## Reproduce and inspect

Run:

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m scripts.run_m7_metamorphic
```

Inspect:

- `runs/explanations/m7_metamorphic/m7_summary.json` — aggregate result;
- `runs/explanations/m7_metamorphic/m7_metamorphic_results.json` — full
  source state, target state, decisions, primitives, measurements, and
  provenance for each explanation pair;
- `runs/explanations/m7_metamorphic/m7_metamorphic_results.csv` — compact
  table for analysis.

