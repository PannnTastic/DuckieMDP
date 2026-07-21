# M11 — Bottom-Up Primitive Discovery and Reconciliation

## Outcome

M11 is complete and eligible as a descriptive bottom-up validation result.
No policy was retrained, no checkpoint was changed, and no primitive label was
used to construct features, select hyperparameters, or assign clusters.

The result is deliberately not presented as perfect confirmation of the M2
lexicon. HDBSCAN found reproducible behavioral structure, but its agreement
with the frozen top-down primitives is moderate. Q-learning is substantially
more aligned than SAC, while rare stop/yield behaviors remain unsupported by
the five-seed clustering ensemble.

## Independence contract

The M2 lexicon was frozen before M11 in
`docs/primitive_lexicon_v1.freeze.json`. M11 consumes the matched original-policy
rollouts from M12, but it does **not** consume M12 `segments.csv`: those segment
boundaries were created from primitive-label transitions and would leak the
answer into clustering.

Instead, M11 uses non-overlapping fixed windows of five decisions. A window
boundary depends only on episode order and time:

```text
steps 0..4, 5..9, 10..14, ...
```

Each 37-dimensional signature contains only:

- lane state statistics (`d`, `phi`, speed, curvature);
- semantic stop and Duckie context without event or primitive names;
- continuous command statistics (`v_cmd`, `omega_cmd`);
- command changes, deceleration fraction, steering reversal rate.

Explicitly excluded from clustering features are primitive, trigger,
undesirable flag, reward, event/termination names, action name/id, scenario,
and solver. Solver and seed remain metadata used only for stratified reporting.

## Dataset and split

| Item | Value |
|---|---:|
| Original decision steps | 2,500 |
| Fixed behavioral windows | 500 |
| Window size | 5 decisions |
| Q-learning windows | 250 |
| SAC windows | 250 |
| Development windows | 300 |
| Held-out windows | 200 |
| Signature dimensions | 37 |

For each solver, seeds 20101, 20202, and 20303 are development; seeds 20404
and 20505 are held out. Hyperparameters are selected from development features
without opening M2 primitive labels.

## HDBSCAN primary result

Development-only unsupervised selection chose:

```text
min_cluster_size = 8
min_samples       = 5
selection_method  = eom
```

Because scikit-learn HDBSCAN has no inductive `predict` method, these parameters
are frozen first and then one final transductive fit uses all feature vectors.
Held-out primitive labels remain unopened until cluster assignment is complete.

| Diagnostic | Value |
|---|---:|
| Clusters | 11 |
| Clustered windows | 347 / 500 |
| Coverage | 69.40% |
| Noise rate | 30.60% |
| Silhouette | 0.4483 |

### Reconciliation with frozen M2 labels

| Stratum | Coverage | Purity | NMI | ARI |
|---|---:|---:|---:|---:|
| All | 69.40% | 59.94% | 0.4299 | 0.1967 |
| Development | 69.67% | 60.77% | 0.4395 | 0.1965 |
| Held-out | 69.00% | 59.42% | 0.4407 | 0.1859 |
| Q-learning | 74.00% | 70.27% | 0.6120 | 0.3737 |
| SAC | 64.80% | 48.15% | 0.2056 | 0.0914 |

Development and held-out values are close, so the moderate alignment is not
explained by a development-only overfit. It is a property of this dataset and
representation.

## K-means sensitivity

Silhouette selection on development features chose `k=9`. The model is fit on
development features and predicts all windows.

| Stratum | Coverage | Purity | NMI | ARI |
|---|---:|---:|---:|---:|
| All | 100% | 48.20% | 0.3626 | 0.1675 |
| Held-out | 100% | 48.00% | 0.3570 | 0.1559 |
| Q-learning | 100% | 53.20% | 0.4997 | 0.2903 |
| SAC | 100% | 48.40% | 0.2762 | 0.0727 |

The sensitivity analysis preserves the same qualitative conclusion: Q
behavior aligns with the discrete lexicon more strongly than SAC behavior, and
forcing every sample into a cluster increases coverage but not agreement.

## What the emergent clusters contain

Several HDBSCAN clusters have a clear top-down interpretation:

- clusters 0 and 1 are 100% `ApproachCrossing`, separated by SIDE_FAR versus
  SIDE_NEAR Duckie context;
- cluster 2 is 100% `ResumeAfterStop` and is dominated by satisfied-stop state;
- cluster 4 is 100% `CruiseCurveLeft` with high positive speed and steering;
- cluster 10 is 100% `LaneCorrectLeft` in high-heading-error SAC windows.

The clustering also exposes structure not cleanly represented by one M2 label.
Q-learning cluster 3 is dominated by high steering variance, steering
reversals, and command jerk. Its windows contain mostly `CruiseCurveLeft` and
`LaneCorrectRight`, with only the individual reversal decision labeled
`OscillatorySteering`. Thus M11 discovers a broader oscillatory behavior mode
while the top-down M2 label is a one-step transition label.

All HDBSCAN clusters are solver-specific even though solver is not a feature.
This means discrete Q commands and smooth SAC commands form different raw
behavioral modes. The shared primitive vocabulary is therefore a semantic
abstraction across solver-specific control styles, not a one-to-one partition
of raw action trajectories.

## Rare safety-context limitation

The fixed-window majority labels contain only:

- 13 `StopHold` windows;
- 10 `YieldHold` windows;
- 4 `DecelerateStop` windows.

All of these are classified as HDBSCAN noise. Therefore this run does not
provide bottom-up confirmation of the rare stop/yield primitives. It does not
invalidate the M5/M7 safety evidence or the M2 labeler; it limits the M11 claim.
A thesis-level follow-up should collect a larger scenario-stratified rollout
ensemble using stop and pedestrian scenario manifests, without sampling by
primitive label.

## Window-label ambiguity

The fixed boundaries are independent of M2 but can cross a true behavioral
transition:

| Statistic | Value |
|---|---:|
| Mean majority-label purity | 78.12% |
| Fully homogeneous windows | 207 / 500 |
| Mixed-label windows | 293 / 500 |

This is an intentional independence/temporal-resolution trade-off. Using M2
label transitions as boundaries would increase purity artificially and destroy
the validity of bottom-up reconciliation.

## Acceptance disposition

All predeclared engineering and independence checks pass:

- fixed windows do not use M2/M12 primitive boundaries;
- feature names pass the label-leakage guard;
- primitive labels are absent from HDBSCAN/K-means selection;
- development and held-out seeds are disjoint;
- the lexicon freeze predates clustering;
- HDBSCAN reruns are deterministic;
- HDBSCAN produces at least two clusters and at least 50% coverage;
- every window receives a frozen-label evaluation only after clustering.

No acceptance threshold is placed on purity, NMI, or ARI. Low alignment is a
scientific result, not an implementation failure to be tuned away using labels.

## Code and artefacts

Implementation:

- `src/explainability/signatures.py`;
- `src/explainability/cluster_primitives.py`;
- `src/explainability/reconcile_clusters.py`;
- `scripts/run_m11_bottom_up_clustering.py`.

Primary outputs:

- `runs/explanations/m11_bottom_up_clustering/m11_summary.json`;
- `signatures_unlabeled.csv`;
- `cluster_assignments_with_frozen_labels.csv`;
- HDBSCAN/K-means search and reconciliation JSON;
- confusion-matrix and cluster-profile CSV files;
- fitted scaler and clustering model files with hashes.

Reproduce:

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m scripts.run_m11_bottom_up_clustering
```

## Scientific conclusion

M11 partially validates the top-down lexicon: several concrete driving modes
emerge independently, especially for Q-learning, but the lexicon is not a
natural one-to-one clustering of raw behavior. SAC behavior is more continuous
and overlaps multiple semantic primitives; rare stop/yield modes lack density
in this small ensemble. The correct claim is **partial, solver-dependent
bottom-up support**, not universal discovery of the frozen primitive taxonomy.
