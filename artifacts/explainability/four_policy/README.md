# Frozen Four-Policy Explanation Result

This directory is the compact, Git-tracked evidence bundle for the
explanation-derived driving primitive experiment.

## Main result

The frozen real-evidence file contains:

| Primitive family | Temporal instances | Explained decisions | Solver coverage |
| --- | ---: | ---: | --- |
| LaneKeeping | 91 | 879 | Q-learning, SARSA, SAC, TD3 |
| CurveNegotiation | 95 | 2,038 | Q-learning, SARSA, SAC, TD3 |
| StopCompliance | 30 | 631 | Q-learning, SARSA, SAC, TD3 |
| PedestrianYield | 22 | 452 | Q-learning, SARSA, SAC, TD3 |
| **Total** | **238** | **4,000** | **four policies** |

Every representative evidence card includes:

- `why`: state intervention and nearest decision boundary;
- `what_if`: factual versus foil physical outcome;
- `verification`: applicable metamorphic/safety results;
- `temporal`: a human-readable behavior arc plus raw state evolution.

The authoritative machine-readable file is
[`primitive_real_evidence.json`](primitive_real_evidence.json).

## Directory contents

- `m1_m13_summaries/`: accepted compact stage summaries;
- `paired_outcomes/`: six Q-learning/SAC COViz-inspired local outcomes;
- `primitive_real_evidence.json`: four explanation-derived primitive families;
- `explanation_derived_report.json`: aggregate temporal result;
- `segmentation_summary.json`, `discovery_summary.json`, and
  `runtime_assignments.json`: downstream temporal grouping outputs;
- `reproducibility_manifest.json`: hashes over policies, evidence, and critical
  source files;
- `SHA256SUMS`: command-line integrity check.

The full decision-level JSONL is intentionally excluded because it is over
50 MB and can be regenerated. See
[`docs/explanation_reproducibility.md`](../../../docs/explanation_reproducibility.md).
