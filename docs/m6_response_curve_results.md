# M6 — Valid-Manifold State Counterfactuals and Response Curves

Status: **implemented and validated on 2026-07-20** for Q-learning and SAC.

M6 answers a different question from the paired rollouts in M5:

> Which semantic state change is sufficient to change the policy action or
> driving primitive, while all other state features remain fixed?

The result is a behavioral sensitivity explanation. It is not yet a temporal
outcome explanation; M5 provides the future consequences of an action foil.

## Where to view the explanation

The main comparison figure is available as:

```text
figures/fig_m6_response_curves.pdf
figures/fig_m6_response_curves.png
```

The complete machine-readable result is:

```text
runs/explanations/m6_response_curves/m6_summary.json
```

Every individual curve has both JSON and CSV files in the same directory. For
example:

```text
lane_q_learning_d.json
lane_sac_d.json
stop_q_learning_stop_distance.json
stop_sac_stop_distance.json
duck_q_learning_duck_longitudinal.json
duck_sac_duck_longitudinal.json
```

## Reproduce the experiment

From the repository root:

```bash
.venv-sac/bin/python -m scripts.run_m6_response_curves
.venv-sac/bin/python figures/gen_fig_m6_response_curves.py
```

The first command captures three real SAC rollout anchors, runs the same
semantic sweeps against both policies, and writes the JSON/CSV audit. The second
command produces a vector PDF and a 300-DPI PNG.

## Pipeline

```text
real rollout state
      ↓
semantic intervention on exactly one target feature
      ↓
physical/semantic manifold validation
      ↓
loss-aware projection into Q-learning or SAC representation
      ↓
solver-input validation
      ↓
deterministic policy query
      ↓
action response + driving-primitive response
```

Rejected states are retained in a separate audit and are never sent to either
policy. This prevents an invalid synthetic input from being reported as policy
behavior.

## Real rollout anchors

| Scenario | Seed | Decision step | Source |
|---|---:|---:|---|
| Lane following | 701 | 5 | SAC lane rollout |
| Stop sign | 30101 | 21 | SAC full-task rollout |
| Active Duckie | 30101 | 82 | SAC full-task rollout |

The same physical anchor is projected into each solver. Q-learning receives its
actual categorical representation; SAC receives its continuous metric state.
For example, a metric Duckie location is mapped to
`crossing_near/crossing_far/side_near/side_far/none` before querying Q-table.

## Experiment coverage

```text
Curves generated:         20
Valid-manifold queries:  114
Rejected audit queries:   22
Action flip points:       76
Primitive flip points:    60
```

Sweeps cover:

- lateral offset;
- heading error;
- curvature;
- stop distance;
- stop-hold progress;
- Duckie longitudinal and lateral position;
- Duckie present/absent;
- Duckie active/inactive;
- crossing available/unavailable.

## Main findings

### 1. Q-learning has stepwise, representation-limited behavior

At the selected lane anchor, changing curvature magnitude does not change the
Q-learning action or primitive. The Q-table only observes the three-valued road
category, not metric curvature. This is expected representational aliasing,
not a plotting failure.

For stop distance, all seven valid values from 0 to 3 m produce the same
Q-learning action at this anchor. Changing dwell progress from 0 to 0.66 also
does not change its action; only completion at 1.0 changes the semantic
primitive to `ResumeAfterStop`. This exposes the binary nature of
`sigma_stop` in the tabular state.

### 2. SAC responds continuously

Small changes in a continuous state feature change the raw actor output. The
explanation therefore uses **primitive flip**, not arbitrary floating-point
action change, as the main boundary.

At the lane anchor, moving lateral offset by approximately 0.045 m to `d=0`
changes the SAC primitive from `LaneCorrectRight` to `LaneCorrectLeft` under the
fixed remaining state. A curvature change from 0 to -1 changes the primitive to
`UnnecessaryBrake`, revealing a potentially conservative actor response worth
further testing.

### 3. Pedestrian boundaries differ between policies

Starting from `YieldHold`:

- Q-learning changes primitive at Duckie longitudinal position 0.6 m, a change
  of about 0.242 m from the anchor, and selects `UnsafeProceed` at that sampled
  state.
- SAC changes primitive at 0.9 m, a change of about 0.542 m, to
  `ApproachCrossing`.
- Removing the Duckie changes both policies away from `YieldHold`, as expected.

The Q-learning `UnsafeProceed` point is a behavioral query of the stored table,
not yet evidence of a frequently learned/reached state. The checkpoint has no
visit-count artifact, so Q support is explicitly recorded as
`unknown_no_visit_count_artifact`. M8 will stratify exhaustive Q-table results
when support evidence is available.

## Action flip versus primitive flip

Two counterfactuals are stored separately:

- `minimal_action_counterfactual`: the smallest sampled feature change that
  modifies raw action output;
- `minimal_primitive_counterfactual`: the smallest sampled feature change that
  changes human-readable driving behavior.

The second is the main explanation for SAC because almost any continuous input
perturbation can modify actor output by a tiny numerical amount.

## Acceptance rules

M6 is accepted only when:

1. every anchor comes from a real rollout;
2. every synthetic state stores its anchor hash and requested/applied changes;
3. physical validation happens before lossy solver projection;
4. rejected states never reach policy inference;
5. valid and rejected points are reported separately;
6. Q and SAC use deterministic evaluation modes;
7. Q-table shape remains `(5,5,3,3,4,2,5,7)`;
8. JSON, CSV, PDF, and PNG outputs are reproducible from saved scripts.
