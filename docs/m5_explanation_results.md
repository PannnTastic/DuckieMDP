# M5 — Paired Action-Outcome Explanations

Status: **validated on 2026-07-20** for Q-learning and SAC.

This stage answers a local RL question:

> Given the same simulator branch state, what happens when the policy's selected
> action is executed versus one predeclared contrast action (foil)?

Only the first action differs. After that first decision, both branches return
to the same deterministic evaluation policy. These are simulator-based
interventional outcomes, not real-world causal claims and not probabilities.

## Where to read the explanations

Human-readable reports are generated in:

```text
runs/explanations/q_lane_correction.txt
runs/explanations/q_stop_hold.txt
runs/explanations/q_pedestrian_yield.txt
runs/explanations/sac_lane_correction.txt
runs/explanations/sac_stop_hold.txt
runs/explanations/sac_pedestrian_yield.txt
```

Machine-readable reports with the full trajectories, per-term rewards, branch
manifest, physical outcomes, and invariants use the same names with `.json`.
The `runs/` directory remains a local generated-output directory and is ignored
by Git; the compact acceptance evidence is preserved in
`docs/m5_paired_outcomes_acceptance.json`.

From the repository root, read a report with:

```bash
sed -n '1,160p' runs/explanations/sac_pedestrian_yield.txt
```

Or inspect selected fields in JSON:

```bash
python -m json.tool runs/explanations/sac_pedestrian_yield.json | less
```

## What each report means

Each explanation includes:

1. **Selected action and primitive** — the deterministic policy decision.
2. **Contrast action and primitive** — the predeclared foil, not a cherry-picked
   action after observing the result.
3. **Temporal reward profile** — discounted and undiscounted totals, separately
   retaining every reward component.
4. **Physical outcome profile** — lane error, progress, Duckie clearance, stop
   events, steering reversals/jerk, termination reason, and primitive sequence.
5. **Branch invariants** — proof that the branch manifest is identical, only the
   first action is forced, and teacher assistance is disabled.

## Validated examples

| Solver | Scenario | Selected primitive | Foil primitive | 30-step discounted return: selected / foil |
|---|---|---|---|---:|
| Q-learning | lane correction | `LaneCorrectLeft` | `CruiseCurveRight` | `1.913 / 1.837` |
| Q-learning | stop sign | `StopHold` | `DecelerateStop` | `18.054 / 18.007` |
| Q-learning | pedestrian | `YieldHold` | `YieldDecelerate` | `3.117 / 3.191` |
| SAC | lane correction | `LaneCorrectLeft` | `CruiseStraight` | `-33.688 / -33.246` |
| SAC | stop sign | `StopHold` | `DecelerateStop` | `12.358 / 12.093` |
| SAC | pedestrian | `YieldHold` | `PrematureResume` | `-0.583 / -10.409` |

A lower selected return in one finite branch is retained as a negative result;
it is not rewritten as policy success. In particular, the Q-learning pedestrian

All three Q-learning cases use the same frozen full-task checkpoint. The lane
case uses the lane-only environment config only to produce a clean lane
correction branch; it does not load the lane-only Q-table. All three SAC cases
likewise use the same frozen full-task SAC checkpoint.
foil and the SAC lane foil slightly outperform the selected first action over
this 30-decision horizon. The SAC pedestrian contrast is the clearest safety
case: forcing a premature resume is about 9.83 discounted-return units worse.

## Validation outcome

All six paired reports satisfy:

```text
same_manifest = true
same_policy_selected_action_at_branch = true
only_first_action_forced = true
selected_and_foil_differ = true
teacher_active = false
single_rollout_is_probability = false
```

During validation, the primitive labeler was revised from v1.0.0 to v1.0.1:
an unsatisfied stop obligation now outranks a merely inactive/side Duckie,
while an active pedestrian corridor risk retains higher safety precedence.
The lexicon content hashes were re-frozen after the change.
