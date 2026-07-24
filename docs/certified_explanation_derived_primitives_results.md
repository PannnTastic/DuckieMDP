# C-EDDP v2 Implementation and Pilot Results

## Status

The C-EDDP implementation is complete through CEDP10 and passed its
explainability regression suite. A deliberately small five-seed pilot was run
to validate the complete data path. It is **not** the final paper experiment:
the configured main budget is at least 1,000 certified explanation instances
per solver and broader stop/pedestrian/curve coverage.

## Main-input contract

The main pipeline uses newly computed, full-trajectory M1--M13 explanation
instances:

```text
real policy decision
  -> M1--M13 state counterfactual
  -> paired factual/foil simulator outcome
  -> metamorphic verification
  -> certified explanation instance
  -> temporal explanation signature
  -> discovered primitive
  -> primitive-level certificate
```

EDDP v1 atoms and cluster labels are not consumed by main discovery. They are
retained only as a sparse-anchor baseline and schema migration smoke test.

## Implementation map

| Milestone | Implementation |
|---|---|
| CEDP0 | `certified_primitives/provenance.py`, `run_cedp_freeze.py` |
| CEDP1 | `certificate_adapter.py`, `collection.py` |
| CEDP2 | `trajectory.py`, `run_cedp_collect.py` |
| CEDP3 | `signature.py` |
| CEDP4 | `segmentation.py`, `run_cedp_segment.py` |
| CEDP5 | `discovery.py`, `run_cedp_discovery.py` |
| CEDP6 | `descriptor.py` |
| CEDP7 | `certificate_checker.py`, `run_cedp_certify.py` |
| CEDP8 | `reconciliation.py`, `run_cedp_reconcile.py` |
| CEDP9 | `runtime.py`, `run_cedp_runtime.py` |
| CEDP10 | `reporting.py`, `run_cedp_ablation.py`, `run_cedp_report.py` |

## Pilot protocol

- Policies: teacher-free greedy Q-learning, teacher-free greedy SARSA, and
  deterministic actor-mean SAC.
- Seeds: `20101`, `20202`, `20303`, `20404`, `20505`.
- Decisions: six per policy/seed.
- Full explanation instances: 90 total, 30 per solver.
- Certified local instances: 90; abstained: 0; failed: 0.
- Segmentation used for the pilot discovery: fixed window of three decisions.
- Development seeds: first three per solver; held-out seeds: final two.
- M2 was opened only after the cluster assignment freeze.

## Pilot results

Discovery produced two clusters:

| Descriptor | Support | Solver support | Final pilot status |
|---|---:|---|---|
| `LaneRecovery_C00` | 6 | Q-learning, SARSA | `PRIMITIVE_CANDIDATE` |
| `LaneRecovery_C01` | 7 | SAC | `PRIMITIVE_CANDIDATE` |

Neither cluster was promoted to a certified primitive:

- C00 failed minimum support and had no held-out assignment;
- C01 failed minimum support;
- therefore the runtime assigner returned `UNKNOWN` for all 30 pilot segments.

This is the intended behavior of the primitive certificate checker. Local
explanation validity does not automatically certify a cluster.

Discovery diagnostics:

| Metric | Value |
|---|---:|
| Development clusters | 2 |
| Development coverage | 0.6667 |
| Held-out coverage | 0.0833 |
| Overall coverage | 0.4333 |
| Development silhouette | 0.2657 |
| Bootstrap ARI | 0.7083 |
| Outcome coherence ratio | 0.7645 |

Post-freeze reconciliation with the manually frozen M2 taxonomy:

| Metric | Value |
|---|---:|
| Covered-segment purity | 0.8462 |
| NMI | 0.8245 |
| ARI | 0.7903 |

These reconciliation numbers are diagnostic only; M2 did not change the
clusters or their certificate status.

## Ablation result

At this pilot size:

- fixed-window 3 and 5 discovery ran successfully;
- full, decision-only, outcome-only, without-verification, and
  without-temporal feature variants ran successfully;
- primary change-point segmentation did not yield two development clusters.

The negative change-point result is retained. It indicates that six decisions
per episode are insufficient to evaluate explanation change points; it is not
a reason to silently replace the main method.

## Validation

The combined explanation regression run completed with:

```text
103 passed, 1 third-party deprecation warning
```

Tests cover schema gates, leakage resistance, full-versus-legacy isolation,
trajectory gaps, deterministic segmentation, development/held-out separation,
primitive certification, runtime abstention, and the previous M1--M13 suite.

## Reproduction

Use the SAC-compatible environment and expose the repository root:

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_freeze.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_collect.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_segment.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_discovery.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_certify.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_reconcile.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_runtime.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_ablation.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_cedp_report.py
```

Pilot artifacts are under `runs/explanations/cedp_v2_pilot/`. The canonical
configuration is `configs/explainability/cedp_v2.yaml`.

## Remaining main experiment

The implementation is ready, but the paper-scale run remains pending. It must:

1. reach the frozen budget of at least 1,000 certified instances per solver;
2. include longer episodes with curve, stop, and repeated-pedestrian phases;
3. use primary change-point segmentation;
4. run the human-readable naming protocol after cluster freeze;
5. add TD3 before making the four-policy claim in the abstract.

---

## C-EDDP v2.1 — Main-scale run with corrected derivation (2026-07-22)

The 3,000-instance main collection (1,000 per solver, 5 seeds x 200 decisions)
completed under `runs/explanations/cedp_v2/`. Auditing the automatic
post-processing exposed three derivation defects; all three were fixed without
recollecting any rollout. The v2 artifacts are preserved as the pre-fix record.

### Fixes

1. **Counterfactual repair gap** (`src/explainability/counterfactual.py`):
   forcing `stop_satisfied=False` left `stop_hold_progress=1.0`, so the
   validator rejected the intervention and every ResumeAfterStop decision
   abstained. The repair now resets the hold progress; regression test
   `test_unsatisfied_stop_intervention_resets_full_hold_progress`.
2. **Pooled change-point threshold** (`scripts/run_cedp_segment.py`): one
   global distance threshold was dominated by the discrete solvers (24% of
   Q/SARSA steps exceeded it versus 0.4% for SAC), leaving SAC with 11
   segments. Segmentation models are now fitted per solver.
3. **Descriptor naming bias**
   (`src/explainability/certified_primitives/descriptor.py`): naming now uses
   per-domain evidence scores compared against the segment population
   (z-scores), with `MixedBehavior`/`ProgressRegulation` fallbacks, instead of
   raw feature sums that always favoured LaneRecovery.

### Corrected dataset (`runs/explanations/cedp_v2_1/`)

`scripts/run_cedp_reprofile.py` recomputed the simulator-free evidence
(state-counterfactual and verification profiles) from stored anchors while
reusing all paired rollouts. All 344 previous abstentions (144 Q, 144 SARSA,
56 SAC) were caused by the repair bug and are now CERTIFIED; no instance
regressed. Result: 3,000/3,000 certified, 15 full trajectories.

### Main results

| Variant | Segments (Q/SARSA/SAC) | Clusters | Coverage | Bootstrap ARI | Outcome coherence | Status |
|---|---|---|---|---|---|---|
| change-point (primary) | 123/124/85 | 2 | 94.0% | 0.176 | 0.971 | 2 candidates, 0 certified |
| fixed-window-3 (baseline, `runs/explanations/cedp_v2_1_fixed3/`) | 1,005 total | 40 | 68.4% | 0.983 | 0.410 | **11 CERTIFIED + 2 SOLVER_SPECIFIC** |

M2 reconciliation after freeze: change-point purity 0.260; fixed-window
purity 0.742, NMI 0.538. Certified fixed-window primitives map to coherent
M2 behaviours (for example a pure StopHold cluster now named
StopCompliance_C07, and a pure CruiseCurveLeft cluster named
CurveFollowing_C36). Runtime on fixed-window segments: 213
CERTIFIED_PRIMITIVE, 259 SOLVER_SPECIFIC, 533 UNKNOWN.

### Honest caveats

- The first certified primitives come from the fixed-window baseline, not the
  primary change-point method; at main scale, change-point segmentation
  produces coarse unstable clusters (RQ3 currently favours fixed windows).
- Label-free descriptor names align with the external M2 audit only
  partially (roughly half of certified clusters); certificate statuses are
  computed by the checker and are independent of names.
- Verification-relation applicability is a poor proxy for behaviour phases
  (a held stop is outside the stop relation precondition) and is no longer
  used to gate naming on the population path.

Full regression suite: 174 passed.

---

## C-EDDP v2.2 — Gated 4-policy comparison, explanation-derived naming, first certified primitive (2026-07-23)

### Fourth policy (TD3) and the gated set

TD3 was trained as a fourth continuous policy under a duck-detection gate
(`duck_detection_range 1.20`, forward corridor `0.60` — parity with the
tabular `classify_duck`) and consolidated to drive + stop + yield
(held-out: task success 1.0, stop compliance 1.0, yield 0.86). The gated
C-EDDP run `runs/explanations/cedp_v2_gated/` collected 2000 certified
explanations (1000 SAC-gated + 1000 TD3) under identical gated observation.
Q-learning and SARSA (tabular, inherently corridor-gated via `classify_duck`)
were reused from `runs/explanations/cedp_v2_1/`, giving a 4-policy set of 4000
certified explanations.

### Faithfulness — the primitives are genuinely explanation-derived

The discovery signature is explanation-only by construction
(`assert_explanation_only_contract` rejects `state`/`action`/`context`
tokens). Empirically, a classifier on the 102-dim explanation signature
alone (no raw state) predicts the behaviour family (StopSign / Pedestrian /
Curve / Lane) at **0.952 balanced accuracy** (5-fold) versus a 0.393
majority baseline. The explanation encodes the primitive, so mapping a new
explanation to its primitive from its signature is legitimate.

### Explanation-derived naming

Naming was moved off raw state onto the explanation signature. A shallow tree
over explanation-only features reaches **0.91 balanced accuracy** and yields
transparent rules now wired into `descriptor.py`:

| Name | Explanation-signature trigger (checked in this order) |
|---|---|
| LaneKeeping | `verification.lane_symmetry_applicable` (straight-lane relation applies) |
| PedestrianYield | continuous: `outcome.factual_minimum_duck_clearance` ≤ 0.40 m (measured near-miss); **or** any representation: `stop_satisfied_flip` **and** `verification.pedestrian_applicable` co-active (a stop the Duckie triggered) |
| StopCompliance | `stop_satisfied_flip` with the pedestrian relation idle (a stop-line obligation) |
| CurveNegotiation | none of the above (sustained steering) |

The pedestrian rule is **representation-aware** (framing B). A continuous-action
policy exposes a metric near-miss clearance; a tabular policy has only a
categorical Duckie and yields by *stopping*, so its yield surfaces as a stop
whose pedestrian metamorphic relation is co-active — the discriminator is the
verification relation, never raw state. This recovers pedestrian yielding for
Q-learning and SARSA, which the earlier metric-only rule missed.
`descriptor_uses_context_or_m2` remains `false`.

### Certification breakthrough — PCA and the 4-policy certified catalogue

The bootstrap-stability gate had stalled at 0.70 because the ~600-dim
aggregated segment features made HDBSCAN unstable under resampling. Adding a
**PCA projection** to `discovery.py` before clustering (coherence still
measured on the named full features; centroids, held-out assignment, and
runtime all operate in the subspace, so the space stays explanation-derived)
lifts bootstrap stability well above the gate. The optimal component count is
dataset-dependent (2-policy: 15 → 0.924; 4-policy: 20 → 0.912).

The full **4-policy** run `runs/explanations/cedp_v2_4policy/` (4000 certified
explanations, 1000 each for Q-learning, SARSA, SAC-gated, TD3) yields 415
segments, 23 clusters, bootstrap 0.912, coherence 0.475, and **8
`CERTIFIED_DRIVING_PRIMITIVE`s** (15 remain candidate, mostly on support):

| Primitive | Support | Seeds | Solvers |
|---|---|---|---|
| PedestrianYield_C07 | 30 | 10 | SAC-gated + TD3 |
| StopCompliance_C05 | 13 | 9 | Q-learning + SARSA |
| LaneKeeping_C16 | 17 | 10 | Q-learning + SARSA |
| LaneKeeping_C17 | 17 | 10 | Q-learning + SARSA |
| CurveNegotiation_C14 | 17 | 10 | Q-learning + SARSA |
| CurveNegotiation_C21 | 15 | 9 | Q-learning + SARSA |
| CurveNegotiation_C22 | 14 | 8 | Q-learning + SARSA |
| CurveNegotiation_C13 | 13 | 9 | Q-learning + SARSA |

Runtime assigns 137 of 415 segments to a certified primitive and abstains
(UNKNOWN) on 278. Each certified primitive is **shared within a representation
family** — the tabular pair (Q + SARSA) or the continuous pair (SAC + TD3) —
because the explanation signatures cluster by representation. The two
obligation/hazard-response primitives each certify in one family:
**PedestrianYield certifies for the continuous pair** (C07) and
**StopCompliance for the tabular pair** (C05). PedestrianYield is additionally
*discovered* as a cluster in both families (continuous C03/C07, tabular
C04/C06), though only the continuous one clears every certification gate. These
are the first genuinely *Certified Explanation-Derived Driving Primitives*.

### Tooling

`scripts/cedp_explanation_naming.py` (naming rules + faithfulness test),
`scripts/cedp_macro_primitives.py` and `scripts/cedp_legible.py`
(presentation overlays), and the gated configs
(`configs/explainability/cedp_v2_gated.yaml`, `shared_gated_duck_config.yaml`).

Full regression suite: 179 passed.

## C-EDDP v2.3 — claim levels and representation-aware pedestrian yield (2026-07-23)

### Three distinct result levels (do not conflate)

Earlier phrasing risked calling all 415 segments "certified". They are not. The
pipeline produces three separate levels, and the word *certified* applies only
to the last:

| Level | Result |
|---|---|
| Local M1–M13 explanations valid | 4000 / 4000 |
| Temporal segments given a semantic **family** label | 415 |
| Segments inside an HDBSCAN **cluster** | 318 / 415 (76.6%) |
| Noise / outside any cluster | 97 |
| Runtime assignment to a **certified** primitive | 137 / 415 |
| UNKNOWN / abstained | 278 |
| **Certified** primitive clusters | 8 of 23 |

So the four names (LaneKeeping 78, StopCompliance/CurveNegotiation, PedestrianYield)
are legitimately **explanation-derived driving-primitive families**; only the 8
clusters that clear every gate — and the 137 segments runtime-assigned to them —
are *certified*. The rest are family assignments on candidate or noise segments.

### Representation-aware pedestrian yield (framing B)

The metric-only pedestrian rule found zero tabular yield because tabular rollouts
never emit a metric clearance (sentinel `available=0`, `clearance_m=2.0`). But the
tabular explanation *does* carry pedestrian reasoning: `duck_risk_flip` is a live
counterfactual lever (mean 0.71) and the pedestrian metamorphic relation
applies/passes in ~85% of segments. Q-learning and SARSA simply yield **by
stopping** — so their yield surfaces as a stop whose pedestrian relation is
co-active. Splitting tabular stop segments by that relation separates them
cleanly (25 duck-triggered vs 19 stop-line, `pedestrian_pass` 0.53 vs 0.07).
Framing B keys PedestrianYield on this relation, recovering yield across all four
policies at the cluster level (continuous C03/C07, tabular C04/C06).

### Honest scope of the tabular↔continuous difference

That pedestrian yield *certifies* only for the continuous policies is stated as
**representation-dependent separability, not a proven single cause.** Two factors
act together and were not disentangled:

1. **Observation.** Q/SARSA receive a categorical Duckie threat, not a continuous
   clearance, so there is no metric to isolate a near-miss.
2. **Counterfactual action-flip definition.** A discrete `argmax` flips trivially,
   so `heading_flip` (1.00) and `lateral_flip` (0.99) saturate and the duck is
   never the strict top lever (0 / 247 tabular segments); a continuous action must
   cross a numeric tolerance to register a flip.

The absence of a *free-standing, certified* tabular PedestrianYield cluster is
therefore consistent with the interaction between the tabular representation and
the explainer's flip mechanism — the richer geometric pedestrian representation
and less-saturated action sensitivity of SAC/TD3 — rather than proof that the
state representation is the sole cause.

> Paper-safe summary: Four explanation-derived temporal driving-primitive
> families were identified across 415 segments. Lane keeping, stop compliance,
> and curve negotiation were represented across all four policies, whereas
> pedestrian yielding formed a *separable, certified* primitive only for
> continuous-action policies. This difference is consistent with the richer
> geometric pedestrian representation and less-saturated counterfactual action
> sensitivity of SAC and TD3.

Naming wired in `descriptor.py::_explanation_derived_name`; regression covers the
tabular duck-triggered-stop case (`tests/test_certified_primitives.py`).
