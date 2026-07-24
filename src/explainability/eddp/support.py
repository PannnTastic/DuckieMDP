"""Empirical support provenance for C-EDP explanation interventions.

The support contract deliberately distinguishes three questions:

* Was the source state observed in the frozen policy rollout?
* Is the source/target state sufficiently represented by the frozen anchors?
* Is a counterfactual merely semantically valid but outside empirical support?

Tabular support is measured exactly in the finite state representation by
counting discrete cells in the C-EDP anchor set. Continuous support is a local
nearest-neighbour test in the actor observation space, conditioned on the
semantic object-presence flags so that an absent-object state cannot support a
present-object intervention.

These are evaluation-support measures. They must never be described as
training visitation unless an actual training visitation artefact is supplied.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..schema import CanonicalState, TABULAR_SOLVERS


_EXACT_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class SupportEvidence:
    """Support classification for one policy state."""

    solver: str
    stratum: str
    basis: str
    reachable: bool
    supported: bool
    observation_count: int = 0
    nearest_distance: Optional[float] = None
    support_radius: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ContinuousGroup:
    matrix: np.ndarray
    radius: float
    observations: int


def _solver_name(policy: Any) -> str:
    kind = getattr(policy, "solver_kind", None)
    if kind is None:
        raise TypeError("policy has no solver_kind")
    return str(getattr(kind, "value", kind))


def _semantic_group(state: CanonicalState) -> Tuple[int, ...]:
    """Condition continuous support on discrete semantic mode flags."""

    return (
        int(bool(state.stop_present)),
        int(bool(state.stop_satisfied)),
        int(bool(state.duck_present)),
        int(bool(state.duck_active)) if state.duck_active is not None else -1,
        (
            int(bool(state.duck_crossing_available))
            if state.duck_crossing_available is not None
            else -1
        ),
    )


def _continuous_radius(
    matrix: np.ndarray,
    quantile: float,
    multiplier: float,
) -> float:
    """Estimate a conservative local support radius without model fitting."""

    unique = np.unique(np.asarray(matrix, dtype=np.float64), axis=0)
    if len(unique) < 2:
        return _EXACT_TOLERANCE
    nearest = []
    for index, row in enumerate(unique):
        distance = np.linalg.norm(unique - row, axis=1)
        distance[index] = np.inf
        nearest.append(float(np.min(distance)))
    radius = float(np.quantile(np.asarray(nearest), float(quantile)))
    return max(_EXACT_TOLERANCE, radius * float(multiplier))


class SupportOracle:
    """Classify source and counterfactual states using frozen rollout support."""

    def __init__(
        self,
        tabular_counts: Mapping[str, Counter],
        continuous_groups: Mapping[str, Mapping[Tuple[int, ...], _ContinuousGroup]],
        *,
        tabular_support_threshold: int = 3,
        continuous_minimum_group_size: int = 3,
    ) -> None:
        if int(tabular_support_threshold) < 1:
            raise ValueError("tabular_support_threshold must be positive")
        if int(continuous_minimum_group_size) < 1:
            raise ValueError("continuous_minimum_group_size must be positive")
        self.tabular_counts = {
            str(name): Counter(values)
            for name, values in tabular_counts.items()
        }
        self.continuous_groups = {
            str(name): dict(values)
            for name, values in continuous_groups.items()
        }
        self.tabular_support_threshold = int(tabular_support_threshold)
        self.continuous_minimum_group_size = int(
            continuous_minimum_group_size
        )

    @classmethod
    def from_anchors(
        cls,
        anchors: Sequence[Any],
        policies: Mapping[str, Any],
        *,
        tabular_support_threshold: int = 3,
        continuous_support_quantile: float = 0.95,
        continuous_radius_multiplier: float = 1.25,
        continuous_minimum_group_size: int = 3,
    ) -> "SupportOracle":
        if not 0.0 < float(continuous_support_quantile) <= 1.0:
            raise ValueError("continuous_support_quantile must be in (0, 1]")
        if float(continuous_radius_multiplier) <= 0.0:
            raise ValueError("continuous_radius_multiplier must be positive")

        tabular: Dict[str, Counter] = defaultdict(Counter)
        continuous_rows: Dict[
            str, Dict[Tuple[int, ...], list]
        ] = defaultdict(lambda: defaultdict(list))
        for anchor in anchors:
            solver = str(getattr(anchor.solver, "value", anchor.solver))
            if solver not in policies:
                raise KeyError("anchor solver has no policy: %s" % solver)
            decision = policies[solver].decide(anchor.state)
            if decision.solver in TABULAR_SOLVERS:
                key = tuple(
                    int(value)
                    for value in decision.diagnostics["discrete_state"]
                )
                tabular[solver][key] += 1
            else:
                vector = np.asarray(
                    decision.diagnostics["observation"], dtype=np.float64
                )
                if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                    raise ValueError("invalid continuous support observation")
                continuous_rows[solver][_semantic_group(anchor.state)].append(
                    vector
                )

        groups: Dict[str, Dict[Tuple[int, ...], _ContinuousGroup]] = {}
        for solver, by_group in continuous_rows.items():
            groups[solver] = {}
            for semantic_group, rows in by_group.items():
                matrix = np.asarray(rows, dtype=np.float64)
                matrix.setflags(write=False)
                groups[solver][semantic_group] = _ContinuousGroup(
                    matrix=matrix,
                    radius=_continuous_radius(
                        matrix,
                        continuous_support_quantile,
                        continuous_radius_multiplier,
                    ),
                    observations=len(matrix),
                )
        return cls(
            tabular,
            groups,
            tabular_support_threshold=tabular_support_threshold,
            continuous_minimum_group_size=continuous_minimum_group_size,
        )

    def classify(self, policy: Any, state: CanonicalState) -> SupportEvidence:
        decision = policy.decide(state)
        solver = str(getattr(decision.solver, "value", decision.solver))
        if decision.solver in TABULAR_SOLVERS:
            key = tuple(
                int(value)
                for value in decision.diagnostics["discrete_state"]
            )
            count = int(self.tabular_counts.get(solver, Counter()).get(key, 0))
            reachable = count > 0
            supported = count >= self.tabular_support_threshold
            return SupportEvidence(
                solver=solver,
                stratum=(
                    "evaluation_supported"
                    if supported
                    else "reachable"
                    if reachable
                    else "unseen"
                ),
                basis=(
                    "cedp_anchor_evaluation_count>=%d"
                    % self.tabular_support_threshold
                ),
                reachable=reachable,
                supported=supported,
                observation_count=count,
            )

        group = self.continuous_groups.get(solver, {}).get(
            _semantic_group(state)
        )
        if group is None:
            return SupportEvidence(
                solver=solver,
                stratum="interventional_only",
                basis="continuous_semantic_group_absent",
                reachable=False,
                supported=False,
            )
        vector = np.asarray(
            decision.diagnostics["observation"], dtype=np.float64
        )
        distances = np.linalg.norm(group.matrix - vector, axis=1)
        nearest = float(np.min(distances))
        exact_count = int(np.count_nonzero(distances <= _EXACT_TOLERANCE))
        reachable = exact_count > 0
        supported = (
            group.observations >= self.continuous_minimum_group_size
            and nearest <= group.radius
        )
        return SupportEvidence(
            solver=solver,
            stratum=(
                "empirical_supported"
                if supported
                else "reachable"
                if reachable
                else "interventional_only"
            ),
            basis=(
                "same_semantic_group_knn_radius_from_cedp_anchors"
            ),
            reachable=reachable,
            supported=supported,
            observation_count=group.observations,
            nearest_distance=nearest,
            support_radius=float(group.radius),
        )


def pair_stratum(
    source: SupportEvidence,
    target: SupportEvidence,
) -> str:
    if source.supported and target.supported:
        return "both_supported"
    if source.reachable and target.reachable:
        return "both_reachable"
    if source.reachable and not target.supported:
        return "reachable_source_interventional_target"
    return "unsupported_pair"
