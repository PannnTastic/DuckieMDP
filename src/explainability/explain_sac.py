"""Internal diagnostics for the deterministic continuous SAC policy."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import torch

from ..agents.sac import Critic, SACAgent, SACConfig
from ..continuous_state import (
    OBSERVATION_NAMES,
    ContinuousStateConfig,
    continuous_observation_space,
)
from .counterfactual import SyntheticStateRecord, make_counterfactual, state_anchor_id
from .primitives import PrimitiveLabel, label_primitive
from .sac_policy_adapter import SACPolicyAdapter
from .schema import CanonicalAction, CanonicalState, SolverKind
from .semantic_state import encode_canonical_for_sac


SAC_DIAGNOSTIC_SCHEMA_VERSION = "1.0.0"
ACTION_OUTPUT_NAMES = ("v_cmd", "omega_cmd")
CONCEPT_GROUPS = {
    "lane": ("d",),
    "heading": ("phi",),
    "speed": ("v",),
    "road": ("kappa",),
    "stop": ("stop_present", "d_stop", "sigma_stop", "stop_hold_progress"),
    "pedestrian": (
        "duck_present", "duck_longitudinal", "duck_lateral",
        "duck_v_longitudinal_relative", "duck_v_lateral_relative",
        "duck_active", "duck_crossing_available",
    ),
}


@dataclass(frozen=True)
class IntegratedGradientsResult:
    anchor_id: str
    baseline_name: str
    steps: int
    feature_names: Tuple[str, ...]
    input_observation: Tuple[float, ...]
    baseline_observation: Tuple[float, ...]
    output_at_input: Mapping[str, float]
    output_at_baseline: Mapping[str, float]
    attributions: Mapping[str, Tuple[float, ...]]
    concept_signed: Mapping[str, Mapping[str, float]]
    concept_absolute: Mapping[str, Mapping[str, float]]
    completeness_residual: Mapping[str, float]
    schema_version: str = SAC_DIAGNOSTIC_SCHEMA_VERSION


@dataclass(frozen=True)
class CriticProbe:
    probe_name: str
    action: Tuple[float, float]
    min_q: float
    q1: float
    q2: float
    critic_disagreement: float
    delta_min_q_vs_actor_probe: float
    normalized_distance_to_actor: float
    actor_log_probability: float
    support_label: str
    caveat: str
    schema_version: str = SAC_DIAGNOSTIC_SCHEMA_VERSION


@dataclass(frozen=True)
class BoundaryPoint:
    feature: str
    delta: float
    synthetic: SyntheticStateRecord
    decision_action: Optional[CanonicalAction]
    primitive: Optional[PrimitiveLabel]
    normalized_action_distance: Optional[float]
    primitive_changed: Optional[bool]
    boundary: Optional[bool]
    schema_version: str = SAC_DIAGNOSTIC_SCHEMA_VERSION


@dataclass(frozen=True)
class BoundarySearchResult:
    anchor_id: str
    anchor_action: CanonicalAction
    anchor_primitive: PrimitiveLabel
    points: Tuple[BoundaryPoint, ...]
    valid_points: int
    rejected_points: int
    boundary_points: int
    nearest_boundary_feature: Optional[str]
    nearest_boundary_delta: Optional[float]
    threshold: float
    schema_version: str = SAC_DIAGNOSTIC_SCHEMA_VERSION


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def neutral_baseline_state():
    """Frozen valid canonical baseline from the M9 protocol."""
    return CanonicalState(
        d=0.0, phi=0.0, v=0.17, curvature=0.0,
        curvature_class="straight", stop_present=False, stop_distance=None,
        stop_satisfied=False, stop_hold_progress=0.0, duck_present=False,
        duck_threat=None, duck_longitudinal=None, duck_lateral=None,
        duck_v_longitudinal_relative=None, duck_v_lateral_relative=None,
        duck_active=None, duck_crossing_available=None,
        source_representation="sac_ig_neutral_baseline",
    )


def concept_aggregate(values):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(OBSERVATION_NAMES),):
        raise ValueError("attribution vector has wrong shape")
    lookup = {name: i for i, name in enumerate(OBSERVATION_NAMES)}
    signed, absolute = {}, {}
    for concept, names in CONCEPT_GROUPS.items():
        indices = [lookup[name] for name in names]
        signed[concept] = float(np.sum(vector[indices]))
        absolute[concept] = float(np.sum(np.abs(vector[indices])))
    return signed, absolute


def normalized_influence(values):
    vector = np.asarray(values, dtype=np.float64)
    scale = float(np.sum(np.abs(vector)))
    return np.zeros_like(vector) if scale <= 1e-12 else vector / scale


def explanation_distance(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    return float(1.0 - np.dot(left, right) / (left_norm * right_norm))


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_correlation(left, right):
    left_rank, right_rank = _rank(left), _rank(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 1.0 if np.allclose(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def compare_baselines(primary, alternative):
    result = {}
    for output in ACTION_OUTPUT_NAMES:
        left = np.asarray(primary.attributions[output], dtype=np.float64)
        right = np.asarray(alternative.attributions[output], dtype=np.float64)
        left_concepts = primary.concept_absolute[output]
        right_concepts = alternative.concept_absolute[output]
        dominant_left = max(left_concepts, key=left_concepts.get)
        dominant_right = max(right_concepts, key=right_concepts.get)
        result[output] = {
            "cosine_similarity": 1.0 - explanation_distance(left, right),
            "feature_rank_correlation": rank_correlation(np.abs(left), np.abs(right)),
            "dominant_primary": dominant_left,
            "dominant_alternative": dominant_right,
            "dominant_concept_stable": dominant_left == dominant_right,
        }
    return result


class SACInternalDiagnostics:
    """Actor attribution and double-critic queries without replay allocation."""

    def __init__(
        self, policy, critic1, critic2, action_low, action_high,
        observation_config=ContinuousStateConfig(), device="cpu",
        checkpoint_hash=None, checkpoint_path=None,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.actor = policy.actor.to(self.device).eval()
        self.critic1 = critic1.to(self.device).eval()
        self.critic2 = critic2.to(self.device).eval()
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.observation_config = observation_config
        self.observation_space = continuous_observation_space()
        self.checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(cls, path, observation_config=ContinuousStateConfig(), device="cpu"):
        path = Path(path)
        torch_device = torch.device(device)
        payload = _load_checkpoint(path, torch_device)
        policy = SACPolicyAdapter.from_checkpoint(
            path, observation_config=observation_config, device=device,
            allow_observation_expansion=True,
        )
        cfg = SACConfig(**payload["config"])
        critic1 = Critic(policy.observation_dim, 2, cfg.hidden_size)
        critic2 = Critic(policy.observation_dim, 2, cfg.hidden_size)
        for model, key in ((critic1, "critic1"), (critic2, "critic2")):
            state = payload[key]
            if int(payload["obs_dim"]) != policy.observation_dim:
                state = SACAgent._expanded_critic_state(
                    model, state, int(payload["obs_dim"]), policy.observation_dim
                )
            model.load_state_dict(state)
        return cls(
            policy, critic1, critic2, payload["action_low"], payload["action_high"],
            observation_config, device, _file_sha256(path), str(path),
        )

    def _action_tensor(self, observation):
        distribution = self.actor.distribution(observation)
        return (
            torch.tanh(distribution.mean) * self.actor.action_scale
            + self.actor.action_bias
        )

    def integrated_gradients(self, state, baseline_observation, baseline_name, steps=64):
        if steps < 2:
            raise ValueError("Integrated Gradients requires at least two steps")
        input_array = encode_canonical_for_sac(state, self.observation_config)
        baseline = np.asarray(baseline_observation, dtype=np.float32)
        if baseline.shape != input_array.shape or not self.observation_space.contains(baseline):
            raise ValueError("IG baseline must lie inside the observation Box")
        start = torch.as_tensor(baseline, dtype=torch.float32, device=self.device)
        end = torch.as_tensor(input_array, dtype=torch.float32, device=self.device)
        delta = end - start
        gradients = [torch.zeros_like(start), torch.zeros_like(start)]
        for step in range(steps + 1):
            alpha = float(step) / float(steps)
            point = (start + alpha * delta).detach().requires_grad_(True)
            output = self._action_tensor(point.unsqueeze(0)).squeeze(0)
            weight = 0.5 if step in (0, steps) else 1.0
            for output_index in range(2):
                gradient = torch.autograd.grad(
                    output[output_index], point,
                    retain_graph=output_index == 0,
                )[0]
                gradients[output_index] += weight * gradient.detach()
        with torch.no_grad():
            output_input = self._action_tensor(end.unsqueeze(0)).squeeze(0)
            output_baseline = self._action_tensor(start.unsqueeze(0)).squeeze(0)
        attributions, signed, absolute, residual = {}, {}, {}, {}
        for output_index, output_name in enumerate(ACTION_OUTPUT_NAMES):
            values = delta * gradients[output_index] / float(steps)
            array = values.cpu().numpy().astype(np.float64)
            attributions[output_name] = tuple(float(value) for value in array)
            signed[output_name], absolute[output_name] = concept_aggregate(array)
            explained = float(np.sum(array))
            actual = float((output_input[output_index] - output_baseline[output_index]).item())
            residual[output_name] = actual - explained
        return IntegratedGradientsResult(
            anchor_id=state_anchor_id(state), baseline_name=str(baseline_name),
            steps=steps, feature_names=tuple(OBSERVATION_NAMES),
            input_observation=tuple(float(value) for value in input_array),
            baseline_observation=tuple(float(value) for value in baseline),
            output_at_input={name: float(output_input[i].item()) for i, name in enumerate(ACTION_OUTPUT_NAMES)},
            output_at_baseline={name: float(output_baseline[i].item()) for i, name in enumerate(ACTION_OUTPUT_NAMES)},
            attributions=attributions, concept_signed=signed,
            concept_absolute=absolute, completeness_residual=residual,
        )

    def _actor_log_probability(self, observation, action):
        distribution = self.actor.distribution(observation)
        normalized = (action - self.actor.action_bias) / self.actor.action_scale
        normalized = torch.clamp(normalized, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(normalized)
        correction = torch.log(
            self.actor.action_scale * (1.0 - normalized.pow(2)) + 1e-6
        )
        return float((distribution.log_prob(pre_tanh) - correction).sum().item())

    def critic_probes(self, state, reference_actions=None):
        observation_array = encode_canonical_for_sac(state, self.observation_config)
        observation = torch.as_tensor(
            observation_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            actor_action = self._action_tensor(observation).squeeze(0)
        if reference_actions is None:
            reference_actions = {
                "actor": tuple(float(x) for x in actor_action.cpu().numpy()),
                "brake": (0.0, 0.0),
                "slow_straight": (0.17, 0.0),
                "cruise_straight": (float(self.action_high[0]), 0.0),
                "corrective_left": (0.17, float(self.action_high[1])),
                "corrective_right": (0.17, float(self.action_low[1])),
            }
        scale = np.maximum(self.action_high - self.action_low, 1e-9)
        raw = []
        for name, values in reference_actions.items():
            array = np.clip(np.asarray(values, dtype=np.float32), self.action_low, self.action_high)
            action = torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q1 = float(self.critic1(observation, action).item())
                q2 = float(self.critic2(observation, action).item())
                log_probability = self._actor_log_probability(observation, action)
            distance = float(np.linalg.norm((array - actor_action.cpu().numpy()) / scale))
            raw.append((name, array, q1, q2, log_probability, distance))
        actor_row = next((row for row in raw if row[0] == "actor"), None)
        if actor_row is None:
            raise ValueError("critic probes require an actor reference action")
        actor_min_q = min(actor_row[2], actor_row[3])
        probes = []
        for name, array, q1, q2, log_probability, distance in raw:
            min_q = min(q1, q2)
            probes.append(CriticProbe(
                probe_name=name, action=(float(array[0]), float(array[1])),
                min_q=min_q, q1=q1, q2=q2,
                critic_disagreement=abs(q1 - q2),
                delta_min_q_vs_actor_probe=min_q - actor_min_q,
                normalized_distance_to_actor=distance,
                actor_log_probability=log_probability,
                support_label=(
                    "ACTOR_ACTION" if name == "actor"
                    else "LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT"
                ),
                caveat=(
                    "Critic comparison is heuristic; no replay snapshot is available "
                    "to establish state-action support."
                ),
            ))
        return tuple(probes)


def empirical_centroid(states, config=ContinuousStateConfig()):
    if not states:
        raise ValueError("empirical centroid requires at least one real state")
    observations = np.stack([
        encode_canonical_for_sac(state, config) for state in states
    ]).astype(np.float32)
    centroid = np.mean(observations, axis=0, dtype=np.float64).astype(np.float32)
    if not continuous_observation_space().contains(centroid):
        raise ValueError("empirical centroid lies outside observation Box")
    return centroid


def _perturbation_requests(anchor):
    requests = [
        ("d", -0.005), ("d", 0.005),
        ("phi", -0.01), ("phi", 0.01),
    ]
    requests.extend(
        ("v", delta) for delta in (-0.01, 0.01)
        if anchor.v + delta >= 0.0
    )
    if anchor.stop_present and anchor.stop_distance is not None:
        requests.extend(
            ("stop_distance", delta) for delta in (-0.02, 0.02)
            if anchor.stop_distance + delta >= 0.0
        )
    if anchor.duck_present:
        requests.extend((
            ("duck_longitudinal", -0.02), ("duck_longitudinal", 0.02),
            ("duck_lateral", -0.02), ("duck_lateral", 0.02),
        ))
    return tuple(requests)


def local_boundary_search(policy, diagnostics, anchor, action_threshold=0.05):
    anchor_decision = policy.decide(anchor)
    anchor_primitive = label_primitive(anchor, anchor_decision.action)
    action_range = np.maximum(diagnostics.action_high - diagnostics.action_low, 1e-9)
    anchor_action = np.asarray([
        anchor_decision.action.v_cmd, anchor_decision.action.omega_cmd
    ], dtype=np.float64)
    points = []
    for feature, delta in _perturbation_requests(anchor):
        value = float(getattr(anchor, feature)) + delta
        synthetic = make_counterfactual(
            anchor, SolverKind.SAC, "m9_local_boundary_%s" % feature,
            {feature: value}, state_anchor_id(anchor),
        )
        if not synthetic.validation.valid:
            points.append(BoundaryPoint(
                feature, delta, synthetic, None, None, None, None, None
            ))
            continue
        decision = policy.decide(synthetic.state)
        primitive = label_primitive(synthetic.state, decision.action)
        action = np.asarray([decision.action.v_cmd, decision.action.omega_cmd])
        distance = float(np.linalg.norm((action - anchor_action) / action_range))
        changed = primitive.primitive != anchor_primitive.primitive
        boundary = changed or distance > action_threshold
        points.append(BoundaryPoint(
            feature, delta, synthetic, decision.action, primitive,
            distance, changed, boundary,
        ))
    boundaries = [point for point in points if point.boundary is True]
    nearest = min(boundaries, key=lambda point: abs(point.delta)) if boundaries else None
    return BoundarySearchResult(
        anchor_id=state_anchor_id(anchor), anchor_action=anchor_decision.action,
        anchor_primitive=anchor_primitive, points=tuple(points),
        valid_points=sum(point.decision_action is not None for point in points),
        rejected_points=sum(point.decision_action is None for point in points),
        boundary_points=len(boundaries),
        nearest_boundary_feature=None if nearest is None else nearest.feature,
        nearest_boundary_delta=None if nearest is None else nearest.delta,
        threshold=action_threshold,
    )


def attribution_stability(
    diagnostics, policy, anchors, baseline_observation, steps=32,
    acceptance_p95=0.10,
):
    rows = []
    for anchor in anchors:
        anchor_ig = diagnostics.integrated_gradients(
            anchor, baseline_observation, "neutral", steps=steps
        )
        boundary = local_boundary_search(policy, diagnostics, anchor)
        near_boundary = boundary.boundary_points > 0
        for point in boundary.points:
            if point.decision_action is None:
                continue
            neighbor_ig = diagnostics.integrated_gradients(
                point.synthetic.state, baseline_observation, "neutral", steps=steps
            )
            for output in ACTION_OUTPUT_NAMES:
                left = normalized_influence(anchor_ig.attributions[output])
                right = normalized_influence(neighbor_ig.attributions[output])
                rows.append({
                    "anchor_id": state_anchor_id(anchor),
                    "feature": point.feature,
                    "delta": point.delta,
                    "output": output,
                    "near_boundary": near_boundary,
                    "explanation_distance": explanation_distance(left, right),
                })
    non_boundary = [row["explanation_distance"] for row in rows if not row["near_boundary"]]
    return {
        "rows": rows,
        "anchors": len(anchors),
        "near_boundary_anchors": len({row["anchor_id"] for row in rows if row["near_boundary"]}),
        "non_boundary_comparisons": len(non_boundary),
        "median_non_boundary": None if not non_boundary else float(np.median(non_boundary)),
        "p95_non_boundary": None if not non_boundary else float(np.percentile(non_boundary, 95)),
        "acceptance_threshold": acceptance_p95,
        "accepted": None if not non_boundary else float(np.percentile(non_boundary, 95)) <= acceptance_p95,
        "disposition": (
            "NO_NON_BOUNDARY_ANCHORS" if not non_boundary
            else "PASS" if float(np.percentile(non_boundary, 95)) <= acceptance_p95
            else "FAIL_RETAIN_FOR_AUDIT"
        ),
    }
