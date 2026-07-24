"""Load the exact frozen policies and shared evaluation environment."""

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.actions import ActionConfig
from src.continuous_env import build_continuous_env
from src.env_wrapper import build_env
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter
from src.explainability.td3_policy_adapter import TD3PolicyAdapter


def load_runtime(config_path: Path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    shared_path = Path(config["experiment"]["shared_environment_config"])
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
    shared["environment"]["render_observations"] = False
    actions = ActionConfig(**shared["actions"])

    # Build only the policies declared in the config, so a pilot can run a
    # single solver while the main experiment still loads all four.
    pol = config["policies"]
    policies = {}
    gammas = {}
    if "q_learning" in pol:
        info = pol["q_learning"]
        train = yaml.safe_load(Path(info["training_config"]).read_text(encoding="utf-8"))
        policies["q_learning"] = QPolicyAdapter.from_checkpoint(
            Path(info["checkpoint"]), train["q_learning"]["allowed_actions"], actions,
        )
        gammas["q_learning"] = float(train["q_learning"]["gamma"])
    if "sarsa" in pol:
        info = pol["sarsa"]
        train = yaml.safe_load(Path(info["training_config"]).read_text(encoding="utf-8"))
        policies["sarsa"] = SarsaPolicyAdapter.from_checkpoint(
            Path(info["checkpoint"]), train["sarsa"]["allowed_actions"], actions,
        )
        gammas["sarsa"] = float(train["sarsa"]["gamma"])
    if "sac" in pol:
        info = pol["sac"]
        train = yaml.safe_load(Path(info["training_config"]).read_text(encoding="utf-8"))
        policies["sac"] = SACPolicyAdapter.from_checkpoint(
            Path(info["checkpoint"]), allow_observation_expansion=True,
        )
        gammas["sac"] = float(train["sac"]["gamma"])
    if "td3" in pol:
        info = pol["td3"]
        train = yaml.safe_load(Path(info["training_config"]).read_text(encoding="utf-8"))
        policies["td3"] = TD3PolicyAdapter.from_checkpoint(Path(info["checkpoint"]))
        gammas["td3"] = float(train["td3"]["gamma"])
    return config, shared_path, shared, policies, gammas


def environment_factory(shared: Mapping[str, Any], solver: str):
    if solver in {"sac", "td3"}:
        return lambda seed: build_continuous_env(shared, int(seed))
    return lambda seed: build_env(shared, int(seed))
