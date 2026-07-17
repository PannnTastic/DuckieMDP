"""Factory agent tabular agar evaluator/renderer dapat dipakai lintas solver."""

from typing import Any, Dict

from .q_learning import QLearningAgent, QLearningConfig
from .sarsa import SarsaAgent, SarsaConfig


def algorithm_name(config: Dict[str, Any]) -> str:
    return str(config.get("algorithm", "q_learning")).strip().lower()


def build_tabular_agent(config: Dict[str, Any], seed: int):
    algorithm = algorithm_name(config)
    if algorithm == "q_learning":
        return QLearningAgent(QLearningConfig(**config["q_learning"]), seed)
    if algorithm == "sarsa":
        return SarsaAgent(SarsaConfig(**config["sarsa"]), seed)
    raise ValueError(f"Unsupported tabular algorithm: {algorithm}")
