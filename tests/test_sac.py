import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.sac import SACAgent, SACConfig


def _agent(seed=1, obs_dim=15):
    return SACAgent(
        obs_dim=obs_dim,
        action_low=np.array([0.0, -1.5], dtype=np.float32),
        action_high=np.array([0.41, 1.5], dtype=np.float32),
        cfg=SACConfig(batch_size=8, replay_capacity=64, hidden_size=32),
        seed=seed,
    )


def test_action_respects_asymmetric_bounds():
    agent = _agent()
    for _ in range(20):
        action = agent.select_action(np.zeros(agent.obs_dim, dtype=np.float32))
        assert 0.0 <= action[0] <= 0.41
        assert -1.5 <= action[1] <= 1.5


def test_update_is_finite_and_checkpoint_round_trips(tmp_path):
    agent = _agent()
    rng = np.random.RandomState(2)
    for _ in range(16):
        obs = rng.uniform(-1, 1, size=agent.obs_dim).astype(np.float32)
        next_obs = rng.uniform(-1, 1, size=agent.obs_dim).astype(np.float32)
        action = np.array(
            [rng.uniform(0, 0.41), rng.uniform(-1.5, 1.5)], dtype=np.float32
        )
        agent.replay.add(obs, action, rng.randn(), next_obs, False)
    metrics = agent.update()
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())

    path = tmp_path / "agent.pt"
    before = agent.select_action(
        np.zeros(agent.obs_dim, dtype=np.float32), deterministic=True
    )
    agent.save(path)
    restored = _agent(seed=99)
    restored.load(path)
    after = restored.select_action(
        np.zeros(restored.obs_dim, dtype=np.float32), deterministic=True
    )
    assert np.allclose(before, after)


def test_14_to_15_observation_migration_preserves_actor_and_critic(tmp_path):
    old = _agent(seed=4, obs_dim=14)
    path = tmp_path / "old_14d.pt"
    old.save(path)

    rng = np.random.RandomState(7)
    old_observation = rng.uniform(-1, 1, size=14).astype(np.float32)
    new_observation = np.concatenate(
        [old_observation, np.zeros(1, dtype=np.float32)]
    )
    action = np.array([0.2, -0.4], dtype=np.float32)
    actor_before = old.select_action(old_observation, deterministic=True)
    with torch.no_grad():
        old_obs_tensor = torch.as_tensor(old_observation).unsqueeze(0)
        action_tensor = torch.as_tensor(action).unsqueeze(0)
        critic_before = old.critic1(old_obs_tensor, action_tensor).cpu().numpy()

    migrated = _agent(seed=99, obs_dim=15)
    migrated.load(path, allow_observation_expansion=True)
    actor_after = migrated.select_action(new_observation, deterministic=True)
    with torch.no_grad():
        new_obs_tensor = torch.as_tensor(new_observation).unsqueeze(0)
        critic_after = migrated.critic1(new_obs_tensor, action_tensor).cpu().numpy()

    assert np.allclose(actor_before, actor_after, atol=1e-7)
    assert np.allclose(critic_before, critic_after, atol=1e-7)


def test_observation_migration_must_be_explicit(tmp_path):
    old = _agent(obs_dim=14)
    path = tmp_path / "old.pt"
    old.save(path)
    with pytest.raises(ValueError, match="dimension mismatch"):
        _agent(obs_dim=15).load(path)
