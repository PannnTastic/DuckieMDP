import numpy as np
import torch

from src.agents.td3 import TD3Agent, TD3Config


def _agent(**cfg_overrides):
    cfg = TD3Config(batch_size=8, replay_capacity=200, **cfg_overrides)
    agent = TD3Agent(
        obs_dim=4,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        cfg=cfg,
        seed=0,
        device="cpu",
    )
    rng = np.random.RandomState(0)
    for _ in range(64):
        agent.replay.add(
            rng.randn(4).astype(np.float32),
            rng.uniform(-1, 1, size=2).astype(np.float32),
            float(rng.randn()),
            rng.randn(4).astype(np.float32),
            False,
        )
    return agent


def _actor_snapshot(agent):
    return [p.detach().clone() for p in agent.actor.parameters()]


def test_deterministic_action_is_bounded_and_repeatable():
    agent = _agent()
    obs = np.zeros(4, dtype=np.float32)
    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2)
    assert np.all(a1 >= agent.action_low) and np.all(a1 <= agent.action_high)


def test_actor_update_start_freezes_actor_then_releases_it():
    warmup = 6
    agent = _agent(actor_update_start=warmup, policy_delay=2)
    before = _actor_snapshot(agent)
    critic_before = [p.detach().clone() for p in agent.critic1.parameters()]

    # Updates below the warm-up threshold must not move the actor.
    for _ in range(warmup - 1):
        agent.update()
    during = _actor_snapshot(agent)
    assert all(
        torch.equal(a, b) for a, b in zip(before, during)
    ), "actor changed during critic warm-up"

    # The critic must still be learning while the actor is frozen.
    critic_during = [p.detach().clone() for p in agent.critic1.parameters()]
    assert any(
        not torch.equal(a, b) for a, b in zip(critic_before, critic_during)
    ), "critic did not learn during warm-up"

    # Past the threshold the actor is allowed to move again.
    for _ in range(warmup + 4):
        agent.update()
    after = _actor_snapshot(agent)
    assert any(
        not torch.equal(a, b) for a, b in zip(before, after)
    ), "actor never moved after warm-up ended"


def test_warmup_is_relative_to_a_resumed_update_count():
    # A checkpoint restores a large cumulative ``updates`` count. The warm-up
    # must still freeze the actor for ``actor_update_start`` updates after the
    # resume, not treat the window as already elapsed.
    warmup = 6
    agent = _agent(actor_update_start=warmup, policy_delay=2)
    agent.updates = 100_000
    agent._warmup_baseline = 100_000
    before = _actor_snapshot(agent)
    for _ in range(warmup - 1):
        agent.update()
    during = _actor_snapshot(agent)
    assert all(
        torch.equal(a, b) for a, b in zip(before, during)
    ), "actor moved during warm-up after a simulated resume"
    for _ in range(warmup + 4):
        agent.update()
    after = _actor_snapshot(agent)
    assert any(
        not torch.equal(a, b) for a, b in zip(before, after)
    ), "actor never moved after warm-up ended post-resume"


def test_default_config_updates_actor_from_the_start():
    agent = _agent(policy_delay=1)
    before = _actor_snapshot(agent)
    for _ in range(3):
        agent.update()
    after = _actor_snapshot(agent)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
