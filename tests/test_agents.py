import numpy as np
import pytest

from agents import Config, greedy_path, q_learning_target, sarsa_target, train
from gridworld import GridWorld


def test_env_walls_goal_and_pit():
    env = GridWorld(size=5)
    assert env.reset() == 0
    s, r, done = env.step(0)  # up from (0,0) hits the wall
    assert (s, r, done) == (0, -1.0, False)
    env.state = (4, 3)
    _, r, done = env.step(3)
    assert (r, done) == (10.0, True)
    env.state = (2, 1)
    _, r, done = env.step(3)
    assert (r, done) == (-10.0, True)


def test_targets_differ_only_in_bootstrap():
    q = np.zeros((25, 4))
    q[7] = [1.0, 5.0, 2.0, 0.0]
    assert q_learning_target(q, -1.0, 7, False, 0.9) == pytest.approx(-1 + 0.9 * 5.0)
    assert sarsa_target(q, -1.0, 7, 2, False, 0.9) == pytest.approx(-1 + 0.9 * 2.0)
    # Terminal transitions never bootstrap.
    assert q_learning_target(q, 10.0, 7, True, 0.9) == 10.0
    assert sarsa_target(q, 10.0, 7, 1, True, 0.9) == 10.0


def test_training_is_reproducible():
    cfg = Config(episodes=50)
    q1, r1 = train("sarsa", cfg, seed=3)
    q2, r2 = train("sarsa", cfg, seed=3)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_learns_the_optimal_8_step_path(algorithm):
    q, _ = train(algorithm, Config(episodes=1000), seed=0)
    path = greedy_path(q)
    assert path[-1] == (4, 4)
    assert len(path) - 1 == 8
    assert (2, 2) not in path


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError):
        train("dqn")
