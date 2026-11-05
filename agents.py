"""Tabular Q-learning and SARSA on the GridWorld, written out by hand.

Both agents share the environment, the epsilon-greedy behaviour policy and the
epsilon schedule. The only difference is the bootstrap target:

    Q-learning (off-policy): r + gamma * max_a' Q(s', a')
    SARSA      (on-policy):  r + gamma * Q(s', a'), where a' is the action the
                             behaviour policy will actually take next.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridworld import GridWorld


@dataclass(frozen=True)
class Config:
    episodes: int = 1000
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    max_steps: int = 200  # safety cap so an early random policy cannot loop forever


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    # Break ties at random so an all-zero row does not always pick action 0.
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def q_learning_target(q: np.ndarray, reward: float, next_state: int, done: bool, gamma: float) -> float:
    return reward if done else reward + gamma * float(q[next_state].max())


def sarsa_target(q: np.ndarray, reward: float, next_state: int, next_action: int, done: bool, gamma: float) -> float:
    return reward if done else reward + gamma * float(q[next_state, next_action])


def train(algorithm: str, cfg: Config = Config(), seed: int = 0, size: int = 5):
    """Train one agent. Returns (Q-table, per-episode cumulative reward)."""
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError(f"unknown algorithm: {algorithm}")
    rng = np.random.default_rng(seed)
    env = GridWorld(size=size)
    q = np.zeros((env.num_states, len(env.action_space)))
    epsilon = cfg.epsilon_start
    rewards = np.zeros(cfg.episodes)

    for ep in range(cfg.episodes):
        state = env.reset()
        action = epsilon_greedy(q[state], epsilon, rng)
        total = 0.0
        for _ in range(cfg.max_steps):
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(q[next_state], epsilon, rng)
            if algorithm == "q_learning":
                target = q_learning_target(q, reward, next_state, done, cfg.gamma)
            else:
                target = sarsa_target(q, reward, next_state, next_action, done, cfg.gamma)
            q[state, action] += cfg.alpha * (target - q[state, action])
            total += reward
            state, action = next_state, next_action
            if done:
                break
        rewards[ep] = total
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
    return q, rewards


def greedy_path(q: np.ndarray, size: int = 5, max_steps: int = 50) -> list[tuple[int, int]]:
    """Follow argmax Q from the start. Returns visited cells, including the start."""
    env = GridWorld(size=size)
    state = env.reset()
    path = [env.state]
    for _ in range(max_steps):
        state, _, done = env.step(int(np.argmax(q[state])))
        path.append(env.state)
        if done:
            break
    return path
