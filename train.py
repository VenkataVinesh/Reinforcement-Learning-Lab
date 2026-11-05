"""Train Q-learning and SARSA on the same GridWorld over several seeds.

    python train.py              # 10 seeds x 1000 episodes, writes reward_curve.png
    python train.py --seeds 3    # quicker
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import Config, greedy_path, train
from gridworld import GridWorld

OPTIMAL_STEPS = 8  # (0,0) -> (4,4) on a 5x5 grid, avoiding the pit at (2,2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    cfg = Config(episodes=args.episodes)
    goal = GridWorld().goal

    fig, ax = plt.subplots(figsize=(9, 4.5))
    print(f"{'algorithm':<11} {'mean reward, last 100 ep':>25} {'greedy path reaches goal':>25} {'path = 8 steps':>15}")
    for algo, colour in (("q_learning", "#2b6cb0"), ("sarsa", "#c05621")):
        curves, tail, reached, optimal = [], [], 0, 0
        for seed in range(args.seeds):
            q, rewards = train(algo, cfg, seed=seed)
            curves.append(rewards)
            tail.append(rewards[-100:].mean())
            path = greedy_path(q)
            if path[-1] == goal:
                reached += 1
                optimal += int(len(path) - 1 == OPTIMAL_STEPS)
        curves = np.array(curves)
        window = 20
        smooth = np.array([np.convolve(c, np.ones(window) / window, mode="valid") for c in curves])
        mean, std = smooth.mean(0), smooth.std(0)
        x = np.arange(len(mean)) + window
        ax.plot(x, mean, color=colour, label=algo.replace("_", "-"))
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15)
        print(f"{algo:<11} {np.mean(tail):>19.2f} ± {np.std(tail):.2f} {reached:>18}/{args.seeds} {optimal:>12}/{args.seeds}")

    ax.set_title(f"GridWorld 5x5: cumulative reward per episode ({args.seeds} seeds, mean ± 1 sd, 20-ep smoothing)")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.axhline(10 - (OPTIMAL_STEPS - 1), color="grey", lw=0.8, ls="--", label="optimal episode (+3)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("reward_curve.png", dpi=120)
    print("saved reward_curve.png")


if __name__ == "__main__":
    main()
