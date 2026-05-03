import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, SAC


# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.circle_tracking_env import CircleTracking3DEnv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Model configuration (no PPO)
# ---------------------------------------------------------------------
MODEL_CONFIG = {
    "TD3": {
        "class": TD3,
        "path": PROJECT_ROOT / "models" / "td3_circle_optimized" / "td3_circle_optimized_final_model",
    },
    "SAC": {
        "class": SAC,
        "path": PROJECT_ROOT / "models" / "sac_circle_optimized" / "sac_circle_optimized_final_model",
    },
    # Map DDPG to TD3 (no native DDPG in stable-baselines3)
    "DDPG": {
        "class": TD3,
        "path": PROJECT_ROOT / "models" / "ddpg_circle_optimized" / "ddpg_circle_optimized_final_model",
    },
}


# ---------------------------------------------------------------------
# Gym / Gymnasium compatibility helpers
# ---------------------------------------------------------------------
def _unpack_reset(res):
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0]
    return res


def _step_return(env_step_out):
    # normalize to (obs, reward, done, info)
    if len(env_step_out) == 4:
        return env_step_out
    if len(env_step_out) == 5:
        obs, reward, terminated, truncated, info = env_step_out
        done = terminated or truncated
        return obs, reward, done, info
    raise RuntimeError("Unexpected env.step() signature")


# ---------------------------------------------------------------------
# Utilities for model paths and loading
# ---------------------------------------------------------------------
def resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path
    zip_path = Path(str(path) + ".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"Model not found: {path} or {zip_path}")


def load_models():
    models = {}
    for algo_name, config in MODEL_CONFIG.items():
        model_path = resolve_model_path(config["path"])
        model_class = config["class"]
        print(f"Loading {algo_name} model from: {model_path}", flush=True)
        models[algo_name] = model_class.load(str(model_path))
    return models


# ---------------------------------------------------------------------
# Run episodes and collect basic traces
# ---------------------------------------------------------------------
def run_episode(model, max_steps=500):
    env = CircleTracking3DEnv(max_steps=max_steps)
    res = env.reset()
    obs = _unpack_reset(res)

    positions = []
    actions = []
    rewards = []
    targets = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        obs, reward, done, info = _step_return(step_out)

        # Try to extract position and target — be defensive
        pos = None
        if isinstance(obs, dict):
            # some envs return dict observations
            # try common keys
            for k in ("position", "pos", "agent_pos"):
                if k in obs:
                    pos = np.asarray(obs[k])
                    break
            # if observation contains both state and target
            for k in ("target", "desired", "target_pos"):
                if k in obs:
                    targets.append(np.asarray(obs[k]))
        else:
            # assume observation has position in first 3 dims
            try:
                arr = np.asarray(obs)
                if arr.ndim >= 1:
                    pos = arr[:3]
            except Exception:
                pos = None

        # If info carries target/position, prefer it
        if pos is None:
            for k in ("position", "pos", "agent_pos"):
                if isinstance(info, dict) and k in info:
                    pos = np.asarray(info[k])
                    break

        if isinstance(info, dict) and "target" in info and len(targets) == 0:
            # some envs place target in info only once
            targets.append(np.asarray(info["target"]))

        positions.append(np.asarray(pos) if pos is not None else np.array([np.nan, np.nan, np.nan]))
        actions.append(np.asarray(action))
        rewards.append(float(reward))

        if done:
            break

    env.close()

    episode = {
        "positions": np.vstack(positions),
        "actions": np.vstack(actions) if len(actions) > 0 else np.zeros((0,)),
        "rewards": np.array(rewards),
        "targets": np.vstack(targets) if len(targets) > 0 else None,
    }
    return episode


def evaluate_model(model, n_episodes=5, max_steps=500):
    episodes = []
    total_rewards = []
    for _ in range(n_episodes):
        ep = run_episode(model, max_steps=max_steps)
        episodes.append(ep)
        total_rewards.append(ep["rewards"].sum())
    summary = {
        "episodes": episodes,
        "total_rewards": np.array(total_rewards),
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
    }
    return summary


# ---------------------------------------------------------------------
# Plotting helpers (simple, robust)
# ---------------------------------------------------------------------
def plot_reward_comparison(results):
    algos = []
    means = []
    stds = []
    for algo, res in results.items():
        algos.append(algo)
        means.append(res["mean_reward"])
        stds.append(res["std_reward"])

    plt.figure(figsize=(6, 4))
    x = np.arange(len(algos))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algos)
    plt.ylabel("Mean total reward")
    plt.title("Mean episode reward (comparison)")
    plt.tight_layout()
    out = PLOTS_DIR / "reward_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}", flush=True)


def plot_3d_trajectories(results):
    import math

    algos = list(results.keys())
    n = len(algos)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    axes = []
    for i, algo in enumerate(algos):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        axes.append(ax)
        idx = int(np.argmax(results[algo]["total_rewards"]))
        ep = results[algo]["episodes"][idx]
        pos = ep["positions"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"{algo}", linewidth=2)
        if ep.get("targets") is not None:
            targ = ep["targets"]
            if targ.shape[1] >= 3:
                ax.plot(targ[:, 0], targ[:, 1], targ[:, 2], "--", color="k", linewidth=1.5, label="Target")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{algo} - best episode")
        ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "trajectory_3d_faceted.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}", flush=True)


def plot_distance_over_time(results):
    import math

    algos = list(results.keys())
    n = len(algos)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    any_plotted = False
    for i, algo in enumerate(algos):
        r = results[algo]
        ep = r["episodes"][0]
        pos = ep["positions"]
        targ = ep.get("targets")
        ax = axs[i // cols][i % cols]
        if targ is None or pos.size == 0:
            ax.text(0.5, 0.5, "No target trace", transform=ax.transAxes, ha="center")
            ax.set_title(algo)
            continue

        targ = np.asarray(targ)
        if targ.ndim == 1:
            targ = targ.reshape(1, -1)
        if len(targ) == 1 and len(pos) > 1:
            targ_full = np.tile(targ[0], (len(pos), 1))
        else:
            if len(targ) < len(pos):
                last = targ[-1]
                pad = np.tile(last, (len(pos) - len(targ), 1))
                targ_full = np.vstack([targ, pad])
            else:
                targ_full = targ[: len(pos)]

        L = min(len(pos), len(targ_full))
        if L == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(algo)
            continue

        d = np.linalg.norm(pos[:L] - targ_full[:L], axis=1)
        ax.plot(np.arange(L), d, label=algo)
        ax.set_xlabel("Step"); ax.set_ylabel("Distance")
        ax.set_title(algo)
        ax.legend()
        any_plotted = True

    # clear unused axes
    total_axes = rows * cols
    for j in range(len(algos), total_axes):
        ax = axs[j // cols][j % cols]
        ax.axis("off")

    plt.tight_layout()
    if any_plotted:
        out = PLOTS_DIR / "distance_over_time_faceted.png"
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}", flush=True)
    else:
        print("No target traces found; skipping distance faceted plot.", flush=True)
    plt.close()


def plot_distance_over_time_faceted(results):
    """
    Facet layout: rows = algorithms, cols = episodes (one panel per algo-per-episode).
    """
    algos = list(results.keys())
    if len(algos) == 0:
        print("No algorithms to plot.", flush=True)
        return

    # determine max episodes across algorithms
    n_eps = max(len(results[a]["episodes"]) for a in algos)
    cols = max(1, n_eps)
    rows = len(algos)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    any_plotted = False
    for i, algo in enumerate(algos):
        res = results[algo]
        for j in range(cols):
            ax = axs[i][j]
            if j >= len(res["episodes"]):
                ax.text(0.5, 0.5, "No episode", ha="center", transform=ax.transAxes)
                ax.set_title(f"{algo} - ep {j+1}")
                ax.set_xlabel("Step"); ax.set_ylabel("Distance")
                continue

            ep = res["episodes"][j]
            pos = ep["positions"]
            targ = ep.get("targets")
            if pos.size == 0 or targ is None:
                ax.text(0.5, 0.5, "No data/target", ha="center", transform=ax.transAxes)
                ax.set_title(f"{algo} - ep {j+1}")
                ax.set_xlabel("Step"); ax.set_ylabel("Distance")
                continue

            targ = np.asarray(targ)
            if targ.ndim == 1:
                targ = targ.reshape(1, -1)
            if len(targ) == 1 and len(pos) > 1:
                targ_full = np.tile(targ[0], (len(pos), 1))
            else:
                if len(targ) < len(pos):
                    last = targ[-1]
                    pad = np.tile(last, (len(pos) - len(targ), 1))
                    targ_full = np.vstack([targ, pad])
                else:
                    targ_full = targ[: len(pos)]

            L = min(len(pos), len(targ_full))
            if L == 0:
                ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            else:
                d = np.linalg.norm(pos[:L] - targ_full[:L], axis=1)
                ax.plot(np.arange(L), d, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10])
                ax.set_xlabel("Step"); ax.set_ylabel("Distance")
                any_plotted = True
            ax.set_title(f"{algo} - ep {j+1}")

    # turn off any extra axes (unlikely)
    total_axes = rows * cols
    count = 0
    for i in range(rows):
        for j in range(cols):
            count += 1
    plt.tight_layout()
    if any_plotted:
        out = PLOTS_DIR / "distance_over_time_faceted_by_episode.png"
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}", flush=True)
    else:
        print("No distance traces found; skipping faceted distance plot.", flush=True)
    plt.close()


def plot_action_magnitude(results):
    import math

    algos = list(results.keys())
    n = len(algos)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    any_plotted = False
    for i, algo in enumerate(algos):
        r = results[algo]
        ep = r["episodes"][0]
        acts = ep["actions"]
        ax = axs[i // cols][i % cols]
        if acts.size == 0:
            ax.text(0.5, 0.5, "No action trace", transform=ax.transAxes, ha="center")
            ax.set_title(algo)
            continue
        mag = np.linalg.norm(acts, axis=1)
        ax.plot(mag, label=algo)
        ax.set_xlabel("Step"); ax.set_ylabel("Action magnitude"); ax.set_title(algo)
        ax.legend()
        any_plotted = True

    # clear unused axes
    total_axes = rows * cols
    for j in range(len(algos), total_axes):
        ax = axs[j // cols][j % cols]
        ax.axis("off")

    plt.tight_layout()
    if any_plotted:
        out = PLOTS_DIR / "action_magnitude_faceted.png"
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}", flush=True)
    else:
        print("No action traces; skipping action magnitude faceted plot.", flush=True)
    plt.close()


# ---------------------------------------------------------------------
# Summary and CSV output
# ---------------------------------------------------------------------
def print_summary(results):
    print("\nSummary:", flush=True)
    for algo, res in results.items():
        print(f"{algo}: mean={res['mean_reward']:.3f} std={res['std_reward']:.3f}", flush=True)


def save_summary_csv(results):
    import csv

    out = RESULTS_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "mean_reward", "std_reward"])
        for algo, res in results.items():
            writer.writerow([algo, res["mean_reward"], res["std_reward"]])
    print(f"Saved summary CSV: {out}", flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    models = load_models()

    results = {}
    for algo, model in models.items():
        print(f"Evaluating {algo} ...", flush=True)
        results[algo] = evaluate_model(model, n_episodes=3, max_steps=500)

    # plots
    plot_reward_comparison(results)
    plot_3d_trajectories(results)
    plot_distance_over_time(results)
    plot_distance_over_time_faceted(results)
    plot_action_magnitude(results)

    print_summary(results)
    save_summary_csv(results)


if __name__ == "__main__":
    main()