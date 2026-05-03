import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from stable_baselines3 import TD3, SAC

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.circle_tracking_env import CircleTracking3DEnv

# Paths
RESULTS_DIR = PROJECT_ROOT / "results"
ANIMATION_DIR = RESULTS_DIR / "animations"
RESULTS_DIR.mkdir(exist_ok=True)
ANIMATION_DIR.mkdir(exist_ok=True)

# Model configuration (DDPG mapped to TD3)
MODEL_CONFIG = {
    "TD3": {
        "class": TD3,
        "path": PROJECT_ROOT / "models" / "td3_circle_optimized" / "td3_circle_optimized_final_model",
    },
    "SAC": {
        "class": SAC,
        "path": PROJECT_ROOT / "models" / "sac_circle_optimized" / "sac_circle_optimized_final_model",
    },
    "DDPG": {
        "class": TD3,
        "path": PROJECT_ROOT / "models" / "ddpg_circle_optimized" / "ddpg_circle_optimized_final_model",
    },
}


# Helpers for Gym / Gymnasium compatibility
def _unpack_reset(res):
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0]
    return res


def _step_return(out):
    # normalize to (obs, reward, done, info)
    if len(out) == 4:
        return out
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    raise RuntimeError("Unexpected env.step() signature")


def resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path
    zip_path = Path(str(path) + ".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"Model not found: {path} or {zip_path}")


def load_models():
    models = {}
    for algo, cfg in MODEL_CONFIG.items():
        model_path = resolve_model_path(cfg["path"])
        model_cls = cfg["class"]
        print(f"Loading {algo} model from: {model_path}", flush=True)
        models[algo] = model_cls.load(str(model_path))
    return models


def rollout_model(model, max_steps=500):
    env = CircleTracking3DEnv(max_steps=max_steps)
    res = env.reset()
    obs = _unpack_reset(res)

    positions = []
    targets = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        out = env.step(action)
        obs, reward, done, info = _step_return(out)

        # try to extract agent position
        pos = None
        if isinstance(obs, dict):
            for k in ("position", "pos", "agent_pos", "state"):
                if k in obs:
                    pos = np.asarray(obs[k])
                    break
            for k in ("target", "desired", "target_pos"):
                if k in obs:
                    targets.append(np.asarray(obs[k]))
        else:
            try:
                arr = np.asarray(obs)
                if arr.size >= 3:
                    pos = arr[:3]
            except Exception:
                pos = None

        # fallback to info
        if pos is None and isinstance(info, dict):
            for k in ("position", "pos", "agent_pos"):
                if k in info:
                    pos = np.asarray(info[k])
                    break
            if "target" in info:
                targets.append(np.asarray(info["target"]))

        # ensure pos is a 3-vector
        if pos is None:
            pos = np.array([np.nan, np.nan, np.nan])
        else:
            pos = np.asarray(pos).reshape(-1)[:3]

        positions.append(pos)

        if done:
            break

    env.close()
    positions = np.vstack(positions)
    targets = np.vstack(targets) if len(targets) > 0 else None
    return {"positions": positions, "targets": targets}


def prepare_rollouts(models, max_steps=500):
    rollouts = {}
    for algo, model in models.items():
        print(f"Rolling out {algo} ...", flush=True)
        rollouts[algo] = rollout_model(model, max_steps=max_steps)
    return rollouts


def get_axis_limits(rollouts, padding=0.2):
    all_pos = np.vstack([r["positions"] for r in rollouts.values() if r["positions"].size > 0])
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    ranges = maxs - mins
    mins = mins - ranges * padding
    maxs = maxs + ranges * padding
    return {"x": (mins[0], maxs[0]), "y": (mins[1], maxs[1]), "z": (mins[2], maxs[2])}


def create_3d_animation(rollouts, fps=20, trail_length=80):
    print("Creating 3D animation...", flush=True)
    min_frames = min(len(r["positions"]) for r in rollouts.values())
    axis_limits = get_axis_limits(rollouts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["y"])
    ax.set_zlim(axis_limits["z"])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Optimized algorithms - 3D tracking")

    lines = {}
    trails = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (algo, r) in enumerate(rollouts.items()):
        lines[algo], = ax.plot([], [], [], label=algo, color=colors[i % len(colors)], linewidth=2)
        trails[algo], = ax.plot([], [], [], linestyle="-", color=colors[i % len(colors)], linewidth=6, alpha=0.3)

    # optional target (first model that has it)
    target_line = None
    for r in rollouts.values():
        if r["targets"] is not None:
            target = r["targets"]
            target_line, = ax.plot(target[:, 0], target[:, 1], target[:, 2], "--", color="k", linewidth=1.5, label="Target")
            break

    ax.legend()

    def update(frame):
        for algo, r in rollouts.items():
            pos = r["positions"]
            if frame < len(pos):
                x, y, z = pos[: frame + 1].T
            else:
                x, y, z = pos.T
            lines[algo].set_data(x, y)
            lines[algo].set_3d_properties(z)

            # trail
            start = max(0, frame - trail_length)
            tx, ty, tz = pos[start: frame + 1].T
            trails[algo].set_data(tx, ty)
            trails[algo].set_3d_properties(tz)
        return list(lines.values()) + list(trails.values())

    anim = FuncAnimation(fig, update, frames=min_frames, interval=1000 / fps, blit=False)
    out = ANIMATION_DIR / "optimized_algorithms_3d_tracking.gif"
    writer = PillowWriter(fps=fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    print(f"Saved: {out}", flush=True)


def create_2d_top_view_animation(rollouts, fps=20, trail_length=80):
    print("Creating 2D top-view animation...", flush=True)
    min_frames = min(len(r["positions"]) for r in rollouts.values())
    axis_limits = get_axis_limits(rollouts)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["y"])
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Top view - optimized algorithms")

    lines = {}
    trails = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (algo, r) in enumerate(rollouts.items()):
        lines[algo], = ax.plot([], [], label=algo, color=colors[i % len(colors)], linewidth=2)
        trails[algo], = ax.plot([], [], color=colors[i % len(colors)], linewidth=6, alpha=0.3)

    # target line if any
    for r in rollouts.values():
        if r["targets"] is not None:
            targ = r["targets"]
            ax.plot(targ[:, 0], targ[:, 1], "--", color="k", linewidth=1.5, label="Target")
            break

    ax.legend()

    def update(frame):
        for algo, r in rollouts.items():
            pos = r["positions"]
            x, y = pos[: frame + 1, 0], pos[: frame + 1, 1]
            start = max(0, frame - trail_length)
            tx, ty = pos[start: frame + 1, 0], pos[start: frame + 1, 1]
            lines[algo].set_data(x, y)
            trails[algo].set_data(tx, ty)
        return list(lines.values()) + list(trails.values())

    anim = FuncAnimation(fig, update, frames=min_frames, interval=1000 / fps, blit=False)
    out = ANIMATION_DIR / "optimized_algorithms_top_view.gif"
    writer = PillowWriter(fps=fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    print(f"Saved: {out}", flush=True)


def main():
    models = load_models()
    rollouts = prepare_rollouts(models, max_steps=500)
    create_3d_animation(rollouts, fps=20, trail_length=80)
    create_2d_top_view_animation(rollouts, fps=20, trail_length=80)


if __name__ == "__main__":
    main()