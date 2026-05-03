import argparse
import json
import os

# Safer threading behavior on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from envs.circle_tracking_env import CircleTracking3DEnv


def make_env(max_steps=500):
    env = CircleTracking3DEnv(max_steps=max_steps)
    env = Monitor(env)
    return env


def load_best_params(algo_name, n_actions=None):
    algo_name = algo_name.lower()

    path = f"tuning_results/best_{algo_name}_params.json"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best params file not found: {path}\n"
            f"Run Bayesian optimization first for {algo_name.upper()}."
        )

    print("\n" + "=" * 80, flush=True)
    print(f"Loading optimized hyperparameters from: {path}", flush=True)
    print("=" * 80, flush=True)

    with open(path, "r") as f:
        params = json.load(f)

    # TD3 and DDPG share TD3-style params here (DDPG mapped to TD3)
    if algo_name in ("td3", "ddpg"):
        action_noise_sigma = params.pop("action_noise_sigma", 0.2)

        params["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_sigma * np.ones(n_actions),
        )

        params["train_freq"] = (1, "step")
        params["gradient_steps"] = 1

    if algo_name == "sac":
        params["train_freq"] = 1
        params["gradient_steps"] = 1

    return params


def print_params(params):
    print("\nOptimized parameters used for final training:", flush=True)
    print("-" * 80, flush=True)

    for key, value in params.items():
        print(f"{key}: {value}", flush=True)

    print("-" * 80, flush=True)


def build_model(algo_name, env, params):
    algo_name = algo_name.upper()

    common_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 2,
        "tensorboard_log": None,
        "seed": 42,
        "device": "cpu",
    }

    if algo_name == "TD3":
        return TD3(**common_params, **params)

    if algo_name == "SAC":
        return SAC(**common_params, **params)

    # There is no DDPG in stable_baselines3 — treat "DDPG" as TD3
    if algo_name == "DDPG":
        return TD3(**common_params, **params)

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["TD3", "SAC", "DDPG", "td3", "sac", "ddpg"],
        help="Algorithm to train using best tuned hyperparameters",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Final training timesteps",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode",
    )

    args = parser.parse_args()

    algo_name = args.algo.upper()
    algo_lower = algo_name.lower()

    models_dir = f"models/{algo_lower}_circle_optimized"
    logs_dir = f"logs/{algo_lower}_circle_optimized"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("FINAL MODEL TRAINING STARTED", flush=True)
    print("=" * 80, flush=True)
    print(f"Algorithm: {algo_name}", flush=True)
    print(f"Training timesteps: {args.timesteps}", flush=True)
    print(f"Max steps per episode: {args.max_steps}", flush=True)
    print(f"Model output directory: {models_dir}", flush=True)
    print(f"Log directory: {logs_dir}", flush=True)

    print("\nChecking custom Gymnasium environment...", flush=True)
    check_env(CircleTracking3DEnv(max_steps=args.max_steps), warn=True)
    print("Environment check completed.", flush=True)

    env = make_env(max_steps=args.max_steps)
    eval_env = make_env(max_steps=args.max_steps)

    n_actions = env.action_space.shape[-1]

    params = load_best_params(
        algo_name=algo_lower,
        n_actions=n_actions,
    )

    print_params(params)

    print("\nBuilding model...", flush=True)

    model = build_model(
        algo_name=algo_name,
        env=env,
        params=params,
    )

    print("Model created successfully.", flush=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=models_dir,
        name_prefix=f"{algo_lower}_optimized_checkpoint",
        verbose=1,
    )

    print("\n" + "=" * 80, flush=True)
    print("STARTING FINAL LEARNING", flush=True)
    print("=" * 80, flush=True)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"{algo_name}_Optimized",
        log_interval=1,
        progress_bar=False,
    )

    final_model_path = os.path.join(
        models_dir,
        f"{algo_lower}_circle_optimized_final_model",
    )

    model.save(final_model_path)

    env.close()
    eval_env.close()

    print("\n" + "=" * 80, flush=True)
    print("FINAL MODEL TRAINING FINISHED", flush=True)
    print("=" * 80, flush=True)
    print(f"Final optimized {algo_name} model saved to:", flush=True)
    print(final_model_path, flush=True)


if __name__ == "__main__":
    main()