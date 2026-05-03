import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from envs.circle_tracking_env import CircleTracking3DEnv


warnings.filterwarnings("ignore")


def make_env(max_steps=500):
    env = CircleTracking3DEnv(max_steps=max_steps)
    env = Monitor(env)
    return env


def sample_td3_params(trial, n_actions):
    action_noise_sigma = trial.suggest_float("action_noise_sigma", 0.05, 0.5)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [500, 1_000, 2_000, 5_000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 3]),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.05, 0.4),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.1, 0.7),
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "action_noise": NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_sigma * np.ones(n_actions),
        ),
    }

    return params


def sample_sac_params(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [500, 1_000, 2_000, 5_000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", "auto_0.1", "auto_0.5"]),
        "train_freq": 1,
        "gradient_steps": 1,
    }

    return params


def build_model(algo_name, env, params, tensorboard_log=None, seed=42):
    algo_name = algo_name.upper()

    common_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 2,
        "tensorboard_log": tensorboard_log,
        "seed": seed,
        "device": "auto",
    }

    if algo_name == "TD3":
        return TD3(**common_params, **params)

    if algo_name == "SAC":
        return SAC(**common_params, **params)

    # Map DDPG to TD3 (stable-baselines3 has no native DDPG)
    if algo_name == "DDPG":
        return TD3(**common_params, **params)

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def get_params(algo_name, trial, env):
    algo_name = algo_name.upper()

    if algo_name == "TD3":
        n_actions = env.action_space.shape[-1]
        return sample_td3_params(trial, n_actions)

    if algo_name == "SAC":
        return sample_sac_params(trial)

    # For DDPG use TD3 parameterization
    if algo_name == "DDPG":
        n_actions = env.action_space.shape[-1]
        return sample_td3_params(trial, n_actions)

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def print_trial_params(trial_number, params):
    printable_params = {}

    for key, value in params.items():
        if key == "action_noise":
            printable_params[key] = str(value)
        else:
            printable_params[key] = value

    print("\n" + "=" * 80, flush=True)
    print(f"TRIAL {trial_number} PARAMETERS", flush=True)
    print("=" * 80, flush=True)

    for key, value in printable_params.items():
        print(f"{key}: {value}", flush=True)


def objective(trial, algo_name, train_timesteps, eval_episodes, max_steps):
    print("\n" + "#" * 80, flush=True)
    print(f"Starting trial {trial.number} for {algo_name.upper()}", flush=True)
    print("#" * 80, flush=True)

    env = make_env(max_steps=max_steps)
    eval_env = make_env(max_steps=max_steps)

    params = get_params(algo_name, trial, env)
    print_trial_params(trial.number, params)

    model = build_model(
        algo_name=algo_name,
        env=env,
        params=params,
        tensorboard_log=None,
        seed=trial.number,
    )

    try:
        print("\n" + "-" * 80, flush=True)
        print(f"Training trial {trial.number} for {train_timesteps} timesteps...", flush=True)
        print("-" * 80, flush=True)

        model.learn(
            total_timesteps=train_timesteps,
            log_interval=1,
            progress_bar=True,
        )

        print("\n" + "-" * 80, flush=True)
        print(f"Evaluating trial {trial.number} for {eval_episodes} episodes...", flush=True)
        print("-" * 80, flush=True)

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )

        trial.set_user_attr("std_reward", float(std_reward))

        score = float(mean_reward)

        print("\n" + "=" * 80, flush=True)
        print(f"TRIAL {trial.number} FINISHED", flush=True)
        print(f"Algorithm: {algo_name.upper()}", flush=True)
        print(f"Mean reward: {mean_reward:.4f}", flush=True)
        print(f"Std reward: {std_reward:.4f}", flush=True)
        print("=" * 80, flush=True)

    except Exception as exc:
        print("\n" + "!" * 80, flush=True)
        print(f"TRIAL {trial.number} FAILED", flush=True)
        print(f"Error: {exc}", flush=True)
        print("!" * 80, flush=True)

        score = -1e9

    finally:
        env.close()
        eval_env.close()

    return score


def save_results(study, algo_name):
    algo_name = algo_name.upper()

    os.makedirs("tuning_results", exist_ok=True)

    best_params_path = f"tuning_results/best_{algo_name.lower()}_params.json"
    trials_csv_path = f"tuning_results/{algo_name.lower()}_optuna_trials.csv"

    best_params = study.best_params

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    df = study.trials_dataframe()
    df.to_csv(trials_csv_path, index=False)

    print("\n" + "=" * 80, flush=True)
    print("OPTIMIZATION FINISHED", flush=True)
    print("=" * 80, flush=True)
    print(f"Algorithm: {algo_name}", flush=True)
    print(f"Number of trials: {len(study.trials)}", flush=True)
    print(f"Best reward: {study.best_value:.4f}", flush=True)
    print(f"Best params saved to: {best_params_path}", flush=True)
    print(f"All trials saved to: {trials_csv_path}", flush=True)

    print("\nBest hyperparameters:", flush=True)
    print(json.dumps(best_params, indent=4), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["TD3", "SAC", "DDPG", "PPO", "td3", "sac", "ddpg", "ppo"],
        help="Algorithm to tune",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Bayesian optimization trials",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="Training timesteps per trial",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes per trial",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode",
    )

    args = parser.parse_args()

    algo_name = args.algo.upper()

    os.makedirs("tuning_results", exist_ok=True)

    storage_path = f"sqlite:///tuning_results/{algo_name.lower()}_study.db"

    sampler = optuna.samplers.TPESampler(seed=42)

    print("\n" + "=" * 80, flush=True)
    print("STARTING BAYESIAN OPTIMIZATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Algorithm: {algo_name}", flush=True)
    print(f"Trials: {args.trials}", flush=True)
    print(f"Timesteps per trial: {args.timesteps}", flush=True)
    print(f"Evaluation episodes: {args.eval_episodes}", flush=True)
    print(f"Max steps per episode: {args.max_steps}", flush=True)
    print(f"Storage: {storage_path}", flush=True)

    study = optuna.create_study(
        study_name=f"{algo_name}_circle_tracking_optimization",
        direction="maximize",
        sampler=sampler,
        storage=storage_path,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            algo_name=algo_name,
            train_timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
        ),
        n_trials=args.trials,
    )

    save_results(study, algo_name)


if __name__ == "__main__":
    main()