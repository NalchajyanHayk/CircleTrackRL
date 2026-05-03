# CircleTrackRL

Compact research codebase for training and evaluating continuous-control RL agents on a 3D circular trajectory tracking task (quadrotor-like).

- Environment: [`envs.circle_tracking_env.CircleTracking3DEnv`](envs/circle_tracking_env.py) — a small Gymnasium environment implementing a moving circular reference and dense reward.
- Hyperparameter tuning: [`tuning.tune_circle_optuna.objective`](tuning/tune_circle_optuna.py) (Optuna-based tuning). See [`tuning/tune_circle_optuna.py`](tuning/tune_circle_optuna.py).
- Final training: [`training.train_best_circle_model.main`](training/train_best_circle_model.py) — loads best params and trains final models. See [`training/train_best_circle_model.py`](training/train_best_circle_model.py).
- Visualization: [`visualization.plot_optimized_results.main`](visualization/plot_optimized_results.py) and [`visualization.animate_optimized_comparison.main`](visualization/animate_optimized_comparison.py). See files: [`visualization/plot_optimized_results.py`](visualization/plot_optimized_results.py), [`visualization/animate_optimized_comparison.py`](visualization/animate_optimized_comparison.py).
- Pipeline runner: [run_pipeline.sh](run_pipeline.sh)
- Requirements: [requirements.txt](requirements.txt)
- Ignore list: [.gitignore](.gitignore)

Repository layout (top-level)
- [envs/circle_tracking_env.py](envs/circle_tracking_env.py)
- [tuning/tune_circle_optuna.py](tuning/tune_circle_optuna.py)
- [training/train_best_circle_model.py](training/train_best_circle_model.py)
- [visualization/plot_optimized_results.py](visualization/plot_optimized_results.py)
- [visualization/animate_optimized_comparison.py](visualization/animate_optimized_comparison.py)
- [run_pipeline.sh](run_pipeline.sh)
- [requirements.txt](requirements.txt)
- [tuning_results/best_td3_params.json](tuning_results/best_td3_params.json)
- [tuning_results/best_sac_params.json](tuning_results/best_sac_params.json)
- [tuning_results/best_ddpg_params.json](tuning_results/best_ddpg_params.json)
- [results/optimized_model_evaluation_summary.csv](results/optimized_model_evaluation_summary.csv)
- [results/summary.csv](results/summary.csv)
- logs/, models/ (output directories)

Quick start (recommended)
1. Create a virtualenv and install dependencies:

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```