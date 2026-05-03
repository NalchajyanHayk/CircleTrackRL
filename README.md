# CircleTrackRL

Compact research codebase for training and evaluating continuous-control RL agents on a 3D circular trajectory tracking task (quadrotor-like).

Co‑authors: Hayk Nalchajyan, Liana Zhamkochyan

## Overview
CircleTrackRL provides:
- A lightweight Gymnasium-compatible 3D circular‑tracking environment.
- Optuna-based hyperparameter tuning for continuous-control algorithms (TD3, SAC, DDPG-as‑TD3).
- Final training, automated evaluation, and visualization (plots + animations).
- A reproducible pipeline for research and benchmarking.

## Key features
- Deterministic environment with configurable circle radius, angular speed and noise.
- Dense reward shaping for smooth tracking and low steady-state error.
- Searchable tuning space (learning rate, network size, batch size, tau, etc.).
- Scripts to run full pipeline (tune → train best → evaluate → visualize).

## How it works (brief)
1. The environment (envs/circle_tracking_env.py) provides continuous observations (position, velocity, error to reference) and continuous actions (forces/thrust).
2. Optuna objective (tuning/tune_circle_optuna.py) runs short training/evaluation loops and reports a validation metric (mean cumulative reward) to guide hyperparameter search.
3. training/train_best_circle_model.py loads the best found parameters and trains a final model for a longer budget.
4. Visualization scripts load saved evaluation traces and generate static plots or animated 3D trajectories.

## Quick start

1. Create environment and install dependencies (macOS):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full pipeline (tuning → final training → visualization):
```bash
./run_pipeline.sh
```



## Example (sample) results
Below is a small example summary (sample/simulated values). Replace with your run outputs found in results/*.csv.

- Evaluation over 50 episodes (example):
  - TD3: mean cumulative reward = 124.6 ± 17.3, mean tracking RMSE = 0.12 m
  - SAC: mean cumulative reward = 118.2 ± 20.1, mean tracking RMSE = 0.15 m
  - DDPG-as‑TD3: mean cumulative reward = 110.5 ± 25.4, mean tracking RMSE = 0.18 m

To reproduce real numbers, run the evaluation routine in training/train_best_circle_model.py or the dedicated evaluation entry in visualization scripts. The scripts write CSV summaries to results/ and per-episode traces to logs/.

## Generating plots
Use the included plotting script to create comparison figures:
```bash
python visualization/plot_optimized_results.py \
  --summary results/optimized_model_evaluation_summary.csv \
  --out results/plots/optimized_comparison.png
```
The script expects a CSV with columns like: algorithm,mean_reward,std_reward,mean_rmse,median_rmse.

If you prefer to inspect CSV manually, a minimal Python snippet:
```python
import pandas as pd
df = pd.read_csv('results/optimized_model_evaluation_summary.csv')
print(df[['algorithm','mean_reward','mean_rmse']])
```

## Files and layout
- envs/circle_tracking_env.py — Gym env
- tuning/tune_circle_optuna.py — Optuna tuning
- training/train_best_circle_model.py — final training & evaluation
- visualization/*.py — plotting & animation
- run_pipeline.sh — example pipeline runner
- tuning_results/, models/, results/, logs/ — outputs

## Tips & troubleshooting
- Use a small trial/timesteps budget for quick experiments, then scale up.
- If Optuna reuses a DB, remove tuning_results/*.db to restart.
- Use the Gymnasium environment checker to validate custom env behavior.

## Conclusion
CircleTrackRL provides a compact, reproducible pipeline for benchmarking continuous-control RL on a 3D circular tracking task. The repository emphasizes:
- Reproducible hyperparameter search (Optuna) and straightforward final training.
- Clear separation between tuning, final training, evaluation and visualization.
- Extensible components for new algorithms, reward functions or visualizations.

Next steps:
- Add automated CI tests for the environment dynamics and basic agent training.
- Expand the benchmark with additional reference trajectories (figure‑8, helix).
- Add interactive plots (Plotly) or video exports for presentations.

License and contribution guidelines are not included — add a LICENSE and CONTRIBUTING.md if you plan public distribution.