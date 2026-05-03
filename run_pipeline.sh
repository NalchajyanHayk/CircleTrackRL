
set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Running Bayesian optimization..."
echo "============================================================"

python -u -m tuning.tune_circle_optuna --algo TD3 --trials 10 --timesteps 20000 --eval-episodes 3
python -u -m tuning.tune_circle_optuna --algo SAC --trials 10 --timesteps 20000 --eval-episodes 3
python -u -m tuning.tune_circle_optuna --algo DDPG --trials 10 --timesteps 20000 --eval-episodes 3

echo "============================================================"
echo "Training final models using best optimized parameters..."
echo "============================================================"

python -u -m training.train_best_circle_model --algo TD3 --timesteps 300000
python -u -m training.train_best_circle_model --algo SAC --timesteps 300000
python -u -m training.train_best_circle_model --algo DDPG --timesteps 300000

echo "============================================================"
echo "Generating visualizations..."
echo "============================================================"

python -u visualization/plot_optimized_results.py
python -u visualization/animate_optimized_comparison.py

echo "============================================================"
echo "Pipeline finished."
echo "============================================================"