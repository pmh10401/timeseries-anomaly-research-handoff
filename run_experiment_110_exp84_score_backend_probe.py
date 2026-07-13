import sys

from run_model_hard_research_experiments import main_for_experiment


if __name__ == "__main__":
    argv = sys.argv[1:] or ["--workers", "1"]
    main_for_experiment("experiment_110_exp84_score_backend_probe", argv)
