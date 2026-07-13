import sys

from run_model_hard_research_experiments import main_for_experiment


if __name__ == "__main__":
    argv = sys.argv[1:] or ["--workers", "1"]
    main_for_experiment("experiment_111b_vit_manifold_compression_probe", argv)
