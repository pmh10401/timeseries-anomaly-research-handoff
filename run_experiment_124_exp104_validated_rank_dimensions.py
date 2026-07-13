from validated_exp93_chain import configure_exp93
import run_experiment_104_score_dimensionality_sweep as target

EXPERIMENT_ID = "experiment_124_exp104_validated_rank_dimensions"
configure_exp93(target, EXPERIMENT_ID)

if __name__ == "__main__":
    target.run_experiment()
