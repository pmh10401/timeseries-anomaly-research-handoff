from validated_exp93_chain import configure_exp93
import run_experiment_105_score_combination_methods as target

EXPERIMENT_ID = "experiment_125_exp105_validated_rank_combinations"
configure_exp93(target, EXPERIMENT_ID)

if __name__ == "__main__":
    target.run_experiment()
