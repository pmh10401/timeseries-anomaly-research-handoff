from validated_exp93_chain import DATA_DIR, configure_exp93
import run_experiment_103_higher_dim_review_sources as target

EXPERIMENT_ID = "experiment_123_exp103_validated_rank_review_sources"
configure_exp93(target, EXPERIMENT_ID)
target.EXP95_PATH = DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv"

if __name__ == "__main__":
    target.run_experiment()
