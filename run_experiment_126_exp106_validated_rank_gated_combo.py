from validated_exp93_chain import DATA_DIR, configure_exp93
import run_experiment_106_gated_score_combo_selector as target

EXPERIMENT_ID = "experiment_126_exp106_validated_rank_gated_combo"
configure_exp93(target, EXPERIMENT_ID)
target.EXP95_PATH = DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv"
target.EXP103_PATH = DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_results.csv"
target.EXP105_PATH = DATA_DIR / "experiment_125_exp105_validated_rank_combinations_results.csv"

if __name__ == "__main__":
    target.run_experiment()
