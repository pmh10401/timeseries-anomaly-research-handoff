from validated_exp93_chain import DATA_DIR, configure_exp93
import run_experiment_107_exp103_combo_disagreement as target

EXPERIMENT_ID = "experiment_127_exp107_validated_rank_disagreement"
configure_exp93(target, EXPERIMENT_ID)
target.EXP103_PATH = DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_results.csv"
target.EXP106_PATH = DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_results.csv"

if __name__ == "__main__":
    target.run_experiment()
