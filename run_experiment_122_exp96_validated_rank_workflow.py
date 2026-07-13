from validated_exp93_chain import DATA_DIR
import run_experiment_96_review_tier_operational_workflow as target

EXPERIMENT_ID = "experiment_122_exp96_validated_rank_workflow"
target.EXPERIMENT_ID = EXPERIMENT_ID
target.STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
target.EXP95_PATH = DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv"

if __name__ == "__main__":
    target.run_experiment()
