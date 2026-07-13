from validated_exp93_chain import configure_exp93
import run_experiment_95_topk_review_tier as target

EXPERIMENT_ID = "experiment_121_exp95_validated_rank_review"
configure_exp93(target, EXPERIMENT_ID)

if __name__ == "__main__":
    target.run_experiment()
