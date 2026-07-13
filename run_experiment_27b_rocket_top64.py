import os

os.environ["ROCKET_EXPERIMENT_ID"] = "experiment_27b_rocket_top64"
os.environ["ROCKET_CONFIG_NAME"] = "rocket_256_robust_top64"
os.environ["ROCKET_SCORE_MODE"] = "robust_topk"
os.environ["ROCKET_NUM_KERNELS"] = "256"
os.environ["ROCKET_TOP_DEVIATIONS"] = "64"

import run_rocket_variant_experiment as runner

if __name__ == "__main__":
    runner.main()
