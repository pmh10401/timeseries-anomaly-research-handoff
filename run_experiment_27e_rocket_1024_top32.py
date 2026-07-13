import os

os.environ["ROCKET_EXPERIMENT_ID"] = "experiment_27e_rocket_1024_top32"
os.environ["ROCKET_CONFIG_NAME"] = "rocket_1024_robust_top32"
os.environ["ROCKET_SCORE_MODE"] = "robust_topk"
os.environ["ROCKET_NUM_KERNELS"] = "1024"
os.environ["ROCKET_TOP_DEVIATIONS"] = "32"

import run_rocket_variant_experiment as runner

if __name__ == "__main__":
    runner.main()
