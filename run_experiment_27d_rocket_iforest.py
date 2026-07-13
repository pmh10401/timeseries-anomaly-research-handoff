import os

os.environ["ROCKET_EXPERIMENT_ID"] = "experiment_27d_rocket_iforest"
os.environ["ROCKET_CONFIG_NAME"] = "rocket_256_iforest"
os.environ["ROCKET_SCORE_MODE"] = "iforest"
os.environ["ROCKET_NUM_KERNELS"] = "256"

import run_rocket_variant_experiment as runner

if __name__ == "__main__":
    runner.main()
