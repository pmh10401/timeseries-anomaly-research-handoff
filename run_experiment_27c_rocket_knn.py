import os

os.environ["ROCKET_EXPERIMENT_ID"] = "experiment_27c_rocket_knn"
os.environ["ROCKET_CONFIG_NAME"] = "rocket_256_knn5"
os.environ["ROCKET_SCORE_MODE"] = "knn"
os.environ["ROCKET_NUM_KERNELS"] = "256"
os.environ["ROCKET_KNN_NEIGHBORS"] = "5"

import run_rocket_variant_experiment as runner

if __name__ == "__main__":
    runner.main()
