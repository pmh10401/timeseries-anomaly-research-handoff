import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from run_experiment_26_rocket import (
    DETAIL_OUT_PATH,
    SUMMARY_OUT_PATH,
    STRATEGIES,
    load_dataset_names,
    run_dataset,
    summarize,
    write_csv,
)


DATA_DIR = "/Users/minho/Documents/Dataset"
LOG_PATH = f"{DATA_DIR}/experiment_26_rocket_parallel.log"
WORKERS = int(os.environ.get("ROCKET_WORKERS", "4"))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment26RocketParallel")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def append_rows(path, rows, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_one(dataset_name):
    return dataset_name, run_dataset(dataset_name)


def main():
    for path in [DETAIL_OUT_PATH, SUMMARY_OUT_PATH]:
        if os.path.exists(path):
            os.remove(path)

    dataset_names = load_dataset_names()
    logger.info("Starting Experiment 26 ROCKET parallel on %d datasets with %d workers.", len(dataset_names), WORKERS)
    detail_rows = []
    fieldnames = None
    completed = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, name): name for name in dataset_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                dataset_name, rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
                rows = []
                dataset_name = name
            completed += 1
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
            logger.info("Dataset done: %s | progress [%4d/%4d] rows=%d", dataset_name, completed, len(dataset_names), len(detail_rows))

            if completed % 25 == 0 or completed == len(dataset_names):
                summary_rows = summarize(detail_rows)
                write_csv(SUMMARY_OUT_PATH, summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s meanF1=%.4f medianF1=%.4f zero=%d",
                        completed,
                        len(dataset_names),
                        len(detail_rows),
                        best["strategy"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["zero_f1_count"],
                    )

    logger.info("Experiment 26 ROCKET parallel finished.")


if __name__ == "__main__":
    main()
