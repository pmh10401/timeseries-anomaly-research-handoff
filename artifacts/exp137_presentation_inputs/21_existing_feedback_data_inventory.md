# Existing Feedback Data Inventory

Search scope: `/Users/minho/Documents/Dataset` (top-level and one subdirectory level) for alert, feedback, maintenance, recipe, equipment, action, and label artifacts; SQLite schema inspection for `univariate_ts.db`.

## Found

- `univariate_ts.db`: `datasets` and `instances` only. Available fields are dataset name, split, instance index, label, and value/label blobs plus dataset-level count/length metadata.
- Experiment result CSVs: model candidate indices, TP/FP/FN metrics, thresholds, train counts, and selector reasons.

## Not found / not verifiable

- persistent alert history distinct from experiment CSVs
- user confirmation or user ID
- maintenance or equipment action record
- recipe/equipment metadata
- operational timestamp or run ID
- join key connecting an alert to a human or maintenance action

`experiment_66_train_normal_alert_budget_*` files found by filename are experiment outputs, not user feedback records.