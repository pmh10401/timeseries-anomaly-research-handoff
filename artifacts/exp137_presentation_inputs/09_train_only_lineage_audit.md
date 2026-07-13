# TRAIN-only Lineage Audit

`08_train_only_lineage_audit.csv` is the line-level evidence table.

## Findings

- Exp137 final routing does not use TEST labels, position, or historical family performance. TEST length is used only to reject out-of-bounds indices.
- Candidate score thresholds in Exp40, Exp131, Exp134, and Exp135 derive from TRAIN normal score distributions.
- TEST labels are loaded in several stages for offline metrics, which is acceptable only because selected indices are computed before those metrics.
- **Presentation risk:** Exp93 inherits Exp84 `family_guard_v1` rows, and Exp89/90 ancestor code uses `large_data_budget(y_test, ...)`. These are upstream historical candidate-generation dependencies, not final Exp137 routing conditions. A strict train-only deployment should remove or revalidate these inherited dependencies before promotion.
- **Presentation risk:** Priority review uses a retrospective rule and remains review-only. The `tail` word in Exp134 refers to the TRAIN score tail, not TEST time position.

## Required presentation scope

The evidence verifies the final Exp137 routing boundary: it does not use TEST
labels, TEST anomaly positions, or historical family TEST performance. It does
not yet prove that every upstream candidate-generation dependency in the full
historical lineage is strict train-only. Do not describe the full pipeline as
fully validated until the inherited family guard and test-length-dependent
budget dependencies are removed or independently revalidated.

## Dependency Graph

See `10_experiment_dependency_graph.mmd`.
