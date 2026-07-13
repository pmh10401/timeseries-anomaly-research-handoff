# Current Tasks
Last updated: 2026-07-10 14:14 KST

- Runtime policy: future rank experiments default to `6` worker processes through `RANK_EXPERIMENT_WORKERS`. This uses all four performance cores plus two efficiency cores; callers can still override the environment value for a constrained run.

## Running

- `experiment_112_parametric_umap_oof_probe`
  - Runs in the isolated `.venv-parametric-umap` Python 3.11 environment with TensorFlow Metal GPU and the existing PyTorch ViT-B/32 extractor.
  - Targets `15` datasets with at least `200` normal training samples: `8` from train count `200-1000` and `7` from `1001+`.
  - Compares Gaussian RP64 and PCA64 with Parametric UMAP8 direct calibration and Parametric UMAP8 3-fold out-of-fold normal calibration.
  - Parametric UMAP is configured for a short reproducible encoder probe (`n_epochs=10`, one Keras training cycle); OOF applies each fold encoder to heldout normal samples and test samples.
  - Expected detail rows: `180`.

## Queue

- `experiment_119a_exp93_rank_order_validation`
  - Replays Exp93 with stable score-descending candidate ranks and compares the result to the historical unordered-top-set output.
- `experiment_119b_rocket256_512_validated_rank_compare`
  - Starts only after Exp119a reaches full coverage; compares 256, 512, and a single-vote 256/512 ROCKET-family tie-breaker under the validated rank rule.

## Deferred

- `experiment_109_vit_compression_alternatives`
  - Deferred because the full ViT compression matrix is too slow for the current iteration cycle.
  - The UMAP small-train failure is fixed: effective dimension is `min(64, train_count - 2)`, with requested and effective dimensions recorded separately.
  - Partial outputs and logs are retained; the experiment was removed from the active queue at `2026-07-10 10:22 KST`.

## Recently Completed

- `experiment_137_operational_triage`
  - Completed `1117/1117` datasets and `1117` detail rows through the sequential runner with no errors, missing datasets, tier overlap, or label/position/family-performance routing leakage.
  - Autonomous Hard alert: `2005` alerts, `1691` TP, `314` FP, precision `84.339%`, mean F1 `0.600772`, zero-F1 `362`.
  - Standard review: `639` requests, `292` TP, `347` FP, precision `45.696%`.
  - High-priority review: `9` requests, `8` TP, `1` FP, precision `88.889%`.
  - Human-assisted diagnostic union: mean F1 `0.700776`, zero-F1 `233`, mean FP `0.592659`. This is not autonomous model performance.
  - Operating interpretation: use Block-B-supported Exp93 indices as Hard alerts and unsupported Exp93 indices as Standard review. Keep the narrow Exp135 branch as research-backed Priority review until genuinely new equipment, recipe, or time-period data validates it.
- `experiment_136_family_holdout_review_audit`
  - Completed `1117/1117` datasets, `2234` rows, no errors, through the sequential queue.
  - The fixed Exp135 `all-standard + Block-C` review rule kept `8` TP and `1` FP (`88.9%` review precision) across deterministic five-fold family partitions; fold precision ranged from `50%` to `100%`.
  - This is a retrospective robustness audit, not prospective validation: the rule was discovered after viewing the current evaluation results. Keep it research-only until a separate future dataset validates it.

- `experiment_111b_vit_manifold_compression_probe`
  - Completed `45/45` datasets, `1215` rows, no missing datasets, errors, or PCA fallbacks, in `9.08` minutes.
  - Overall winner remained Gaussian RP64 with `count_cap_3pct`: mean F1 `0.269801`, zero-F1 `24`, mean FP `3.066667`.
  - Best nonlinear candidate was Isomap16 with `count_cap_3pct`: mean F1 `0.159476`, zero-F1 `21`, mean FP `5.2`; it did not beat PCA64 or Gaussian RP64.
  - UMAP8/16 produced severe FP inflation (`41.6-68.0` mean FP at `count_cap_3pct`) and poor seed stability: mean selected-index Jaccard `0.402299`, identical selections `0/45`.
  - KernelPCA16 and LLE8 were conservative but missed too many anomalies; neither is an operating candidate.
- `experiment_111_vit_fast_compression_probe`
  - Completed `90/90` datasets, `1890` rows, with no missing datasets or errors in `5.94` minutes.
  - Mean-F1/FP winner: Gaussian RP64 with `count_cap_3pct`, mean F1 `0.333484`, median F1 `0.119298`, zero-F1 `43`, mean FP `2.411111`.
  - Sensitivity winner: PCA128 with `count_cap_3pct`, mean F1 `0.329274`, median F1 `0.333333`, zero-F1 `1`, but mean FP `7.111111`.
  - Interpretation: Gaussian RP is the conservative/review candidate; PCA128 is the sensitive candidate and needs an FP guard before operational use.
- `experiment_110_exp84_score_backend_probe`
  - Completed `90/90` datasets, `1620` rows, with no missing datasets or errors in `4.89` minutes.
  - Overall winner remained Exp84 local-gap KNN3 with `count_cap_3pct`: mean F1 `0.570025`, median F1 `0.666667`, zero-F1 `19`, mean FP `2.366667`.
  - On train count `<=10`, LOF novelty led with mean F1 `0.577778` versus KNN `0.555556`; this is a conditional tiny-train signal, not evidence for broad KNN replacement.
- `experiment_108_imaging_pretrained_vit_feature_probe`
  - Pretrained ViT-B/32 image feature probe completed on the hard subset (`339` datasets).
  - Inputs: `spectrogram`, `rp`, and `spectrogram|gasf|rp`; score path: ViT embedding -> PCA64 -> train-normal KNN score -> automatic threshold.

- `experiment_81_aeon_multirocket_official_full`
  - Official aeon MultiROCKET full benchmark completed.
  - Details: [Exp81 official MultiROCKET](experiments/exp81_aeon_multirocket_official_full.md)
- `experiment_82_hydra_hard_family_subset`
  - HYDRA feature extractor on Phoneme, CricketZ, InlineSkate, GestureMidAirD3 completed.
  - Details: [Exp82 HYDRA](experiments/exp82_hydra_hard_family_subset.md)
- `experiment_83_multirocket_hydra_hard339`
  - MultiROCKET + HYDRA concatenated hard339 benchmark completed.
  - Details: [Exp83 MultiROCKET+HYDRA](experiments/exp83_multirocket_hydra_hard339.md)
- `experiment_84_feature_pruning_operational_stability`
  - Stable-tail feature pruning completed and became the strongest hard-subset specialist.
  - Details: [Exp84 feature pruning](experiments/exp84_feature_pruning_operational_stability.md)
- `experiment_85_exp84_hard_specialist_selector`
  - Exp84 specialist was combined with Exp74d full operational baseline as a selector experiment.
  - Best candidate: `gain_family_exp84_cap3_else_primary`, mean F1 `0.677638`, zero-F1 `246`, mean FP `0.667860`.
  - Details: [Exp85 hard specialist selector](experiments/exp85_exp84_hard_specialist_selector.md)
- `experiment_86_train_only_agreement_exp84_selector`
  - Exp84 selector without family-performance prior. Uses train-only safety and Exp74d internal agreement/no-alert signals.
  - Best candidate: `exp84_noalert_repair_te015_fg_else_primary`, mean F1 `0.668192`, zero-F1 `271`, mean FP `0.541629`.
  - Main finding: broad train-safe/agreement switching reduced zero-F1 but raised FP enough to lose mean F1; no-alert repair is safer but small.
- `experiment_87_exp84_index_diagnostics`
  - Exp84 best specialist config rerun with `selected_indices`, score-ranked top indices, and score margins.
  - Completed `339/339` datasets, `1017` detail rows.
  - Best row matches Exp84 best: `count_cap_3pct`, mean F1 `0.576015`, zero-F1 `52`, mean FP `2.165192`.
  - Details: [Exp87 index diagnostics](experiments/exp87_exp84_index_diagnostics.md)
- `experiment_88_true_agreement_exp84_selector`
  - True index-level agreement selectors using Exp74d `selected_indices` and Exp87 `selected_indices`.
  - Best candidate: `top1_noalert_margin_fg_else_primary`, mean F1 `0.668192`, zero-F1 `271`, mean FP `0.541629`.
  - Main finding: pure agreement lowers FP but loses too many TP; top-1 no-alert repair remains the safest small gain.
  - Details: [Exp88 true agreement selector](experiments/exp88_true_agreement_exp84_selector.md)
- `experiment_89_74d_with_exp84_candidate`
  - Exp84 was added as a fourth candidate source inside the Exp74d-style selector family.
  - Best candidate: `exp84_four_model_fg_plus_noalert_repair`, mean F1 `0.671613`, median F1 `1.0`, zero-F1 `271`, mean FP `0.552372`.
  - Compared with Exp74d, it improved 34 datasets, worsened 6, and left 1077 unchanged.
  - Main finding: fully including Exp84 as a guarded candidate is better than only using it as a confidence annotation, but FP cost means it should remain guarded.
  - Details: [Exp89 74d with Exp84 candidate](experiments/exp89_74d_with_exp84_candidate.md)
- `experiment_90_zero_f1_repair_selector`
  - Zero-F1 repair selectors were tested on top of Exp89.
  - Highest F1 candidate: `candidate_union_rerank_repair`, mean F1 `0.697217`, zero-F1 `230`, mean FP `0.627574`.
  - Operationally cleaner candidate: `noalert_top1_train_safe_repair`, mean F1 `0.696978`, zero-F1 `240`, mean FP `0.591764`.
  - Main finding: no-alert top-1 repair fixed 31 zero-F1 datasets with no per-dataset F1 regressions; candidate-union fixed 41 zero-F1 datasets but worsened 22 non-zero datasets.
  - Details: [Exp90 zero-F1 repair selector](experiments/exp90_zero_f1_repair_selector.md)
- `experiment_93_nonpos_candidate_reranker`
  - Non-position candidate reranking was tested on top of Exp90.
  - Best operating candidate: `nonpos_weak_alert_replace`, mean F1 `0.697575`, zero-F1 `239`, mean FP `0.590868`.
  - Compared with Exp90 operating baseline, it improved 1 dataset, worsened 0, fixed 1 zero-F1 row, and reduced total FP by 1.
  - Main finding: candidate expansion is dangerous as hard alerts, but very conservative weak-alert replacement gives a small clean operating improvement.
  - Details: [Exp93 non-position candidate reranker](experiments/exp93_nonpos_candidate_reranker.md)
- `experiment_91_guarded_candidate_union_repair`
  - Candidate-union repair was guarded with tail-window add/replace rules.
  - Best candidate: `noalert_plus_sparse_tail_replace`, mean F1 `0.706528`, zero-F1 `230`, mean FP `0.582811`.
  - Status: rejected for operating use because tail-window replacement exploits the synthetic evaluation layout.
- `experiment_92_operational_hybrid_selector`
  - Final hybrid combines Exp90 no-alert top-1 repair with Exp91 sparse tail replacement only.
  - Best candidate: `operational_noalert_top1_sparse_tail_replace`, mean F1 `0.707721`, zero-F1 `228`, mean FP `0.581021`.
  - Compared with Exp89, it improved 43 datasets, worsened 0, and fixed 43 zero-F1 datasets.
  - Compared with Exp90 no-alert top-1, it improved 12 datasets, worsened 0, and fixed 12 additional zero-F1 datasets.
  - Status: rejected for operating use because the gain comes from tail-position replacement, which is not a valid general anomaly-detection rule.
- `experiment_99_spectral_derivative_feature_score`
  - Spectral/derivative train-normal robust feature score completed.
  - Best research-only selector: `research_spectral_q98_cap2`, mean F1 `0.699962`, zero-F1 `235`, mean FP `0.595345`.
  - Compared with Exp93, it improved `4` datasets, worsened `0`, fixed `4` zero-F1 rows, and added `5` FP.
  - Broad train-family activation failed badly, creating over `100` new zero-F1 regressions depending on threshold.
  - Main finding: spectral/derivative features are useful as a narrow repair or review signal, not as a broad hard-alert replacement.
  - Details: [Exp99 spectral/derivative feature score](experiments/exp99_spectral_derivative_feature_score.md)
- `experiment_100_spectral_review_and_guard`
  - Exp99 follow-up completed: spectral candidates were tested as review-only additions and guarded hard-alert replacements.
  - Best operating-safe direction: `review_spectral_family_when_exp93_weak_agrees`.
  - It keeps Exp93 hard alerts unchanged, adds only `12` review candidates across `1117` datasets, and finds `2` true anomaly review hits.
  - Guarded hard replacement improved `2` datasets but worsened `1`, so it is not clean enough for default use.
  - Details: [Exp100 spectral review and guard](experiments/exp100_spectral_review_and_guard.md)
- `experiment_101_shapelet_normal_prototype`
  - Shapelet normal prototype feature score completed.
  - Research selector fixed `2` zero-F1 rows with no regressions, but broad train-family activation caused large regressions.
  - Main finding: shapelet is a narrow review/specialist signal, not a broad hard-alert replacement.
  - Details: [Exp101 shapelet normal prototype](experiments/exp101_shapelet_normal_prototype.md)
- `experiment_102_feature_source_selector`
  - Spectral and shapelet sources were combined into a feature source selector.
  - Best operating-safe direction: `review_spectral_shapelet_weak_agreement`, which adds `14` review candidates across `1117` datasets and finds `2` true hits.
  - Hard feature-source replacement improved `2` datasets but worsened `1`, so it remains diagnostic only.
  - Details: [Exp102 feature source selector](experiments/exp102_feature_source_selector.md)
- `experiment_103_higher_dim_review_sources`
  - Higher-dimensional feature sources were tested as review-lane candidates.
  - Best review direction: `review_all_higher_dim_sources_when_exp93_weak`.
  - It keeps Exp93 hard alerts unchanged, adds `162` review candidates across `1117` datasets, and finds `48` true review hits.
  - Combined review F1 improves from `0.697575` to `0.716644`, and combined zero-F1 drops from `239` to `191`.
  - This is a review-lane improvement, not an automatic hard-alert replacement.
  - Details: [Exp103 higher-dim review sources](experiments/exp103_higher_dim_review_sources.md)
- `experiment_104_score_dimensionality_sweep`
  - Score sources were evaluated directly at 64/128/256 dimensions, plus 64+128+256 rank-mean combinations.
  - Exp93 remains much stronger as the hard-alert baseline: mean F1 `0.697575`, zero-F1 `239`, mean FP `0.591`.
  - Higher-dimensional spectrogram scores recover many zero-F1 cases but create too many false positives for direct hard-alert use.
  - `spectrogram_pca256/count_cap_2pct` fixed `220` Exp93 zero-F1 datasets, but worsened `849` datasets and raised mean FP to `14.371`.
  - Dimension-combination rank means have useful oracle/ranking signal, but thresholded hard-alert F1 is poor.
  - Details: [Exp104 score dimensionality sweep](experiments/exp104_score_dimensionality_sweep.md)
- `experiment_105_score_combination_methods`
  - Alternative 64/128/256 score-combination methods were tested: rank mean/max/min, weighted combinations, and 2-of-3/3-of-3 agreement.
  - Exp93 remains the hard-alert baseline. No combination method is safe as a direct replacement.
  - Best non-baseline mean F1: `glcm_rp_agreement_2of3`, mean F1 `0.364751`, zero-F1 `235`, mean FP `2.852`.
  - `spectrogram_agreement_2of3` fixed `214` Exp93 zero-F1 datasets but mean FP rose to `10.304`; it is too aggressive for hard alerts.
  - `spectrogram_agreement_3of3` is more conservative: fixes `192` zero-F1 datasets with mean FP `5.201`, still too high for direct replacement.
  - `spectrogram_glcm_rp_all_dims_rank_min` is the most useful conservative combination candidate for future gated review/repair.
  - Details: [Exp105 score combination methods](experiments/exp105_score_combination_methods.md)
- `experiment_106_gated_score_combo_selector`
  - Exp105 score combinations were activated only when Exp93 was weak and candidate indices agreed with existing context.
  - Best row: `review_exp103_plus_combo_cap3`, combined mean F1 `0.716644`, combined zero-F1 `191`, review precision `0.296`.
  - Combo-only conservative review: combined mean F1 `0.714137`, combined zero-F1 `198`, `41` review TP and `109` review FP.
  - Combo-only sensitive review: combined mean F1 `0.715570`, combined zero-F1 `194`, `45` review TP and `120` review FP.
  - Hard replacement stayed inactive, which is appropriate under the conservative guard.
  - Main finding: gated score combinations are useful as supporting review evidence, but they do not improve beyond Exp103's review lane.
  - Details: [Exp106 gated score combo selector](experiments/exp106_gated_score_combo_selector.md)
- `experiment_107_exp103_combo_disagreement`
  - Exp103 review candidates were compared with Exp106 combo candidates to find unique combo value.
  - Best row: `review_exp103_plus_unique_sensitive_cap4`, combined mean F1 `0.716823`, combined zero-F1 `189`.
  - Unique sensitive combo candidates found `2` true positives missed by Exp103: `WordSynonyms_normal_1` and `WordSynonyms_normal_15`.
  - Cost: unique sensitive candidates added `33` review candidates with only `2` TP and `31` FP; review precision drops from `29.6%` to `27.9%` when added to Exp103.
  - Main finding: combo has a small unique signal, but it is too noisy for broad default inclusion.
  - Details: [Exp107 Exp103/combo disagreement](experiments/exp107_exp103_combo_disagreement.md)

## Current Decisions

- `experiment_113_train_normal_conformal_fusion` is queued as the first direct Exp93 replacement test. It retains the three all-dataset score sources (ROCKET Exp40, Exp55 spectrogram, Exp56 GLCM/RP), converts each to leave-self-out train-normal tail p-values, then tests Bonferroni min-p, Cauchy, Fisher, and 2-of-3 p-value fusions at `0.5%` and `1%` targets. It uses no labels, family performance tables, test-position rules, or test-count caps. Because ROCKET may cap its normal reference on large datasets, the fused p-values are combined directly instead of incorrectly joining unequal train-score rows.
- Exp113 completed on all `1117` datasets with `10053` rows and no errors. All direct conformal replacements are rejected: the best, Fisher `1%`, reached mean F1 `0.3753` versus Exp93 `0.6976`, increased mean FP from `0.591` to `1.045`, and created `378` new zero-F1 rows. The primary reason is empirical p-value resolution: with `<=10` normal samples, the `317` affected datasets cannot express a `1%` tail and Fisher emitted no alert on all of them. Treat conformal evidence as a possible confidence/review annotation, not as the sole hard-alert rule. Details: [Exp113 train-normal conformal fusion](experiments/exp113_train_normal_conformal_fusion.md)
- Exp113 index-overlap follow-up: direct p-value candidates must not be added as alerts, but when Bonferroni `1%` or 2-of-3 `1%` selected the same index as an existing Exp93 alert, offline precision was `0.886` and `0.865` respectively, versus `0.693` and `0.686` for the corresponding Exp93-only alerts. This supports a confidence badge or review-priority annotation, not a hard-alert replacement.
- `experiment_115_local_normal_state_score` is queued as the next score-representation probe. It compares global ROCKET local-gap with cross-fitted local normal-state GMM (BIC up to 3 or 5 states) and KMeans-3. Datasets with fewer than `30` train normals intentionally fall back to global local-gap. Count-cap outputs are diagnostic only; score AUC-PR and oracle F1 are the primary representation checks before any threshold-policy promotion.
- Exp115 completed on all `1117` datasets with `14521` rows and no errors. Reject local GMM/KMeans normal-state scores: in the active `n>=30` subset, global ROCKET local-gap has mean AUC-PR `0.696` and F1 `0.551`, while best local GMM5 drops to AUC-PR `0.573` and F1 `0.420` with FP `5.489` versus `2.062`. The failure persists at `>=1000` train normals, so it is feature-space fragmentation rather than only small-train instability. Details: [Exp115 local normal-state score](experiments/exp115_local_normal_state_score.md)
- `experiment_114_pseudo_anomaly_score_probe` is queued as the next representation probe. It uses train-normal data only: random-position spike, finite local level shift, and local shuffle perturbations are transformed with a fixed-seed ROCKET feature extractor, then a 3-fold cross-fitted logistic model scores normal-versus-generic-perturbation likelihood. `n<30` uses global local-gap fallback. The probe evaluates score ranking separately from diagnostic thresholds and does not use test anomaly labels or positions during training.
- Exp114 completed with full coverage (`1117` datasets, `11170` rows, no runtime errors). The stdout log contains `5` sklearn `ConvergenceWarning` events out of about `3570` pseudo-LR fold fits; coefficients were still produced, but future pseudo-score scripts should attach dataset/fold identifiers to warnings. The warnings do not explain the result direction: on the active `n>=30` subset, spike+step pseudo score AUC-PR is `0.358` and F1 `0.282`, versus global local-gap AUC-PR `0.694` and F1 `0.561`. Treat this particular generic pseudo-anomaly representation as rejected rather than rerunning solely for the five warnings.
- Exp116/117 are queued train-only controls. Exp116 changes only Exp93 sparse weak-alert reranking weights using normal-score tail stability, bootstrap threshold variability, and train exceed. Exp117 applies the same reliability estimates to Exp89/90 source selection and a bounded candidate budget. Neither uses test labels, family performance, anomaly position, or prior zero-F1 membership.
- Exp116 completed with full coverage (`1117` datasets, `2234` rows, no errors). Train-only adaptive reliability reproduced Exp93 exactly: mean F1 `0.697575`, zero-F1 `239`, mean FP `0.590868`. Its conservative weak-alert gate activated only on the same two rows already changed by Exp93, so it is safe but adds no new operating gain.
- The first Exp117 output was invalid for Exp93 comparison: it displayed Exp93 as the control but started candidate rows from Exp89 selected indices. Those CSV/log files were retained with the suffix `invalid_exp89_exp93_baseline_mismatch_20260710_142300` and are excluded from dashboard/result use.
- Corrected Exp117 reran with full coverage (`1117` datasets, `4468` rows, no errors) using Exp93 indices end-to-end. Both no-alert variants made no changes because Exp93 had already repaired their eligible no-alert rows. Sparse budget-2 changed `4` datasets, improving one F1 row by adding `1` TP but adding `7` FP across three other rows; mean F1 rose only from `0.697575` to `0.697728`, while mean FP rose from `0.590868` to `0.597135`. Reject it as an operating default under the false-alarm-first objective.
- Dashboard labels corrected Exp117 as `배제`, rather than a research candidate, so the small Mean-F1 increase cannot be mistaken for an operating recommendation.
- Dashboard phase 1 completed: the sequential runner records a standard heartbeat JSON, the dashboard prefers it over heuristic log parsing, the Queue view is restricted to actual running/queued experiments, and runtime health exposes server PID/build time/heartbeat age. `dashboardctl.py` provides verified start/stop/restart/status for a single dashboard server without touching experiment processes.
- Dashboard phase 1 validation: `/api/overview` is about `14KB` and responds in about `0.9s`; it is polled every 5 seconds. The approximately `214KB` full status, including completed-result and strategy detail tables, is fetched only on initial load and every 60 seconds. This removes the old Queue pollution from the 112-item experiment catalog while preserving detailed result views.
- Dashboard phase 2 completed: process detail now lists the experiment worker and child workers; CPU is shown as `Proc` core-equivalent usage, so a value over `100%` means more than one logical core rather than an error. Warnings are separated from row-level errors, completed results can be filtered into operating candidates, research candidates, and excluded results, and the mobile layout keeps the monitoring panels readable.
- Exp79 Conv AE epoch sweep is rejected as a primary direction.
- Next improvement axis is controlled specialist selection: keep the full operating baseline, then attach stronger hard-family representations only where they pay for their FP cost.
- Family-performance prior helped Exp85, but Exp86 shows that train-only/agreement signals alone are too weak unless the switch is very conservative.
- Official aeon feature extractors are used as feature extractors only. Classifier wrappers are not used.
- The existing operational scoring path remains: train-normal features -> KNN/local-gap score -> automatic threshold.
- Arsenal remains deferred because it is useful for an upper-bound check but too heavy for an operating default.
- Current operating-default candidate is Exp93 `nonpos_weak_alert_replace`.
- Exp91/Exp92 tail-replacement results are excluded from operating-candidate ranking. They are retained only as a cautionary diagnostic showing how easily evaluation-layout leakage can inflate metrics.
- Exp70 zero-mode family repair and Exp85 gain-family specialist selector are also excluded from operating-candidate ranking because they depend on prior labeled benchmark/failure outcomes.
- Exp90 `candidate_union_rerank_repair` remains a useful research candidate, but it is not the operating default because it worsens 22 datasets and raises FP more than the no-alert-only repair.
- Exp99 confirms a narrow feature-side signal. Use spectral/derivative features as review/specialist evidence only; do not broadly replace Exp93 hard alerts by spectral family.
- Exp100 confirms that the useful operational path is review-lane augmentation, not hard-alert replacement. The current candidate is `review_spectral_family_when_exp93_weak_agrees`.
- Exp101/102 confirm that adding feature sources broadly is not the right direction. Use spectral/shapelet as low-burden review evidence only.
- Exp103 confirms that higher-dimensional feature sources can become useful when used as a review lane. Keep Exp93 as the hard-alert default, and treat `review_all_higher_dim_sources_when_exp93_weak` as the strongest review-lane candidate so far.
- Exp104 confirms the user's intended dimensionality hypothesis in a nuanced way: higher dimensions improve sensitivity to missed anomalies, but not hard-alert stability. The next valid step is gated use of high-dimensional spectrogram scores, not direct replacement of Exp93.
- Exp105 confirms that changing the combination method alone is not enough to replace Exp93. The useful path is gated evidence: activate high-dimensional combinations only when Exp93 is weak and agreement/review guards support the candidate.
- Exp106 confirms that the current gated combo rule does not beat Exp103. The next useful question is disagreement analysis: find cases where Exp106 combo candidates are uniquely correct and Exp103 misses.
- Exp107 found only two unique combo true positives beyond Exp103, both in `WordSynonyms`, with many extra FP. Do not broadly add combo-only candidates; investigate narrow WordSynonyms-like conditions if this family matters operationally.
- Exp112 Parametric UMAP OOF probe completed on 15 large-normal original datasets with all 180 expected rows and no errors. `vit_spectrogram_gaussian_rp64_knn3/count_cap_3pct` remained best: mean F1 `0.3173`, median F1 `0.3396`, zero-F1 `2`, mean FP `7.80`.
- Exp112's direct Parametric UMAP had mean F1 `0.2579` and mean FP `9.73`. The leakage-resistant 3-fold OOF calibration reduced mean FP to `5.33`, but also reduced TP from `74` to `52`, giving mean F1 `0.2165` and zero-F1 `3`. This is a score-separation limitation, not a reason to promote direct in-sample calibration.
- Keep Gaussian random projection as the fast compression baseline. Parametric UMAP is not an operational hard-alert candidate from this probe; its only remaining research value is a larger, better-trained encoder used as a narrowly guarded review feature.

## Watch Points

- aeon 1.5.0 was installed for official MultiROCKET/HYDRA support.
- Installing aeon adjusted the Python 3.14 environment versions for `numpy`, `scipy`, `pandas`, and `scikit-learn`; if unrelated scripts fail later, check dependency compatibility.
- Exp85's best gain-family selector is a research diagnostic because the family list comes from prior benchmark evidence. It should not be promoted directly to production without a train-only or recipe-prior justification.
- Exp86 does not use family-performance prior, but it also cannot do true Exp84-vs-Exp74d index-level agreement because Exp84 result CSV does not store selected indices.
- Exp87 is the index-saving Exp84 rerun needed for true agreement, top-1 repair, and score-margin selector experiments.
- During Exp87, `rank_experiments_sequential_state.json` disappeared after completion; it was rebuilt from existing result CSVs. The experiment output itself remained intact.
- Exp88 shows that strict true-agreement intersection is too conservative as a full replacement: it reduces FP but drops TP and mean F1.
- Exp89 shows that Exp84 can be included as a guarded fourth candidate source, but the gain comes from modest TP recovery rather than broad replacement. Keep ROCKET/Exp74d as the base.
- Exp90 shows that zero-F1 reduction is best attacked through no-alert repair first. Candidate-union reranking is powerful but needs a second guard because it can add FP to already non-zero datasets.
- Exp90 zero-F1 deep dive: the remaining `240` zero-F1 rows are all wrong-alert cases, not no-alert cases. At least one score source ranks the true anomaly inside top-10 for `233/240` rows, so the next valid improvement is non-position candidate reranking/confidence calibration rather than tail replacement. Details: [Exp90 zero-F1 deep dive](experiments/exp90_zero_f1_deep_dive.md)
- Exp93 confirms the safe version of that idea: weak-alert replacement can improve one zero-F1 case with no regressions, while broad top-candidate add/review variants create too many false positives for hard-alert use.
- Exp94/95 analysis: Exp94 hard-alert replacement failed or had no meaningful effect; broad replacement created many new zero-F1 regressions. Exp95 shows review candidates can rescue up to 36 zero-F1 rows diagnostically, but precision is too low for automatic hard alerts. Details: [Exp94/95 deep analysis](experiments/exp94_exp95_deep_analysis.md)
- Exp96 operational review workflow: keep Exp93 as hard-alert default, and use `review_lane_top1_strict` as the recommended separate review lane. It rescues 9 zero-F1 cases with the lowest review load, about 52 review candidates per 100 datasets. Details: [Exp96 review tier workflow](experiments/exp96_review_tier_operational_workflow.md)
- Exp94b corrected the Exp94 hard-alert replacement design issue by scoring existing hard alerts in the candidate pool. The large Exp94 regression disappeared, but there were still no improved datasets and no zero-F1 fixes. Hard-alert replacement should be closed for now. Details: [Exp94b corrected rank consensus](experiments/exp94b_corrected_nonpos_rank_consensus.md)
- Exp97/98 feature transition: Exp97 classified remaining zero-F1 cases. Exp98 showed tiny-train pooling can fix 43 zero-F1 rows in a research-only setting, but Exp98b showed train-only broad application creates too many false positives. Tiny-train pooling should become a gated review/specialist signal, not a hard-alert replacement. Details: [Exp97/98 feature transition](experiments/exp97_exp98_feature_transition.md)
- Exp99 spectral/derivative score fixed four Phoneme/CricketZ zero-F1 cases in the research-only selector, but broad train-family activation damaged many already-correct datasets. Details: [Exp99 spectral/derivative feature score](experiments/exp99_spectral_derivative_feature_score.md)
- Exp100 spectral review/guard: weak+agreement review adds very low review burden and finds 2 true hits; hard replacement still has a regression. Details: [Exp100 spectral review and guard](experiments/exp100_spectral_review_and_guard.md)
- Exp101/102 feature-source tests: shapelet and combined feature sources add narrow recoveries, but hard replacement still has regressions. Details: [Exp101](experiments/exp101_shapelet_normal_prototype.md), [Exp102](experiments/exp102_feature_source_selector.md)
- Exp103 dashboard fix: live progress and current dataset now depend on a real running process. Completed experiments still appear in compare/completed tables, but they no longer appear as live current-dataset progress. Details: [Exp103](experiments/exp103_higher_dim_review_sources.md)
- Exp104 score-dimensionality sweep: `spectrogram_pca64/128/256`, `glcm_rp_pca64/128/256`, and rank-mean combinations completed on all `1117` datasets. Details: [Exp104](experiments/exp104_score_dimensionality_sweep.md)
- Exp105 score-combination methods completed on all `1117` datasets. Details: [Exp105](experiments/exp105_score_combination_methods.md)
- Exp106 gated score-combo selector completed on all `1117` datasets through the sequential queue. Details: [Exp106](experiments/exp106_gated_score_combo_selector.md)
- Exp107 Exp103/combo disagreement completed on all `1117` datasets through the sequential queue. Details: [Exp107](experiments/exp107_exp103_combo_disagreement.md)
- Exp108 ViT imaging feature probe was implemented as a hard-subset experiment. The first `vit_b_16` smoke run was too slow, so the queued version uses pretrained `vit_b_32` with `spectrogram`, `rp`, and `spectrogram|gasf|rp` image inputs, then PCA64 + KNN score. The experiment is intended as a review/specialist feature probe, not an operating hard-alert replacement.
- Exp109 ViT compression alternatives was expanded after review to include the previously excluded research candidates: small autoencoder compression and UMAP. Metric learning is explicitly deferred because it requires labels/pseudo-labels and could violate the normal-only operating constraint. A second smoke run was stopped because it competed with Exp108 for ViT resources; the queued full run will validate the expanded config after Exp108 completes.
- Exp112 used the TensorFlow Metal backend successfully (`tensorflow_gpu_available=1`) with official `ParametricUMAP`. It intentionally used a direct same-encoder path and a 3-fold out-of-fold calibration path so that the calibration comparison is fair. The repeated Keras/UMAP messages were library warnings, not row-level experiment failures.
- Exp91/92 tail replacement is considered invalid for operating use. It depends on the current synthetic evaluation layout where anomalies often appear late in the test sequence, so it is closer to leakage/cheating than a deployable detection rule.
- Operational leakage audit: [2026-07-09 audit](experiments/operational_leakage_audit_20260709.md)
- Smoke outputs for Exp81-84 and temporary Exp85 pre-runs were archived to avoid dashboard/result confusion.

## How To Check

```bash
/opt/homebrew/bin/python3 run_rank_experiments_sequential.py list
tail -n 40 /Users/minho/Documents/Dataset/experiment_90_zero_f1_repair_selector_stdout.log
```

## Archive

- Previous long task history: [task_history_20260709_before_compaction.md](archive/task_history_20260709_before_compaction.md)
