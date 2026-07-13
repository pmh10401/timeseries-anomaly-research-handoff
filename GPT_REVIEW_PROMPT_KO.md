# GPT Review Prompt

아래 파일을 함께 읽고 Exp137의 policy-level TRAIN-only 검증 결과를 분석해줘.

우선순위 파일:

1. README.md
2. data/DATABASE_GUIDE.md
3. results/exp137_gpt_handoff_20260713/results/experiment_137_operational_triage_summary.csv
4. results/exp137_policy_train_only_validation/b2_full/18_b2_family_neutral_summary.md
5. results/exp137_policy_train_only_validation/budget_full_7w_optimized/36_test_length_budget_binding_summary.md
6. results/exp137_policy_train_only_validation/budget_full_7w_optimized/39_c0_threshold_only_summary.md
7. results/exp137_policy_train_only_validation/budget_full_7w_optimized/55_d1a_policy_train_only_no_budget_summary.md
8. results/exp137_policy_train_only_validation/BLOCKER_REPORT.md

분석 규칙:

- 목표는 mean F1 하나의 최대화가 아니라 false alarm 감소, 사용자 신뢰, 검토 workload 관리다.
- Hard alert만 autonomous 성능으로 해석한다.
- Standard/Priority review와 mean combined F1은 human-assisted 진단 지표로 분리한다.
- B2, C0, D1a는 retrospective counterfactual이며 prospective 성능으로 과장하지 않는다.
- TEST 성능을 보고 threshold, K, budget, source를 다시 선택하라고 제안하지 않는다.
- 실제 DB에는 wafer/run/recipe/sensor/step metadata가 확인되지 않으므로, 이를 가정한 conformal budget을 정당화하지 않는다.

다음 순서로 답해줘.

1. 현재 수치가 보여 주는 사실과 보여 주지 않는 사실을 구분한다.
2. Exp143에서 TEST-length budget이 실제로 binding된 범위와, FP 감소의 인과 기여를 아직 분리할 수 없는 이유를 설명한다.
3. C0와 D1a에서 budget 제거 후 workload와 precision이 어떻게 변했는지 설명한다.
4. B2 family-neutral 결과의 의미와 한계를 설명한다.
5. 다음 실제 운영 검증에 필요한 metadata와 shadow-validation 기록 항목을 제안한다.
