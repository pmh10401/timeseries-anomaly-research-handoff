# Exp145/148 재개 사전등록: 가상 Wafer-Run TRAIN-only 경보 정책

작성일: 2026-07-13
대상 저장소: `pmh10401/timeseries-anomaly-research-handoff`
기준 실험: Exp137, B2, Exp143, Exp144 C0, Exp147 D1a

## 1. 문서 목적

실제 반도체 설비 데이터는 보안과 접근 제한으로 확보할 수 없으므로, 본 프로젝트는 공개 시계열 데이터를 통합해 **가상의 반도체 설비 이상 탐지 벤치마크**를 구축했다.

기존 Exp145와 Exp148은 TRAIN instance가 실제 wafer run인지 DB에서 검증되지 않았다는 이유로 차단되었다. 이 문서는 해당 연결을 실제 설비 사실로 주장하지 않고, 다음의 **명시적 벤치마크 모델링 가정**으로 등록해 TRAIN-only 정책 검증을 다시 진행하기 위한 사전등록이다.

이 문서는 실제 설비 배포 정책을 확정하지 않는다. 모든 결과는 이미 관찰된 TEST 데이터에 대한 retrospective counterfactual이며, prospective equipment validation이 아니다.

## 2. 가상 반도체 설비 모델링 가정

본 연구에서는 다음과 같이 해석한다.

```text
하나의 dataset
= 하나의 고정된 가상 equipment / recipe / sensor / step 탐지 환경

하나의 TRAIN 또는 TEST instance
= 해당 가상 환경에서 수행된 하나의 독립적인 virtual wafer run trace

candidate index
= TEST split 안에서 이상 후보로 선택된 virtual wafer run instance 번호
```

따라서 candidate index는 시계열 내부의 시간 위치가 아니다.

이 가정은 실제 반도체 설비 메타데이터로 검증된 사실이 아니라 연구용 abstraction이다. 결과 문서와 발표에서는 반드시 `virtual`, `simulated benchmark`, `가상 설비` 중 하나를 명시한다.

## 3. 기존 차단 상태의 재해석

### 계속 유효한 차단

- 현재 DB로 실제 설비의 run당 사용자 처리 가능 경보 수를 결정할 수 없다.
- 실제 equipment, recipe, sensor, step, run ID와 timestamp가 없다.
- 실제 생산 설비의 maximum K 또는 review capacity를 정당화할 수 없다.
- 실제 운영 성능이나 prospective 신뢰도를 주장할 수 없다.

### 가상 벤치마크 내부에서 해제 가능한 차단

- 각 instance를 virtual wafer run으로 간주한 TRAIN-only calibration 연구
- TEST 길이를 사용하지 않는 instance-level 경보 정책
- TRAIN 정상 instance에서 계산한 비모수적 nonconformity calibration
- 가상 run 100개당 경보·FP·review workload 비교

## 4. 중요한 정책 결정: batch top-N budget을 운영 정책으로 사용하지 않음

각 instance를 하나의 virtual wafer run으로 해석하면, 실제 배포에 가까운 질문은 다음이다.

> 새로운 virtual wafer run 하나가 들어왔을 때, 과거 TRAIN 정상 run과 비교하여 독립적으로 경보할 것인가?

따라서 전체 TEST split의 크기에 따라 후보 수를 정하는 기존 top-N budget을 새 정책으로 재도입하지 않는다.

Exp145/148 재개 실험은 **TEST batch 후보 수 budget**이 아니라 **TRAIN 정상 run 기반 instance-level false-alarm calibration**을 검증한다.

기존 blocked Exp145/148 산출물은 덮어쓰지 않는다. 의미가 달라졌으므로 신규 실험 ID를 사용한다.

- `experiment_149_virtual_run_conformal_policy_current_source`
- `experiment_150_virtual_run_conformal_policy_b2_source`

## 5. 연구 질문

1. TEST 길이와 top-N cap을 사용하지 않고도 TRAIN 정상 run만으로 자동 경보 FP를 제어할 수 있는가?
2. Current source와 B2 family-independent source에서 같은 TRAIN-only calibration이 어떻게 작동하는가?
3. Hard alert precision, recall, Mean Hard F1과 가상 run당 workload 사이의 trade-off는 무엇인가?
4. TRAIN 정상 instance 수가 적은 데이터셋에서 conformal p-value 해상도 한계가 얼마나 큰가?
5. Family-independent source와 instance-level TRAIN-only calibration을 결합해도 기존 Exp137의 false-alarm 감소 효과가 유지되는가?

## 6. 실행 전 데이터 사전 점검

1117개 평가 데이터셋 각각에 대해 다음을 기록한다.

```text
dataset
train_instance_count
test_instance_count
train_label_values
test_label_values
values_dtype
labels_blob_presence
instance_level_binary_label_compatible
minimum_attainable_conformal_p
```

다음 조건을 확인한다.

1. TEST `label`이 instance-level `0/1`로 해석 가능하다.
2. `candidate index`가 TEST instance index다.
3. `values_blob`은 `float32`로 복원 가능하다.
4. point-anomaly 형식(`label='anomaly_detection'`, 시점별 `labels_blob`)은 instance-level 실험에 조용히 섞지 않는다.
5. point-anomaly 형식이 평가 1117개에 포함되면 별도 상태로 기록하고 해당 데이터셋을 임의 변환하지 않는다.
6. 데이터셋을 제외할 경우 제외 목록과 이유를 prediction 전에 고정한다.

사전 점검 실패 시 `BLOCKER_REPORT_VIRTUAL_RUN.md`를 작성하고 후속 평가를 중단한다.

## 7. TRAIN-only conformal calibration

### 7.1 Prediction과 evaluation 분리

Prediction 함수는 다음 입력만 받는다.

```python
prediction = predict_virtual_runs(
    x_train_normal,
    x_test,
    frozen_feature_config,
    alpha,
)
```

Prediction 함수는 다음을 받지 않는다.

- `y_test`
- TEST 실제 이상 위치
- TEST anomaly count
- dataset family에 따른 정책 분기
- dataset 이름에 따른 정책 분기
- `len(x_test)`를 이용한 threshold 또는 budget 결정

평가는 prediction 결과가 파일과 hash로 동결된 뒤 별도 함수에서 수행한다.

```python
metrics = evaluate_virtual_runs(prediction, y_test)
```

### 7.2 Source-level nonconformity score

기존 source 구조를 유지한다.

- ROCKET Exp40 source
- Exp55 source
- Exp56 source
- Exp84 source

각 source마다 TRAIN 정상 instance의 out-of-fold nonconformity score를 만든다.

### 7.3 Cross-fitting 규칙

- `n_train >= 20`: deterministic 5-fold cross-fitting, `random_state=20260717`
- `5 <= n_train < 20`: leave-one-out cross-fitting
- `n_train < 5`: autonomous Hard alert calibration을 지원하지 않음. 데이터셋 행은 유지하고 `low_train_review_only=1`로 기록하며 자동 경보 후보를 생성하지 않는다.

데이터셋을 결과에서 삭제하지 않는다.

### 7.4 Conformal p-value

TRAIN 정상 out-of-fold score를 `a_1, ..., a_n`, TEST run score를 `s`라 할 때 다음을 사용한다.

```text
p(s) = (1 + count(a_i >= s)) / (n + 1)
```

동점 처리는 위 식으로 고정한다. TEST 결과를 보고 randomized p-value나 smoothing을 추가하지 않는다.

각 데이터셋에 다음을 기록한다.

```text
n_train
crossfit_method
minimum_attainable_p
resolution_limited_at_alpha
```

## 8. Alpha 사전등록

실제 설비의 허용 false-alarm rate가 없으므로 하나의 운영 alpha를 선택하지 않는다.

다음 alpha를 **민감도 분석점**으로 사전 고정한다.

```text
alpha in {0.005, 0.01, 0.02, 0.05}
```

규칙:

- TEST 성능이 가장 좋은 alpha를 선택하지 않는다.
- `best alpha`, `optimal alpha`, `recommended alpha`라는 표현을 사용하지 않는다.
- 모든 alpha 결과를 함께 보고한다.
- alpha별로 conformal 해상도 때문에 경보가 원천적으로 불가능한 데이터셋 수를 표시한다.

## 9. 후보와 lane 구성

각 alpha에서 source-level p-value를 계산한다.

### 1차 탐지 후보

다음 네 source 중 최소 2개에서 `p <= alpha`인 TEST instance를 후보로 한다.

```text
ROCKET Exp40
Exp55
Exp56
Exp84
```

### 자동 경보

1차 탐지 후보 중 기존 독립 교차 확인이 같은 TEST instance를 지목한 경우 Hard alert로 둔다.

### 일반 검토

1차 탐지 후보이지만 자동 경보 교차 근거가 부족한 경우 Standard review로 둔다.

### 우선 검토

Priority review는 기존 retrospective research rule을 유지하며 review-only다. 자동 경보로 승격하지 않는다.

### 알림 없음

어느 lane에도 포함되지 않은 TEST instance다.

모든 lane은 상호 배타적이어야 한다.

## 10. 실험 구성

### Exp149: Current source + virtual-run conformal policy

- 기존 current source configuration 유지
- family-dependent historical source coverage가 남아 있다는 caveat 유지
- TEST-length gate와 top-N budget 사용 금지
- alpha 4개를 모두 실행

목적:

> 기존 source 조건에서 instance-level TRAIN-only calibration의 workload와 autonomous 성능을 확인한다.

### Exp150: B2 source + virtual-run conformal policy

- B2의 full-coverage family-independent Exp84 source 사용
- family 또는 dataset 이름으로 source coverage를 바꾸지 않음
- TEST-length gate와 top-N budget 사용 금지
- alpha 4개를 모두 실행

목적:

> Family-independent source와 instance-level TRAIN-only calibration을 결합한 policy-level 결과를 확인한다.

Exp150은 end-to-end strict TRAIN-only가 아니다. Exp84 feature configuration 선택 provenance는 별도 D2 검증 대상이다.

## 11. 비교 기준

다음 결과를 나란히 보고한다.

```text
Exp137 current retrospective policy
B2 full-coverage family-neutral
Exp144 C0 current source no-budget
Exp147 D1a B2 source no-budget
Exp149 alpha sensitivity
Exp150 alpha sensitivity
```

이 비교는 정책 선택이 아니라 구조적 진단이다.

## 12. 지표

### Autonomous Hard alert

- Hard alert count
- Hard TP
- Hard FP
- micro Hard precision
- dataset-level mean Hard recall
- Mean Hard F1
- Hard FP per 100 virtual TEST runs
- Hard alerts per 100 virtual TEST runs
- 실제 이상 virtual run 보존율

### Human-assisted review

- Standard review count, TP, FP, precision
- Priority review count, TP, FP, precision
- Review requests per 100 virtual TEST runs
- `mean_combined_f1`은 human-assisted diagnostic으로만 별도 표시

### Calibration 및 coverage

- alpha별 resolution-limited dataset 수
- `n_train < 5` review-only dataset 수
- dataset별 minimum attainable p-value
- candidate count median, p90, p95, max
- candidate가 0개인 dataset 수
- exact-index lane 비교

Hard와 review 지표를 합쳐 autonomous 성능으로 발표하지 않는다.

## 13. 불변성 테스트

다음 테스트를 작성한다.

`./scratch/test_exp149_virtual_run_conformal_policy.py`

필수 테스트:

1. `y_test`를 변경해도 prediction이 동일함
2. TEST anomaly 위치를 변경해도 prediction이 동일함
3. TEST instance 수를 바꿔도 기존 instance의 p-value, threshold parameter, alpha가 동일함
4. B2 variant에서 family 이름 변경 후 prediction이 동일함
5. dataset 이름 변경 후 정책 결과가 동일함
6. 입력 행 순서를 바꿔도 dataset별 결과가 동일함
7. 동일 seed에서 out-of-fold TRAIN score hash가 동일함
8. `n_train < 5` 데이터셋은 Hard alert가 없고 review-only 상태가 기록됨
9. alpha 목록이 코드에 고정되어 있고 TEST metric으로 변경되지 않음
10. Prediction 함수가 `y_test`를 인자로 받지 않음

## 14. 산출물

코드:

```text
run_exp149_virtual_run_conformal_policy.py
scratch/test_exp149_virtual_run_conformal_policy.py
```

결과 경로:

```text
outputs/exp137_policy_train_only_validation/virtual_run_conformal/
```

필수 결과:

```text
00_preregistration.md
01_evaluation_contract.json
02_virtual_run_grain_audit.csv
03_exp149_current_source_results.csv
04_exp149_current_source_summary.csv
05_exp149_current_source_summary.md
06_exp150_b2_source_results.csv
07_exp150_b2_source_summary.csv
08_exp150_b2_source_summary.md
09_alpha_sensitivity_comparison.csv
10_dataset_level_candidate_diff.csv
11_invariance_test_log.txt
12_invariance_test_report.md
13_presentation_claims.md
MANIFEST.csv
MANIFEST.md
```

모든 결과에는 다음을 기록한다.

- Git commit
- code SHA256
- DB SHA256
- input result SHA256
- alpha 목록
- seed
- crossfit method
- dataset coverage
- error dataset 목록
- 생성 명령

## 15. 실행 순서

1. 이 문서를 evaluation contract에 복사하고 hash를 기록한다.
2. DB와 evaluation dataset 1117개 목록의 hash를 고정한다.
3. instance-level binary label compatibility를 점검한다.
4. Prediction/evaluation 분리 테스트를 먼저 작성한다.
5. 작은 deterministic fixture에서 conformal p-value 테스트를 통과시킨다.
6. 10개 dataset smoke test를 수행한다.
7. TEST label을 보지 않은 상태에서 1117개 prediction 결과를 생성하고 hash를 고정한다.
8. Prediction coverage와 오류가 0인지 확인한다.
9. 동결된 prediction에 TEST label을 결합해 사후 평가한다.
10. 모든 alpha 결과를 함께 보고한다.
11. 결과를 보고 alpha, feature, source, threshold를 변경하지 않는다.
12. 추가 개선은 새 실험 ID와 새 사전등록 문서로 진행한다.

## 16. 중단 조건

다음 조건에서는 실행을 중단한다.

- 평가 대상에 instance-level binary label로 해석할 수 없는 데이터가 섞여 있음
- Point-anomaly 데이터가 조용히 instance classification으로 변환됨
- Prediction 단계에서 TEST label을 읽음
- TEST instance 수가 threshold 또는 alpha를 바꿈
- B2 source coverage가 1117개에서 재현되지 않음
- Out-of-fold score가 동일 seed에서 재현되지 않음
- 오류 dataset을 임의로 제외해야 결과가 완성됨
- TEST 결과를 보고 alpha 또는 low-train 규칙을 변경하려는 경우

중단 시 `BLOCKER_REPORT_VIRTUAL_RUN.md`에 원인, 영향 dataset, 재개 조건을 기록한다.

## 17. 결과 해석 규칙

### 사용할 수 있는 표현

- `가상 wafer-run 가정에 기반한 TRAIN-only calibration`
- `공개 시계열 벤치마크에서의 retrospective counterfactual`
- `가상 run 100개당 false alert`
- `현재 고정 feature/source configuration에서의 policy-level 결과`

### 사용할 수 없는 표현

- `실제 반도체 설비에서 검증됨`
- `실제 wafer run당 false alarm 보장`
- `실제 사용자의 최적 경보 수가 결정됨`
- `전체 파이프라인이 end-to-end strict TRAIN-only임`
- `가장 좋은 alpha가 운영 정책으로 선택됨`

## 18. 완료 기준

다음을 모두 만족해야 완료다.

- 가상 dataset/instance 정의가 evaluation contract에 기록됨
- 1117개 데이터셋 grain audit 완료
- Prediction과 evaluation 구조 분리
- Exp149와 Exp150의 alpha 4개 결과 생성
- 모든 alpha 결과 동시 보고
- `n_train < 5`와 conformal resolution 한계 별도 보고
- Hard autonomous와 review human-assisted 분리
- 불변성 테스트 통과
- 모든 input/code/output hash 기록
- 실제 설비 검증이 아니라는 caveat 표시

이 문서의 승인으로 Exp145/148의 기존 blocked 산출물을 덮어쓰는 것은 허용되지 않는다. 신규 Exp149/150으로만 실행한다.
