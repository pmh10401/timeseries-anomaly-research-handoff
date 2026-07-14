# Exp149/150 → Exp151/152 수정 전·수정 후 비교

## 요약

Exp149/150은 TRAIN 정상 데이터로 conformal 후보를 만들었지만, 최종 Hard/Priority 라우팅에서 잘못된 row map을 읽어 모든 후보가 Standard review로 이동했다. 또한 prediction row 안에서 TEST label 평가까지 동시에 수행해, 사전등록에서 요구한 prediction/evaluation 구조 분리가 완성되지 않았다.

수정본은 기존 결과를 덮어쓰지 않고 Exp151/152로 재실행한다.

---

## 1. 잘못된 Exp133/Exp135 map 로딩

### 수정 전

```python
def load_routing_maps():
    row133, row135 = exp133.load_maps()
    return {
        name: {
            "hard": parse_indices(row133[name].get("high_confidence_indices")),
            "priority": parse_indices(row135[name].get("review_candidate_indices")),
        }
        for name in row133
    }
```

### 문제

`exp133.load_maps()`는 Exp133/Exp135 결과를 반환하지 않는다. 실제 반환값은 Exp119a/Exp93 후보 row와 Exp131 Block-B row다. 이 row에는 `high_confidence_indices`, `review_candidate_indices`가 없어 `.get()`이 빈 값을 반환했고 Hard/Priority evidence가 항상 빈 집합이 되었다.

### 수정 후

```python
def recompute_independent_evidence(x_train, x_test):
    # Exp131과 동일한 Block-B score/threshold를 새 후보에 대해 다시 계산
    ...
    # Exp135와 동일한 Block-C score/threshold를 새 Standard 후보에 적용
    ...
    return {
        "block_b_indices": block_b_indices,
        "block_c_indices": block_c_indices,
        ...
    }
```

### 이유

새 conformal 후보는 기존 Exp93 후보와 후보 universe가 다르다. 따라서 기존 최종 index를 가져오는 것이 아니라, 독립 score와 TRAIN-normal threshold를 새 후보에 다시 적용해야 한다.

---

## 2. 오래된 후보 universe의 index 재사용

### 수정 전

```python
hard = candidates & old_high_confidence_indices
priority = old_priority_indices & standard
```

### 문제

기존 `high_confidence_indices`와 Priority 후보는 Exp93/Exp134 후보를 대상으로 계산된 결과다. 새 conformal 후보에 그대로 교집합하면, 새 후보가 실제 독립 확인을 받지 못한 채 과거 후보 위치와 우연히 겹치는지만 검사하게 된다.

### 수정 후

```python
lanes = route_lanes_from_recomputed_evidence(
    candidates=conformal_candidates,
    block_b_indices=recomputed_block_b_indices,
    block_c_indices=recomputed_block_c_indices,
    test_size=n_test,
)
```

```python
hard = candidates & block_b_indices
standard_before_priority = candidates - hard
priority_review = standard_before_priority & block_c_indices
standard_review = standard_before_priority - priority_review
```

### 이유

- Block-B: autonomous Hard alert의 독립 교차 확인
- Block-C: Hard가 되지 않은 후보의 보조 확인이며 review-only

모든 lane은 동일한 새 candidate universe 안에서 계산된다.

---

## 3. Prediction과 evaluation 혼합

### 수정 전

```python
def run_one(...):
    x_train, x_test, y_test = load_dataset_data(name)
    record["y_test"] = y_test
    return make_prediction_record(..., record, ...)
```

```python
metrics.update(tier_metrics(record["y_test"], indices))
```

### 문제

후보 생성이 label을 직접 분기 조건으로 사용한 것은 아니지만, prediction 함수와 checkpoint 안에 TEST label 기반 지표가 함께 들어갔다. 따라서 “label을 차단한 prediction을 먼저 동결했다”는 구조적 증거가 부족했다.

### 수정 후

```python
# Phase 1: label-free prediction
python3 ... --phase predict
```

```python
def predict_dataset(dataset_name, current_exp84_coverage, b2_contract_names):
    x_train, x_test, record = load_dataset_series_only(dataset_name)
    # TEST label을 읽지 않음
    ...
```

```python
# Phase 2: frozen hash 검증 후 evaluation
python3 ... --phase evaluate
```

```python
def run_evaluation_phase(...):
    verify_frozen_prediction(prediction_path, expected_hash)
    y_test = load_test_labels(dataset_name)
    ...
```

### 이유

Prediction 파일을 먼저 SHA-256으로 동결하고, evaluation이 그 hash를 재확인한 뒤에만 TEST label을 읽도록 분리한다.

---

## 4. 일반 loader가 TEST label을 함께 로딩

### 수정 전

```python
x_train, x_test, y_test = load_dataset_data(dataset_name)
record = load_original_record(dataset_name, DB_PATH)
```

### 문제

두 helper 모두 TEST label을 로딩한다. Prediction 경로에서 label을 실제로 사용하지 않더라도, 구조적으로 접근 가능했다.

### 수정 후

```python
def load_dataset_series_only(dataset_name, db_path=DB_PATH):
    SELECT values_blob ... split='TRAIN'
    SELECT values_blob ... split='TEST'
    # label 컬럼을 SELECT하지 않음
```

```python
def load_test_labels(dataset_name, db_path=DB_PATH):
    # evaluation phase 전용
    SELECT label ... split='TEST'
```

### 이유

Label 접근 가능성 자체를 phase 경계로 차단한다.

---

## 5. Target length에 TEST 구조가 들어갈 가능성

### 수정 전

```python
target_len = target_len_for_record(record, "actual_median")
```

`actual_median`은 과거 record에서 TRAIN과 TEST series length를 합쳐 계산될 수 있었다.

### 수정 후

```python
train_target_len = median(TRAIN series lengths)
x_train = align_series_lengths(train_series, train_target_len)
x_test = align_series_lengths(test_series, train_target_len)
```

### 이유

Feature preprocessing parameter도 TRAIN 정상 데이터로 고정한다. TEST 값은 고정 transform을 적용받는 입력으로만 사용한다.

---

## 6. `n_train < 5`의 의미 불일치

### 수정 전

문서에는 `review-only`라고 적었지만 실제 코드는 후보를 빈 집합으로 만들어 모든 instance를 No alert로 보냈다.

```python
if len(x_train) < 5:
    candidates = set()
```

### 수정 후

```python
{
    "calibration_status": "insufficient_calibration_n_train_lt5",
    "abstained": 1,
    "candidate_count": 0,
    "hard_count": 0,
    "standard_review_count": 0,
    "priority_review_count": 0,
}
```

### 이유

후보를 만들 근거가 없으므로 정확한 의미는 `review-only`가 아니라 dataset-level calibration abstention이다. Review 후보가 있는 것처럼 표현하지 않는다.

---

## 7. Conformal resolution 집계

### 수정 전

```python
resolution_limited = any(alpha < minimum_p[source] for source in sources)
```

### 문제

정책은 2개 source 동의를 요구한다. 4개 중 1개 source만 alpha에 도달하지 못해도 전체 dataset을 제한으로 표시하는 것은 지나치게 엄격했다.

### 수정 후

```python
eligible_source_count = sum(minimum_p[source] <= alpha for source in sources)
resolution_blocks_two_source_agreement = eligible_source_count < 2
```

### 이유

실제 후보 생성 조건과 같은 단위로 해상도 제한을 측정한다.

---

## 8. B2 manifest가 실제 계약 검증에 사용되지 않음

### 수정 전

B2 manifest를 읽었지만 Exp150 prediction 계산에서 row 내용과 coverage를 assert하지 않았다.

### 수정 후

```python
validate_b2_manifest(
    expected_datasets=all_1117_names,
    config_name="aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3",
    random_state=20260717,
    source_uses_family_name=0,
    source_uses_test_length=0,
)
```

### 이유

Exp152가 B2와 같은 family-independent source contract를 사용한다는 것을 실행 전에 검증한다. B2의 TEST 결과 index는 재사용하지 않는다.

---

## 9. 기존 결과 덮어쓰기 방지

### 수정 전

Exp149/150 output directory를 재사용할 가능성이 있었다.

### 수정 후

```text
experiment_151_virtual_run_conformal_policy_v2
experiment_152_virtual_run_conformal_policy_v2

outputs/exp151_152_virtual_run_conformal_v2/
```

### 이유

이미 TEST 결과를 확인한 Exp149/150을 수정하여 덮어쓰면 연구 이력이 사라진다. 새 번호로 결과를 보존한다.

---

## 10. Alpha 선택 원칙

### 수정 전·후 공통 유지

```python
ALPHAS = (0.005, 0.01, 0.02, 0.05)
```

### 이유

모든 alpha는 사전등록된 sensitivity point다. TEST precision, FP, F1을 보고 하나를 `best`, `optimal`, `recommended`로 선택하지 않는다.
