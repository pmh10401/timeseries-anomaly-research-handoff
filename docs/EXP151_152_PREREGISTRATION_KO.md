# Exp151/152 가상 Wafer-Run Conformal Policy v2 사전등록

## 실험 ID

- `experiment_151_virtual_run_conformal_policy_v2`
- `experiment_152_virtual_run_conformal_policy_v2`

## 목적

Exp149/150의 잘못된 routing map을 수정하고, 새 conformal 후보에 독립 Block-B/Block-C evidence를 다시 적용하여 autonomous Hard alert와 human-assisted review를 정확히 평가한다.

## 고정 조건

- DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- 평가 dataset: Exp137과 동일한 1,117개
- seed: `20260717`
- alpha: `0.005, 0.01, 0.02, 0.05`
- 후보: available source 중 최소 2개에서 `p <= alpha`
- Block-B: neighbors 3, TRAIN count-cap rate 1.5%
- Block-C: TRAIN count-cap rate 1.0%
- `n_train < 5`: calibration abstention, 후보/Hard/review index 없음
- TEST 길이에 따른 top-N budget 없음
- TEST label과 이상 위치는 prediction 이후 evaluation에만 사용

## Lane

```text
Conformal 후보 ∩ 새로 계산한 Block-B
→ Hard alert (autonomous)

Conformal 후보 - Hard
  ∩ 새로 계산한 Block-C
→ Priority review (human-assisted, retrospective research rule)

남은 Conformal 후보
→ Standard review (human-assisted)

후보가 아닌 TEST instance
→ No alert
```

## Exp151과 Exp152 차이

### Exp151

과거 current source coverage를 유지한다. Exp84는 기존 `family_guard_v1` source row가 존재했던 dataset에만 available source로 포함한다. 이 coverage는 historical family dependency caveat를 유지한다.

### Exp152

B2 manifest의 고정 configuration·seed·1,117개 coverage를 검증한 뒤 모든 dataset에 Exp84 source를 포함한다. Family 이름과 TEST 길이는 source coverage를 바꾸지 않는다.

## 평가 원칙

- Prediction JSONL을 label-free로 생성하고 SHA-256을 동결한다.
- Evaluation은 동결 hash가 일치할 때만 TEST label을 로딩한다.
- Hard TP/FP/precision/recall/Mean Hard F1만 autonomous 지표다.
- Standard/Priority review와 combined F1은 human-assisted 진단이다.
- 모든 결과는 retrospective counterfactual이며 실제 설비 prospective validation이 아니다.
- TEST 성능으로 alpha, source, threshold, configuration을 변경하거나 선택하지 않는다.
