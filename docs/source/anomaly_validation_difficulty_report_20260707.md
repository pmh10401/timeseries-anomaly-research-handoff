# Anomaly Validation Difficulty Report - 2026-07-07

현재까지 완료된 신뢰 가능 실험을 기준으로, 이상치 검증이 어려운 데이터셋을 조사했다.

## 사용한 근거

- Original repeated-normal 기준: 실험 32, 35 original, 36 original, 38 original
- Clean-balanced 기준: 실험 33 balanced, 34 balanced, 36 balanced, 39 balanced
- 데이터 구조 기준: DB metadata, train/test normal overlap profile, actual length profile, clean-balanced manifest
- 제외: 현재 실행 중인 실험 39 original partial 결과와 아직 current 결과가 없는 실험 40 original

산출물:

- `/Users/minho/Documents/Dataset/dataset_anomaly_validation_difficulty_20260707.csv`
- `/Users/minho/Documents/Dataset/dataset_anomaly_validation_difficulty_summary_20260707.csv`

## 분류 기준

- `model_hard`: 여러 완료 실험을 통틀어도 score 분리력이 낮거나, best F1/oracle F1/AUC-PR가 낮은 데이터셋
- `evaluation_hard`: 모델은 맞히는 경우도 있지만 test anomaly 1개, 작은 test set, train/test 정상 중복, clean-balanced 부적격, variable length 등으로 검증 신뢰도가 낮은 데이터셋
- `normal`: 현재 기준에서 모델 성능과 평가 구조 모두 큰 경고가 없는 데이터셋

## 전체 결과

| 유형 | 개수 | 의미 |
| --- | ---: | --- |
| `model_hard` | 218 | 모델/score 자체가 이상치를 잘 올리지 못하는 후보 |
| `evaluation_hard` | 789 | F1이나 검증 구조가 불안정한 후보 |
| `normal` | 110 | 현재 기준에서 비교적 검증 신뢰도가 높은 후보 |

주요 리스크 이유:

| 이유 | 개수 | 해석 |
| --- | ---: | --- |
| clean-balanced 부적격 | 945 | train/test overlap 제거 후 2% clean 평가셋을 만들기 어려움 |
| single anomaly F1 불안정 | 788 | test anomaly가 1개라 한 번 맞히면 F1 1, 놓치면 0에 가까움 |
| small test set | 788 | test size가 50 이하라 지표 분산이 큼 |
| test normal duplicate 많음 | 499 | 정상 반복이 많아 원본 benchmark가 쉬워질 수 있음 |
| very small train | 478 | 정상 학습 샘플이 20개 미만 |
| high train/test normal overlap | 460 | 정상 train과 test가 90% 이상 겹침 |
| variable length/metadata mismatch | 153 | DB metadata length와 실제 blob 길이 기준이 흔들림 |
| zero best original F1 | 158 | 완료된 original 후보 중 best도 F1 0 |

## 가장 어려운 데이터셋

original과 clean-balanced 양쪽에서 모두 어려운 핵심 후보:

| dataset | family | original best F1 | original oracle | clean best F1 | clean oracle |
| --- | --- | ---: | ---: | ---: | ---: |
| `DistalPhalanxOutlineCorrect_normal_0` | DistalPhalanxOutlineCorrect | 0.0000 | 0.1348 | 0.0000 | 0.1250 |
| `Earthquakes_normal_0` | Earthquakes | 0.0000 | 0.2381 | 0.0000 | 0.1538 |
| `EthanolLevel_normal_2` | EthanolLevel | 0.0000 | 0.1739 | 0.0000 | 0.1667 |
| `EthanolLevel_normal_3` | EthanolLevel | 0.0000 | 0.2500 | 0.0000 | 0.0952 |
| `HandOutlines_normal_0` | HandOutlines | 0.0000 | 0.2222 | 0.0000 | 0.0833 |
| `MelbournePedestrian_normal_3` | MelbournePedestrian | 0.0000 | 0.2222 | 0.0000 | 0.1333 |
| `MiddlePhalanxOutlineCorrect_normal_0` | MiddlePhalanxOutlineCorrect | 0.0000 | 0.2000 | 0.0000 | 0.0667 |
| `PhalangesOutlinesCorrect_normal_0` | PhalangesOutlinesCorrect | 0.0000 | 0.1651 | 0.0000 | 0.1250 |
| `ScreenType_normal_1` | ScreenType | 0.0000 | 0.3125 | 0.0000 | 0.1333 |
| `SmallKitchenAppliances_normal_2` | SmallKitchenAppliances | 0.0000 | 0.1515 | 0.0000 | 0.0870 |

이 그룹은 단순 threshold 문제가 아니라, 현재 score 공간에서 이상 샘플이 정상보다 앞에 오지 않는 문제로 보는 것이 맞다.

## Family 단위 Trouble Spots

데이터셋이 3개 이상인 family 중 model-hard 비율이 높은 그룹:

| family | model-hard 비율 | hard 수 / 전체 | 평균 difficulty score |
| --- | ---: | ---: | ---: |
| ScreenType | 1.00 | 3 / 3 | 12.00 |
| EthanolLevel | 1.00 | 4 / 4 | 10.00 |
| LargeKitchenAppliances | 1.00 | 3 / 3 | 8.00 |
| ArrowHead | 0.67 | 2 / 3 | 7.33 |
| RefrigerationDevices | 0.67 | 2 / 3 | 5.33 |
| ProximalPhalanxOutlineAgeGroup | 0.67 | 2 / 3 | 5.00 |
| Worms | 0.60 | 3 / 5 | 5.60 |
| Haptics | 0.60 | 3 / 5 | 5.40 |
| ECG5000 | 0.60 | 3 / 5 | 4.60 |
| Fish | 0.57 | 4 / 7 | 4.86 |

## 해석

현재 가장 큰 문제는 두 층으로 나뉜다.

첫째, 실제 모델 개선이 필요한 데이터셋이 있다. 위의 `model_hard` 218개, 특히 original과 clean-balanced 양쪽에서 모두 실패하는 32개는 score 공간 자체를 바꿔야 한다. threshold를 더 만지는 것으로 해결될 가능성은 낮다.

둘째, 검증 자체가 불안정한 데이터셋이 매우 많다. test anomaly가 1개인 데이터셋 788개는 F1이 0 또는 1 쪽으로 크게 튀기 쉽고, clean-balanced 평가셋을 구성할 수 없는 데이터셋이 945개다. 따라서 평균 F1 하나로 전체 프로젝트의 성능을 판단하면 위험하다.

## 권장 방향

1. 최종 성능표에는 `original repeated-normal`, `clean-balanced eligible`, `model-hard subset`, `evaluation-hard subset`을 분리해서 표시한다.
2. score 개선 실험은 먼저 original+clean 양쪽에서 모두 어려운 32개 핵심 후보에 집중한다.
3. `ScreenType`, `EthanolLevel`, `LargeKitchenAppliances`, `RefrigerationDevices`, `Phalanges/Outline` 계열은 family-specific error analysis를 진행한다.
4. test anomaly 1개짜리 데이터셋은 F1보다 rank position, AUC-PR, oracle F1, top-k hit 여부를 함께 본다.
5. clean-balanced 부적격 데이터셋은 실무 운영 검증 후보에서 별도 표시하고, 데이터셋 재구성 또는 외부 raw source 기반 재분할을 장기 과제로 둔다.
