# Experiment Consistency Audit - 2026-07-07

이 문서는 실험 25를 검토한 방식으로 주요 완료/진행 실험의 정합성, 비교 가능성, 재실험 필요성을 빠르게 점검한 결과이다.

## 핵심 결론

- 실험 35/36/37/38 original은 detail coverage와 summary가 맞으므로 재실험이 필요하지 않다.
- 실험 39 original은 현재 실행 중인 partial 결과이므로 완료 전 성적을 인용하지 않는다.
- 실험 40 original은 current results/summary 파일이 없으므로 큐에서 재실행 대상이다.
- 실험 25는 성능이 높지만 `prior_q_02`가 평가셋의 2% 이상치 비율을 사실상 알고 쓰는 방식이므로 운영 성능으로 직접 채택하지 않는다.
- 실험 25 summary는 `config_name`을 버리고 strategy만 pooling하여 `num_datasets=5585`로 표시되던 문제가 있었다. 2026-07-07 복구에서 `config_name + strategy` 단위로 재생성했다.
- clean-balanced 실험의 기준 universe는 1,117개가 아니라 `balanced_2pct` eligible 174개 중 Cornell 제외 173개이다.
- clean-balanced 34/37/38은 정합성 정상이다.
- clean-balanced 35/36은 archive 완전본을 current dashboard 파일로 복구했다.
- clean-balanced 39는 detail 169개 dataset, summary 173개 dataset으로 불일치하던 문제를 누락 4개 dataset 재계산으로 복구했다.

## 판정표

| 구분 | 실험 | 상태 | 판정 |
| --- | --- | --- | --- |
| Rank/VAE | 24 `rank_v1_train_evt` | 1,117 datasets x 5 configs 정상. 단 train-EVT F1이 낮음 | 재실행보다 threshold 재설계가 우선 |
| Rank/VAE | 25 `rank_threshold_calibration` | detail 1,117 datasets x 5 configs x 9 strategies. summary 45행, `config_name + strategy` 단위 | 복구 완료. 운영 성능이 아니라 prevalence-aware 진단용 |
| ROCKET | 26 | 1,117 datasets x 9 strategies 정상 | 재실행 불필요, top-k 진단용 |
| KNN/ROCKET | 29 | 1,117 x 13 configs x 5 methods 정상 | 재실행 불필요 |
| KNN/ROCKET | 30 | 1,117 x 13 methods 정상 | 재실행 불필요 |
| KNN/ROCKET | 31 | 1,117 x 10 methods 정상 | 재실행 불필요 |
| KNN/ROCKET | 32 | 1,117 x 21 config/method 정상 | 현재 운영 후보 기준 |
| Balanced | 34 | 173 x 15 정상 | 재실행 불필요 |
| Balanced | 35 | 173 x 8 정상 | archive에서 current dashboard 파일 복구 완료 |
| Balanced | 36 | 173 x 12 정상 | archive에서 current dashboard 파일 복구 완료 |
| Balanced | 37 | 173 x 9 정상 | 재실행 불필요 |
| Balanced | 38 | 173 x 12 정상 | 재실행 불필요 |
| Balanced | 39 | 173 x 20 정상 | 누락 4개 dataset 재계산 및 summary 재생성 완료 |
| Original | 35 | 1,117 x 8 정상 | 재실행 불필요 |
| Original | 36 | 1,117 x 12 정상 | 재실행 불필요 |
| Original | 37 | 1,117 x 9 정상 | 재실행 불필요 |
| Original | 38 | 1,117 x 12 정상 | 재실행 불필요 |
| Original | 39 | 실행 중 partial | 완료 후 summary 재검증 |
| Original | 40 | current 파일 없음 | 큐 재실행 필요 |

## 운영 해석 주의

`prior_q`, `top_1`, fixed empirical q 계열은 이상치 비율 또는 최소 이상치 존재를 강하게 가정한다. 현재 원본 평가셋은 이상치가 대략 2%로 구성되어 있어 `prior_q_02`가 특히 유리하다. 따라서 이 계열은 score ranking 진단에는 유용하지만, 정상만 들어오는 실무 상황의 자동 threshold 성능으로 직접 비교하면 안 된다.

운영 후보 비교는 train 정상 분포 기반이면서 false alarm budget을 관리하는 `count_cap_2pct`/`count_cap_3pct` 계열을 우선한다. 현재 가장 신뢰 가능한 원본 기준 후보는 실험 32의 `rocket_256_knn3 + count_cap_2pct/3pct`이다.

## 다음 조치

1. 실험 39 original 완료를 기다린 뒤 detail dataset 수 1,117, summary `num_datasets=1117`, rows-per-dataset 20을 확인한다.
2. 실험 40 original은 current 파일이 없으므로 현재 큐에서 실행되도록 유지한다.
3. 실험 25는 복구 완료했다. 남은 과제는 `prior_q_02`를 운영 지표가 아닌 진단 지표로 표시하는 것이다.
4. clean-balanced 35/36/39는 복구 완료했다. 남은 과제는 original benchmark와 clean-balanced 하한선을 대시보드에서 명확히 분리해 보여주는 것이다.
