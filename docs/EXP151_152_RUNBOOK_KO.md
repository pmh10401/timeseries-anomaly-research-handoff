# Exp151/152 실행 가이드

## 1. 파일 배치

압축을 저장소 루트에서 풀어 다음 파일이 존재하도록 한다.

```text
experiments/exp151_152/__init__.py
experiments/exp151_152/virtual_run_policy_core.py
experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py
experiments/exp151_152/test_exp151_152_virtual_run_conformal_policy_v2.py
```

DB 경로는 수정하지 않는다.

```text
/Users/minho/Documents/Dataset/univariate_ts.db
```

## 2. 코드·단위 테스트

저장소 루트에서 실행한다.

```bash
python3 -m py_compile \
  experiments/exp151_152/virtual_run_policy_core.py \
  experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  experiments/exp151_152/test_exp151_152_virtual_run_conformal_policy_v2.py

python3 -m unittest \
  experiments/exp151_152/test_exp151_152_virtual_run_conformal_policy_v2.py
```

## 3. 10개 dataset smoke test

```bash
SMOKE_OUT=outputs/exp151_152_virtual_run_conformal_v2_smoke

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase audit \
  --dataset-limit 10 \
  --skip-db-hash \
  --output-dir "$SMOKE_OUT"

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase predict \
  --dataset-limit 10 \
  --workers 2 \
  --reset-checkpoint \
  --output-dir "$SMOKE_OUT"

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase evaluate \
  --dataset-limit 10 \
  --output-dir "$SMOKE_OUT"
```

Smoke 결과에서 확인한다.

- prediction JSONL에 `hard_tp`, `hard_fp`, `combined_f1`, `y_test` 필드가 없어야 한다.
- `05_prediction_manifest.json`에 두 prediction hash가 있어야 한다.
- evaluation CSV에서는 Hard/Standard/Priority가 상호 배타적이어야 한다.
- Hard가 무조건 0이어서는 안 된다는 뜻은 아니다. 단, Block-B map이 비어 있어서 0인지 반드시 확인한다.

## 4. 전체 1,117개 실행

새 output directory를 사용한다.

```bash
OUT=outputs/exp151_152_virtual_run_conformal_v2

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase audit \
  --output-dir "$OUT"

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase predict \
  --workers 7 \
  --reset-checkpoint \
  --output-dir "$OUT"

python3 experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py \
  --phase evaluate \
  --output-dir "$OUT"
```

전체 audit에서는 `--skip-db-hash`를 사용하지 않는다.

## 5. 핵심 산출물

```text
01_evaluation_contract.json
02_virtual_run_grain_audit.csv
03_exp151_current_source_predictions.jsonl
04_exp152_b2_source_predictions.jsonl
05_prediction_manifest.json
06_exp151_current_source_results.csv
07_exp151_current_source_summary.csv
08_exp151_current_source_summary.md
09_exp152_b2_source_results.csv
10_exp152_b2_source_summary.csv
11_exp152_b2_source_summary.md
12_alpha_sensitivity_comparison.csv
13_dataset_level_candidate_diff.csv
15_presentation_claims.md
MANIFEST.csv
MANIFEST.md
```

## 6. 결과 해석

- Alpha 네 개를 모두 보고한다.
- 가장 높은 precision/F1 또는 가장 낮은 FP를 보고 alpha를 선택하지 않는다.
- Hard alert만 autonomous다.
- Review precision과 combined F1은 사람이 확인한다고 가정한 진단 지표다.
- Exp151/152가 성공해도 feature/configuration 선택 provenance와 실제 설비 prospective 검증은 별도 과제다.
