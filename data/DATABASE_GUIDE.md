# Database Guide

## 공개 범위

원본 SQLite DB는 /Users/minho/Documents/Dataset/univariate_ts.db에 있으며 약 1.6 GB입니다. 이 공개 저장소에는 DB를 포함하지 않습니다. 대신 schema, 데이터셋 metadata catalog, 집계 통계를 제공합니다.

공개 파일:

- [schema.sql](schema.sql): datasets 테이블 구조
- [dataset_catalog.csv](dataset_catalog.csv): 1,119개 데이터셋의 이름, 길이, split별 normal/anomaly 개수
- [dataset_summary.json](dataset_summary.json): 전체 집계

의도적으로 제외한 필드:

- instances.values_blob: 원시 시계열 값
- instances.labels_blob: instance 수준의 라벨 blob
- 원시 UCR 디렉터리와 외부 데이터 복사본

예외: artifacts/exp137_presentation_inputs/cases/에는 발표 설명을 위한 18개 선택 사례의 TRAIN/TEST CSV가 포함됩니다. 이는 전체 데이터셋이나 전체 raw corpus를 대체하지 않으며, 사후 설명용으로만 사용합니다.

## 데이터 구조

datasets 테이블은 한 데이터셋의 통계적 metadata를 저장합니다.

| 필드 | 의미 |
|---|---|
| name | 데이터셋 이름 |
| univariate | 단변량 여부 |
| equal_length | dataset 내부 시계열 길이 동일 여부 |
| series_length | 동일 길이일 때의 시계열 길이 |
| has_missing | 결측 존재 여부 |
| train_normal_count | TRAIN 정상 instance 수 |
| train_anomaly_count | TRAIN 이상 instance 수 |
| test_normal_count | TEST 정상 instance 수 |
| test_anomaly_count | TEST 이상 instance 수 |
| test_total_count | TEST 전체 instance 수 |

instances 테이블은 각 개별 시계열을 저장합니다. 공개하지 않은 values_blob에 실제 숫자 배열이, labels_blob에 해당 instance의 라벨 정보가 있습니다. 이 연구의 candidate index는 TEST 시계열 instance의 index이며, 한 시계열 안의 시간 위치가 아닙니다.

## 현재 DB에서 확인된 집계

- 데이터셋: 1,119개
- TRAIN 정상 instance: 170,937개
- TRAIN 이상 instance: 0개
- TEST 정상 instance: 221,284개
- TEST 이상 instance: 4,516개
- TEST 전체 instance: 225,800개
- TRAIN 정상 instance 수 범위: 1 ~ 18,378
- TEST instance 수 범위: 50 ~ 23,400
- 서로 다른 series_length 값: 139개

Exp137 전수 평가는 CornellWhaleChallenge와 Wafer_normal_1을 제외한 1,117개를 사용합니다. 제외 사유와 정확한 정책은 결과/평가 계약 문서를 기준으로 확인해야 합니다.

## 학습과 평가의 구분

기본 연구 원칙은 다음과 같습니다.

1. TRAIN 정상 instance만으로 feature extractor, score distribution, calibration, threshold, candidate budget을 정합니다.
2. TEST feature/score에서 prediction을 만듭니다.
3. TEST label은 prediction이 고정된 뒤에만 TP, FP, precision, recall, F1 계산에 사용합니다.

이 원칙은 목표 구조입니다. 현재 코드 계보 전체가 end-to-end strict TRAIN-only임이 증명된 것은 아닙니다. Exp137의 최종 routing은 TEST label과 실제 anomaly position을 사용하지 않는 것으로 검증되었지만, 상위 feature configuration과 과거 후보 계보의 provenance는 별도 검증 대상입니다.

## 산업 운영 해석의 한계

사용자와의 논의에서는 한 데이터셋 = wafer run x sensor x step이라는 운영 비유를 사용했습니다. 그러나 이 DB 파일 자체에는 실제 equipment ID, recipe ID, sensor ID, step ID, run 종료 시각, 유지보수 이력, 사용자 판정 이력이 없습니다. 따라서 DB만으로 TRAIN instance 하나가 실제 wafer run 하나인지 확인할 수 없습니다.

이 한계 때문에 TRAIN 정상 run의 후보 수 분포로 conformal budget을 만드는 C1/D1b는 현재 근거만으로 실행하지 않았습니다. 이를 운영 정책으로 쓰려면 실제 설비/레시피/run metadata와 prospective shadow validation이 필요합니다.

## 로컬 검증 명령

    sqlite3 '/Users/minho/Documents/Dataset/univariate_ts.db' 'PRAGMA integrity_check;'
    sqlite3 '/Users/minho/Documents/Dataset/univariate_ts.db' 'SELECT COUNT(*) FROM datasets;'
    sqlite3 '/Users/minho/Documents/Dataset/univariate_ts.db' 'SELECT COUNT(*) FROM instances;'

DB는 별도 관리 대상입니다. 공개 저장소에 복사하거나 커밋하지 마세요.
