# Exp137 발표 입력 자료 인수인계

## 목적과 범위

이 폴더는 `experiment_137_operational_triage`의 결과를 발표용으로
검증하고 원본 자료를 전달하기 위한 패키지다. PPT/PDF 디자인 산출물은
포함하지 않는다. 핵심 목표는 단일 Mean F1 최대화가 아니라 자동 경보의
false alarm을 줄이고 사용자 신뢰를 높이는 것이다.

## 가장 먼저 읽을 파일

1. `MANIFEST.md`: 모든 산출물, 입력, 생성 명령, 주의사항
2. `05_exp137_verification_report.md`: Exp137 수치 재집계 및 무결성 검증
3. `07_before_after_comparison.md`: 기존 후보 대비 자동 경보 성과
4. `09_train_only_lineage_audit.md`: TRAIN-only 원칙 계보 감사와 위험 요소
5. `17_priority_review_rule_audit.md`: 우선 검토 9건의 후향 규칙 한계
6. `18_operational_flow.md`: 발표 용어 기준의 실제 운영 흐름

## 핵심 수치

- 평가 데이터셋: 1,117
- 자동 경보: 2,005건, TP 1,691건, FP 314건, precision 84.339%
- 일반 검토: 639건, TP 292건, FP 347건, precision 45.696%
- 우선 검토: 9건, TP 8건, FP 1건, precision 88.889%
- mean Hard F1: 0.600772 (자동 경보만)
- mean combined F1: 0.700776 (사람이 검토 lane을 확인했다고 가정한 진단 지표)
- 기존 후보 FP 661건에서 자동 경보 FP 314건으로 347건, 52.496% 감소

## 해석 원칙

- `mean_f1`은 자동 경보 성능이다. `mean_combined_f1`은 사람 검토를 포함한
  후향 진단 지표이므로 둘을 같은 의미로 발표하면 안 된다.
- 우선 검토는 retrospective research rule이다. 9건이라는 작은 표본과
  후향 설계 때문에 prospective validation 전에는 자동 경보로 승격하면 안 된다.
- 최종 Exp137 라우팅은 TEST label, TEST 이상 위치, 과거 family별 TEST 성능을
  사용하지 않는 것으로 확인되었다.
- 다만 과거 후보 생성 계보에는 family guard 및 `y_test` 길이를 입력으로 받은
  large-data budget 코드가 존재한다. 운영 배포 전에는 `09_train_only_lineage_audit.md`
  의 strict train-only 위험 항목을 분리 재검증해야 한다.
- 이전 tail replacement 방식은 이 패키지와 운영 후보에 포함하지 않는다.

### 발표 검증 범위 문구

발표에서는 다음 범위를 넘어서 주장하지 않는다.

> Exp137의 최종 라우팅은 TEST 라벨과 실제 이상 위치를 사용하지 않는 것으로
> 검증됐다. 다만 과거 실험에서 이어진 상위 후보 생성 계보 전체는 strict
> train-only, 즉 완전한 학습 정상 데이터 전용 구조인지 추가 검증이 필요하다.
> 따라서 최종 라우팅의 독립성은 확인됐지만, 전체 파이프라인이 완전히 검증됐다고
> 과장하지 않는다.

## 재현에 필요한 외부 입력

원본 데이터베이스와 대형 결과 CSV는 압축본 크기를 통제하기 위해 포함하지 않았다.

- 데이터베이스: `/Users/minho/Documents/Dataset/univariate_ts.db`
- 결과 CSV: `/Users/minho/Documents/Dataset/`
- 저장소: `/Users/minho/Documents/timesries project`
- 기준 commit: `05c471fd650f34ca641944d7524f6484a84f4fb2`

DB는 약 1.6GB이며, 이 패키지 안의 사례별 train/test CSV는 발표 설명과
시각화에 필요한 최소 사본이다. 전체 재현에는 위 절대 경로의 DB와 결과 파일이
필요하다.

## 포함한 코드 사본

`source_snapshot/`에는 Exp137, Exp133~135, Exp89~93, Exp40, Exp29, Exp26의
후보 생성 및 교차 확인 계보와 대시보드/순차 실행 보조 코드, Exp137 단위 테스트를
읽기 전용 사본으로 포함했다. 이 사본은 감사와 설명용이며, 원본 저장소 코드를
수정하지 않는다.

## 검증 상태

`python3 -m unittest scratch/test_experiment_137_operational_triage.py`는
5개 테스트 모두 통과했다. 전체 실행 로그는 `02_unit_test_full_log.txt`에 있다.

## 발표 자료 사용 시 구분

- 자동 경보: 사용자가 즉시 볼 운영 경보
- 일반 검토: 사람이 확인할 후보
- 우선 검토: 후향 분석에서 유망했지만 아직 자동화하면 안 되는 연구 후보
- 사례의 실제 이상 위치: 사후 평가 및 설명 표시에만 사용되며, 라우팅 입력이 아님
