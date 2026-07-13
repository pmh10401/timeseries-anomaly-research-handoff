# Time-Series Anomaly Research Handoff

공개 연구 인계 저장소입니다. 이 저장소는 정상 데이터만으로 학습하는 시계열 이상치 탐지 연구의 코드, 실험 문서, 검증된 결과, 데이터베이스 메타데이터를 제공합니다. 대형 SQLite DB와 전체 원시 시계열 corpus는 포함하지 않습니다.

## 핵심 목적

이 연구의 운영 목표는 단일 Mean F1을 최대화하는 것이 아니라, 자동 경보의 false alarm을 줄여 사용자가 신뢰하고 검토할 수 있는 이상 후보를 만드는 것입니다.

- 학습, feature fitting, score calibration, threshold는 TRAIN 정상 데이터만으로 결정하는 것이 원칙입니다.
- TEST 라벨과 실제 이상 위치는 prediction 이후의 사후 평가에만 사용합니다.
- 자동 경보(Hard alert)는 autonomous 지표로 봅니다.
- 일반 검토(Standard review)와 우선 검토(Priority review)는 human-assisted 진단용이며, 자동 경보로 승격된 결과가 아닙니다.
- 과거 tail replacement 방식은 운영 후보에서 제외했습니다.

## 빠른 시작

1. [데이터베이스 안내](data/DATABASE_GUIDE.md)에서 데이터 구조와 공개 범위를 확인합니다.
2. [Exp137 인계 문서](results/exp137_gpt_handoff_20260713/README_START_HERE.md)와 [운영 흐름](results/exp137_gpt_handoff_20260713/EXPERIMENT_137_FLOW.md)을 읽습니다.
3. [Exp137 summary CSV](results/exp137_gpt_handoff_20260713/results/experiment_137_operational_triage_summary.csv)와 [데이터셋별 결과](results/exp137_gpt_handoff_20260713/results/experiment_137_operational_triage_results.csv)를 검토합니다.
4. [Policy validation 결과](results/exp137_policy_train_only_validation/)에서 B2, Exp143, C0, D1a를 구분해 확인합니다.
5. GPT에 전달할 때는 [GPT_REVIEW_PROMPT_KO.md](GPT_REVIEW_PROMPT_KO.md)와 필요한 CSV를 함께 사용합니다.

## 현재 핵심 증거

### Exp137 운영 라우팅 기준

평가 데이터셋 1,117개에서 Exp137의 autonomous Hard alert는 2,005건, TP 1,691건, FP 314건, micro precision 84.339%, mean Hard F1 0.600772입니다. Standard review는 639건(TP 292, FP 347), Priority review는 9건(TP 8, FP 1)이며 review lane 결과는 human-assisted 진단 지표입니다.

원본 결과: [summary CSV](results/exp137_gpt_handoff_20260713/results/experiment_137_operational_triage_summary.csv)

### Policy-level 검증 현황

- A0 파일 정합성과 A1 selector replay는 기존 Exp137 결과를 1,117/1,117 데이터셋에서 재현했습니다.
- B1 common-support에서는 family guard source가 달라진 38개 데이터셋이 있었으나 Hard alert는 변하지 않았습니다.
- B2 full-coverage family-neutral은 고정 Exp84 source를 1,117개 전체에 같은 방식으로 계산했습니다. Hard alert 2,085, TP 1,759, FP 326, precision 84.365%, mean Hard F1 0.605991입니다.
- Exp143은 TEST-length budget이 220개 데이터셋에서 실제 binding되었고 246개의 후보를 제거했음을 보여 주는 진단 감사입니다. 인과 증명은 아닙니다.
- C0와 D1a는 top-N budget을 제거했을 때 후보 workload와 FP가 크게 증가하는 counterfactual입니다. 자세한 수치는 [budget validation 결과](results/exp137_policy_train_only_validation/budget_full_7w_optimized/)에 있습니다.
- 전체 파이프라인이 end-to-end strict TRAIN-only이거나 실제 설비에서 prospective validation되었다고 말할 근거는 아직 없습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| root Python files | 원래 파일명으로 보존한 실험, 평가, DB 구축, dashboard 스크립트 |
| scratch/ | 원래 경로를 보존한 테스트와 검증 스크립트 |
| docs/source/ | 기존 연구 문서, 실험 분석, 계획, 과거 결과 |
| results/ | Exp137, B1/B2, policy-budget 검증의 CSV/JSON/Markdown 근거 |
| artifacts/ | 발표용 입력 문서와 편집 가능한 SVG/PNG 그림 |
| data/ | DB schema, 1,119개 데이터셋의 metadata catalog, 집계 통계 |
| MANIFEST.csv and MANIFEST.md | 공개 파일과 SHA256, 원본 경로, 제외 근거 |

## 공개하지 않은 항목

- /Users/minho/Documents/Dataset/univariate_ts.db (약 1.6 GB)
- 원시 시계열 value blob, label blob, 원시 UCR 디렉터리
- Python 가상환경과 캐시
- 재개용 JSONL checkpoint 및 임시/중복 archive
- dashboard 사용자명·비밀번호 등 실제 환경변수 값

발표 설명을 위해 선택한 18개 사례의 TRAIN/TEST CSV는 artifacts 아래에 소량 포함합니다. 이는 전체 DB나 전체 raw corpus가 아니라, Exp137 결과를 설명하기 위한 후향 사례 자료입니다.

## 재현 범위

이 저장소의 코드에는 로컬 DB 절대 경로를 참조하는 부분이 있습니다. 원본 DB가 없는 환경에서는 전체 rerun이 불가능합니다. 다만 결과 CSV, 평가 계약, 데이터셋 catalog, test code를 통해 실험의 구조와 결과를 검토할 수 있습니다.

정확한 실행 조건과 제한 사항은 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)를 참고하세요.

## 주의할 표현

- Hard alert precision은 전체 후보를 합산한 micro precision입니다.
- mean Hard F1은 데이터셋별 F1 평균(macro)입니다.
- mean combined F1은 review lane을 사람이 확인했다고 가정한 human-assisted 진단 지표입니다.
- B2/C0/D1a는 이미 관찰된 1,117개 TEST 데이터에 대한 retrospective counterfactual입니다. 새로운 설비, 신규 레시피, 시간적으로 이후 데이터에서 검증된 배포 성능이 아닙니다.

## License

코드와 문서의 공개 라이선스는 아직 확정하지 않았습니다. 재사용 전에 저장소 소유자에게 확인하세요.
