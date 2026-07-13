# Ready-to-Paste Prompt for a New GPT Conversation

```text
당신은 시계열 이상 탐지 연구 프로젝트의 후속 담당자다. 작업 디렉터리는
`/Users/minho/Documents/timesries project`이고, 대용량 데이터와 결과 CSV는
`/Users/minho/Documents/Dataset`에 있다.

먼저 아래 인수인계 파일을 모두 읽고, 현재 상태를 짧고 정확하게 요약해라.

1. `outputs/exp137_gpt_handoff_20260713/HANDOFF_EXP137_DETAILED_KO.md`
2. `outputs/exp137_gpt_handoff_20260713/EXPERIMENT_137_FLOW.md`
3. `outputs/exp137_gpt_handoff_20260713/DATASET_DB_REFERENCE.md`
4. `outputs/exp137_gpt_handoff_20260713/MANIFEST.md`

그 다음 다음 원칙을 반드시 지켜라.

- 운영 목표는 단일 mean F1 최대화가 아니라 false alarm 감소와 사용자 신뢰다.
- TRAIN 정상 데이터만으로 feature, score, threshold, alert budget을 만들어야 한다.
- TEST 라벨, TEST 이상 위치, 과거 family별 TEST 성능은 라우팅/선택/threshold에 사용하지 않는다.
- Exp137의 `mean_f1`은 autonomous Hard alert 성능이다.
- `mean_combined_f1`은 review lane까지 사용자가 확인한 human-assisted 진단 지표이며, autonomous 성능으로 발표하면 안 된다.
- Priority review는 retrospective research rule이다. 새 설비/레시피 데이터에서 prospective validation 전에는 자동 경보로 승격하지 않는다.
- Block A/B/C는 회사 발표에서는 쓰지 않는다. 각각 1차 탐지, 교차 확인, 보조 확인이라고 표현한다.
- 이전 tail replacement 방식은 평가셋의 위치 경향을 이용하는 것으로 판단되어 배제되었다. 재도입하지 않는다.

현재 핵심 실험은 `experiment_137_operational_triage`다.
검증된 결과는 다음과 같다.

- 1,117 datasets
- Hard alert: 2,005 alerts, TP 1,691, FP 314, precision 84.339%, mean hard F1 0.600772
- Standard review: 639 candidates, TP 292, FP 347, precision 45.696%
- Priority review: 9 candidates, TP 8, FP 1, precision 88.889%
- Routing uses test labels/position/family performance: all zero

가장 먼저 할 일:

1. `git status --short --branch`와 필요한 CSV를 확인하여 현재 파일 상태를 검증한다.
2. `python3 -m unittest scratch/test_experiment_137_operational_triage.py`를 실행한다.
3. Exp137 결과 CSV와 summary CSV를 다시 읽고 위 수치가 맞는지 확인한다.
4. 현재 목표에 가장 직접적인 다음 단계로, review lane의 prospective validation 또는 사용자 피드백 기록 설계를 제안한다. 자동 alert를 성급하게 늘리는 방향은 추천하지 않는다.

코드 변경이 필요하면 먼저 관련 테스트와 기존 구현을 읽고, 작은 변경만 수행한 뒤 검증 결과를 근거와 함께 보고하라. 장시간 실험은 한 번에 하나만 실행하며 실행 중인 프로세스와 대시보드 상태를 명확히 보여라.
```
