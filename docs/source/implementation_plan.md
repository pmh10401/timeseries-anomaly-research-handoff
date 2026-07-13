# 실행 계획서: [개선 1, 3, 4 융합형 Ultimate VAE (Fused SOTA VAE) 실험]

본 계획서는 개별 검증 과정에서 각각 독자적 최우수 성능을 갱신했던 세 가지 기법(**잠재 대조 학습, KL 어닐링, 분위수 비례 POT 스케일링**)을 단일 파이프라인에 유기적으로 통합하여 성능적 임계 한계를 극복하는 융합 VAE 설계 및 실증 방안을 수립합니다.

---

## 1. 융합의 당위성 및 상호 통계적 시너지

| 결합 조합 | 통계적/구조적 시너지 메커니즘 |
| :--- | :--- |
| **[실험 1] 대조 학습<br>➕<br>[실험 3] KL Annealing** | * **그라디언트 충돌 해소**: 대조 학습(Contrastive)은 정상 패턴 $z$ 간의 밀집 유도를 수행하지만, KLD 제약이 초반에 너무 강하면 잠재 분포가 강제로 $\mathcal{N}(0, I)$ 붕괴를 겪어 대조 표상 학습이 무력화됩니다.<br>* **해결**: 초반 에폭(1~7)에 KL 가중치를 낮추는 Annealing을 걸어 정상 표상들의 기하적 대조 클러스터링 마진을 선제적으로 넓히고, 후반에 KLD를 인가해 가우시안 규칙성을 조화롭게 융합합니다. |
| **[훈련 융합 모델]<br>➕<br>[실험 4] Dynamic POT** | * **임계치의 칼끝 튜닝**: 대조 학습과 KL 어닐링이 융합된 디코더는 복원 오차 분포의 분산을 극단적으로 슬림하게 제어하여 오참을 최소화합니다.<br>* **해결**: 한결 얇고 날카로워진 에러 꼬리(Tail) 분포 위에, 테스트 셋 스케일 맞춤형 분위수 POT($q = \max(0.001, 1.0 / N_{test})$)를 융합 적용하여 오라클 지표에 근접하는 극강의 탐지력을 완성합니다. |

### 📐 융합 손실 함수 설계:
$$\mathcal{L}_{total} = \text{MSE}_{time} + \beta(t)\text{KLD} + \gamma(1.0 - \text{CosSim}(z_{orig}, z_{aug}))$$
* $\beta(t)$: 1~7 에폭 동안 $0 \to 0.001$로 점진 증가 후 $0.001$ 고정 (KL Annealing)
* $\gamma$: $0.1$ 고정 (Contrastive Cosine Similarity Weight)
* $q_{target}$: $\max(0.001, 1.0 / N_{test})$ (Dynamic Quantile)

---

## 2. 제안되는 변경 내용

### 신규 평가 벤치마크 스크립트 작성
#### [NEW] [run_all_adaptive_cnn_fused_sota.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_fused_sota.py)
* 길이 적응형 VAE 구조 상에 **KL Annealing**과 **InfoNCE 대조 손실**을 융합하여 훈련하고, 테스트 평가 시 **분위수 비례 POT 스케일링**을 연동해 이상 점수 임계값을 동적으로 컷오프하는 최종 통합 벤치마크 코드를 작성합니다.
* 947개 데이터셋 전수에 가속 MPS 디바이스를 할당하여 전체 F1 성능의 돌파(F1 `0.37+` 및 최종 SOTA)를 검증합니다.
* 결과를 [vae_results_adaptive_cnn_fused_sota.csv](file:///Users/minho/Documents/Dataset/vae_results_adaptive_cnn_fused_sota.csv)에 저장합니다.

---

## 3. 검증 계획

### 자동 및 단위 드라이런 테스트
* 융합 신경망의 그라디언트 갱신 중 역전파 충돌(Backpropagation conflict)이 없는지 샘플 데이터셋 `ACSF1_normal_2`에 대해 1에폭 드라이런 검증을 사전 수행합니다:
```bash
/Users/minho/Documents/Dataset/.venv/bin/python /Users/minho/Documents/Dataset/scratch/dry_run_fused_sota.py
```

### 성능 비교 및 깃허브 푸시
* 완성된 벤치마크 평균 지표를 도출하여 `research_report.md`에 추가 기입하고, `walkthrough.md`를 개정하여 깃허브 원격 리포지토리에 동기화 전송합니다.
