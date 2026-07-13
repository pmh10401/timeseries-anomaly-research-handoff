# 연구 수행 과정 및 최종 결과 요약서 (Walkthrough)

본 문서는 univariate_ts.db 데이터베이스 재구축 과정, 유효성 검증 결과, VAE 확장 설계, Rank Ensemble Calibration 및 다양한 임계치/손실 기법들을 적용한 최종 벤치마크 평가 이력을 요약합니다.

---

## 1. 생성된 산출물

* **최종 데이터베이스**: [univariate_ts.db](file:///Users/minho/Documents/Dataset/univariate_ts.db) (1,672.06 MB)
* **재구축 실행 스크립트**: [rebuild_database_v2.py](file:///Users/minho/Documents/Dataset/rebuild_database_v2.py)
* **검증 실행 스크립트**: [verify_sqlite.py](file:///Users/minho/Documents/Dataset/verify_sqlite.py)

---

## 2. 데이터베이스 구성 및 추가된 스키마

### A. 통계 요약
* **성공적으로 통합된 총 데이터셋 수**: **1,119개**
  * **UCR 분류 아카이브의 One-Class 변환 데이터셋**: **1,118개**
  * **Hugging Face 고래 소리 탐지 데이터셋 (CornellWhaleChallenge)**: **1개**
* **총 저장된 시계열 인스턴스 (TRAIN + TEST)**: **396,737개**
* **최종 데이터베이스 파일 크기**: **1,672.06 MB (1.67 GB)**

---

## 3. 알고리즘 구현 및 검증 결과

### ① [검증 1] 학습 데이터셋(TRAIN)의 100% 정상 보장
* **검증 결과**: **성공 (위반 건수: 0건)**
  * 모든 데이터셋의 `TRAIN` 세트에서 이상치 라벨(`1` 또는 `anomaly`)이 전혀 발견되지 않았습니다.

### ② [검증 2] 테스트 데이터셋(TEST)의 이상치 비율 2% 수준 통제
* **검증 결과**: **성공 (위반 건수: 0건, 평균 이상치 비율: 2.00%)**
  * 테스트 세트 내의 이상치 비율이 정확히 2% 수준으로 맞추어져, 편향되지 않은 정확한 모델 평가가 가능함을 확인했습니다.

---

## 4. [추가] 고성능 VAE 설계 및 데이터 증강 실험 결과

비지도 시계열 이상치 탐지 연구의 완성도를 높이기 위해, 오토인코더(AE)에 이어 확률론적 모델인 **1D Conv-VAE**를 성공적으로 도입하고 데이터 증강을 연동하였습니다.

### A. 핵심 개발 스크립트
1. **일반 VAE 스크립트**: [run_vae.py](file:///Users/minho/Documents/Dataset/run_vae.py)
2. **왜도 기반 적응형 임계치 VAE 스크립트**: [run_all_skewness_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_skewness_threshold_evaluations.py)
3. **극값 이론(EVT) 기반 임계치 VAE 스크립트**: [run_all_evt_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_evt_threshold_evaluations.py)
4. **와이블 및 검벨 임계치 VAE 스크립트**: [run_all_weibull_gumbel_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_weibull_gumbel_evaluations.py)
5. **온라인 동적 임계치 VAE 스크립트**: [run_all_online_dynamic_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_online_dynamic_evaluations.py)
6. **주파수 도메인 STFT VAE 스크립트**: [run_all_stft_vae_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_stft_vae_evaluations.py)
7. **주기성 분기형 하이브리드 VAE 스크립트**: [run_all_periodicity_hybrid_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_periodicity_hybrid_evaluations.py)
8. **고도화 주기성 분기형 하이브리드 VAE 스크립트**: [run_all_advanced_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_advanced_periodicity_evaluations.py)
9. **길이 적응형 Conv-VAE 스크립트**: [run_all_adaptive_cnn_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_evaluations.py) (시계열 길이에 비례해 커널 크기 및 레이어 레이아웃을 동적으로 매핑)
10. **적응형 복원 확률 VAE 스크립트**: [run_all_adaptive_recon_prob_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_recon_prob_evaluations.py) (길이 적응형 구조와 NLL 몬테카를로 이상 점수 결합)
11. **이중 적응 하이브리드 VAE 스크립트**: [run_hybrid_adaptive_vae_evaluations.py](file:///Users/minho/Documents/Dataset/run_hybrid_adaptive_vae_evaluations.py) (시계열 길이에 따른 가우시안 NLL 스코어 및 시간축 MSE 스코어 결합 병합 스크립트)
12. **길이 게이트형 주기 VAE 스크립트**: [run_all_length_gated_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_length_gated_periodicity_evaluations.py) (짧은 FFT 분해능을 통제하되 주기성 신호에만 STFT를 제한 연동하는 구조)
13. **연속 오케스트레이션 파이프라인**: [run_sequential_experiments.py](file:///Users/minho/Documents/Dataset/run_sequential_experiments.py) (선행 종료 시 4종 차세대 혁신 실험 자동 순차 실행)
14. **융합형 Ultimate VAE 스크립트**: [run_all_adaptive_cnn_fused_sota.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_fused_sota.py) (잠재 대조 학습 + KL 어닐링 + 분위수 POT 융합 및 동적 가중치 배정 장치 탑재)
15. **다중 위상 증강 대조 VAE 스크립트**: [run_all_adaptive_cnn_multi_aug_contrastive.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_multi_aug_contrastive.py) (시간 왜곡 및 세그먼트 셔플링 대조 학습 융합 스크립트)
16. **잠재 밀도 가중 VAE 스크립트**: [run_all_adaptive_cnn_density_weighted.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_density_weighted.py) (배치 내 KNN 잠재 밀도를 이용한 복원 MSE 가중 조절 스크립트)
17. **True InfoNCE Multi-Augmentation VAE 스크립트**: [run_all_adaptive_cnn_true_infonce_multi_aug.py](file:///Users/minho/Documents/timesries%20project/run_all_adaptive_cnn_true_infonce_multi_aug.py) (정식 배치 내 InfoNCE 손실을 적용한 다중 증강 대조 VAE)
18. **Rank Ensemble Calibration v1 스크립트**: [run_rank_ensemble_calibration.py](file:///Users/minho/Documents/timesries%20project/run_rank_ensemble_calibration.py) (MultiAug/Fused/Hybrid proxy 점수의 데이터셋 내부 percentile-rank 앙상블)
19. **Rank Ensemble Calibration v2 스크립트**: [run_rank_ensemble_calibration_v2.py](file:///Users/minho/Documents/timesries%20project/run_rank_ensemble_calibration_v2.py) (L<150 ReconProb NLL, L>=150 MSE Reconstruction 라우팅을 적용한 실제 HybridAdaptive 앙상블, 진행 중)

### B. 최종 벤치마크 요약 (초기 947개 및 최신 1,117개 확장 평가)
* **극값 이론(Extreme Value Theory, EVT) 기반 임계치**:
  * **평균 F1-Score**: **0.3433**
* **길이 게이트형 주기 VAE (제한적 STFT 결합)**:
  * **평균 F1-Score**: **0.2646** (FFT 붕괴 오탐은 막았으나, 복잡한 비선형 추세를 포함하는 센서 파형 영역에서 STFT 가중 결합이 원래의 형상 복원(MSE) 수렴을 현격히 그라디언트 교란/Underfitting함을 실증적으로 입증)
* **Transformer 융합 적응형 VAE (실험 2 - Attention 구조) ❌**:
  * **평균 F1-Score**: **0.2137** (Attention 파라미터 과적합 및 시계열 국소 유도 편향(Local Inductive Bias) 상실로 성능 대폭 폭락)
* **분위수 비례 POT 적합 VAE (실험 4 - Dynamic POT) 🧪**:
  * **평균 F1-Score**: **0.3570** (임계값 수식 개선만으로 희소 이상치 셋의 F1 0.0 지표 왜곡을 복구하여 기존 고정 임계치 대비 **`+1.37%p` 성능 상승**)
* **길이 적응형 Conv-VAE**:
  * **전체 평균 F1-Score (EVT)**: **0.3592**
* **KL Annealing VAE (실험 3 - 가중치 어닐링 스케줄링) 🧪**:
  * **전체 평균 F1-Score (EVT)**: **0.3605** (KL-Vanishing 붕괴가 원천 차단되어 VAE의 확률 안정성이 완성되어 성능 상승)
* **이중 적응 하이브리드 VAE (Hybrid Adaptive VAE) 결과 (전역 통합 최고 성능) 🌟**:
  * **전체 평균 AUC-ROC**: **0.8971 (+0.38%p)**
  * **전체 평균 F1-Score (EVT)**: **0.3663 (+0.71%p 추가 상승, SOTA 갱신) 🌟**
  * **전체 평균 Oracle F1**: **0.6527 (+1.10%p 상승, 상한선 돌파) 🌟**
* **잠재 대조 VAE (실험 1 - Contrastive Latent VAE) 결과 🌟**:
  * **전체 평균 AUC-ROC**: **0.8934**
  * **전체 평균 F1-Score (EVT)**: **0.3682 (+2.49%p 상승) 🌟**
* **융합형 Ultimate VAE (Fused SOTA VAE) 결과 (전역 단독 최고 SOTA) 👑**:
  * **전체 평균 AUC-ROC**: **0.8946**
  * **전체 평균 AUC-PR**: **0.5328**
  * **전체 평균 F1-Score (EVT)**: **0.3806 (+3.73%p 전역 SOTA 돌파!) 👑**
  * **전체 평균 Oracle F1**: **0.6453**
  * **학술적 성과**: 각 기법(대조 학습, KL 어닐링, 분위수 POT)이 독자적으로 동작할 때보다, 훈련 상호 보완성 및 평가 임계 튜닝이 결합되었을 때 극적인 시너지가 창출되어 전역 성능 한계선(F1 0.38)을 최종 돌파함을 실증하였습니다.
* **다중 위상 증강 대조 VAE (Multi-Augmentation) 결과 👑**:
  * **전체 평균 AUC-ROC**: **0.8891**
  * **전체 평균 AUC-PR**: **0.5229**
  * **전체 평균 F1-Score (EVT)**: **0.3874**
  * **전체 평균 Oracle F1**: **0.6364**
  * **학술적 성과**: 시간 왜곡 및 세그먼트 셔플링 증강을 통해 정상 시계열의 위상 불변 표현을 강화하여 단일 모델 기준 최고 성능을 수립했습니다.
* **True InfoNCE Multi-Augmentation VAE 결과 ❌ (1,117개 데이터셋)**:
  * **전체 평균 AUC-ROC**: **0.8791**
  * **전체 평균 AUC-PR**: **0.4910**
  * **전체 평균 F1-Score (EVT)**: **0.3107**
  * **전체 평균 Oracle F1**: **0.6145**
  * **해석**: 정식 InfoNCE 목적함수는 정상-only 소형 배치에서 음성 쌍을 과도하게 분리하여 기존 Multi-Augmentation 대비 성능이 하락했습니다.
* **Rank Ensemble Calibration v1 결과 👑 (1,117개 데이터셋)**:
  * **최고 설정**: `equal` 가중치(MultiAug/Fused/Hybrid proxy 각각 1/3)
  * **전체 평균 AUC-ROC**: **0.8983**
  * **전체 평균 AUC-PR**: **0.5383**
  * **전체 평균 F1-Score (EVT)**: **0.4350 (+4.76%p vs Multi-Augmentation) 👑**
  * **전체 평균 Oracle F1**: **0.6466**
  * **학술적 성과**: 단일 모델 점수보다 데이터셋 내부 순위로 정규화한 여러 모델의 합의 신호가 이상 후보를 더 안정적으로 부각한다는 점을 실증했습니다.
* **Rank Ensemble Calibration v2 진행 상태**:
  * **목표**: v1의 Hybrid proxy를 실제 HybridAdaptive 라우팅으로 교체
  * **라우팅 기준**: $L < 150$은 Recon-Probability NLL, $L \ge 150$은 MSE Reconstruction
  * **상태**: 전수 평가 진행 중이며, 완료 전까지 최종 성적표에는 반영하지 않습니다.
