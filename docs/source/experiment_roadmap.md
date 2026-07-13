# 비지도 시계열 이상치 탐지 성능 개선 로드맵 (Experiment Roadmap)

본 문서는 1D Conv-VAE 기반의 시간축 보존형 이상치 탐지 프레임워크 성능을 추가로 끌어올리기 위한 향후 실험 계획 및 구체적인 가이드를 정리한다.

---

## 🚀 실험 우선순위 요약 (Priority List)

1. **[완료] 시계열 데이터 증강(Augmentation) 도입**
   - **결과**: 평균 Oracle F1-Score **0.5036 ➡️ 0.6223 (+11.87%p)**로 대폭 증가. 정상 패턴의 일반화 한계를 확장하여 과적합 차단 효과 입증.
   - **소스코드**: [run_all_vae_evaluations_augmented.py](file:///Users/minho/Documents/Dataset/run_all_vae_evaluations_augmented.py)
2. **[완료] 왜도(Skewness) 기반 적응형 임계치 필터**
   - **결과**: 실무 평균 F1-Score **0.2859 ➡️ 0.3413 (+5.54%p)**, 중간값 F1-Score **0.2500 ➡️ 0.3333 (+8.33%p)**로 급상승. 우향 왜도(Right-skewed) 분포 정합성 입증.
   - **소스코드**: [run_all_skewness_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_skewness_threshold_evaluations.py)
3. **[완료] 소형 데이터셋 판별 점수 역전(Score Inversion) 해결**
   - **결과**: PCA-KDE 및 EuclideanMean 하이브리드 Fallback 검증 완료. 소형군에서 VAE 단독 학습의 비선형 피처 복원력이 전통 통계 모델보다 우수함을 입증하여, 데이터 증강형 VAE 단독 모델을 최종 아키텍처로 유지하기로 결정.
   - **소스코드**: [run_all_hybrid_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_hybrid_evaluations.py)
4. **[완료] VAE의 "복원 확률(Reconstruction Probability)" 기반 스코어링**
   - **결과**: 평균 F1-Score **0.2872**로 일반 MSE-VAE 대비 감소. 소형 데이터셋에서 분산 파라미터 $\sigma_x^2(z)$의 추정 수렴 불안정성으로 인해 등분산 가정(Homoscedastic MSE) 손실 함수를 쓰는 1D Conv-VAE의 성능이 더 안정적임을 규명.
   - **소스코드**: [run_all_recon_prob_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_recon_prob_evaluations.py)
5. **[완료] Extreme Value Theory (EVT, 극값 이론) 기반 임계치 파레토 분포 적합**
   - **결과**: 동일 가중치 조건 하에서 왜도 기반 적응형 임계치(0.3339) 대비 F1-Score가 **0.3410 (+0.71%p)**로 개선. 꼬리 분포 영역의 GPD(Generalized Pareto Distribution)를 정밀 모사하여 오탐 억제 효과 검증.
   - **소스코드**: [run_all_evt_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_evt_threshold_evaluations.py)
6. **[완료] 설비 도메인 와이블(Weibull) 및 검벨(Gumbel) 임계치 검증**
   - **결과**: 와이블(0.3223), 검벨(0.3175) 임계치 전략 검증 완료. 오차 분포를 무리하게 단순화하거나 데이터 손실(Block Maxima)이 발생하는 한계로 왜도 적응형 및 파레토 분포 대비 성능 저조 규명.
   - **소스코드**: [run_all_weibull_gumbel_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_weibull_gumbel_evaluations.py)
7. **[완료] 온라인 동적 임계값 및 EMA 필터링 검증**
   - **결과**: 지수이동평균(EMA) 필터 및 윈도우 볼린저 마진 설계 도입 전수 평가 완료. 이상치 점수 희석(Peak Dilution) 및 분산 팽창 피드백 교란 루프로 인해 F1-Score가 **0.0906**으로 폭락함을 발견하여 비지도 환경 하에서의 정적 전역 극값(EVT-GPD) 임계치 고정의 강력한 강인함을 규명함.
   - **소스코드**: [run_all_online_dynamic_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_online_dynamic_evaluations.py)
8. **[완료] 시간-주파수 도메인 STFT 복원 손실 VAE 검증**
   - **결과**: 시간 도메인 MSE 오차와 STFT Magnitude 오차를 융합한 하이브리드 손실 전수 평가 완료. UCR 비주기성 기하 특징 시계열 데이터 조건 하에서 스펙트럼 허상(Boundary Artifacts)이 시간 형태 정보 복원력을 해쳐(Underfitting) 성능이 대폭락(F1 0.2384, AUC 0.6848)함을 실증함.
   - **소스코드**: [run_all_stft_vae_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_stft_vae_evaluations.py)
9. **[완료] 주기성 판별 분기형 하이브리드 VAE 검증**
   - **결과**: ACF 분석 기반 주기성 자동 진단 및 Selective Spectral Loss ($\lambda_{spec}$) 분기 파이프라인 전수 평가 완료. 시계열 내 강한 선형 추세(Trend)가 존재할 시 ACF가 느리게 감소하여 주기성(`True`)으로 오판되는 통계적 취약성이 발견되었고, 이로 인한 STFT 손실 혼입이 형태 복원을 교란하여 성능이 저하(F1 0.3239)됨을 규명함.
   - **소스코드**: [run_all_periodicity_hybrid_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_periodicity_hybrid_evaluations.py)
10. **[완료] 고도화 주기성 판별 분기형 하이브리드 VAE 검증**
    - **결과**: 선형 디트렌딩(Detrend) 및 FFT 스펙트럼 엔트로피 연동형 고도화 판별 파이프라인 전수 평가 완료. 디트렌딩에 의한 불규칙 노이즈의 ACF 왜곡 및 시계열의 물리적 차원이 극도로 짧을 시 발생하는 PSD 엔트로피 수식 붕괴 효과로 인해 주기성 판정 비율이 **90.81%**로 비정상 폭증하며 최종 성능 폭락(F1 0.2480)을 초래함을 실증함.
    - **소스코드**: [run_all_advanced_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_advanced_periodicity_evaluations.py)
11. **[완료] 적응형 복원 확률 VAE 기반 1D CNN 표현력 제약 극복 검증**
    - **결과**: 길이 적응형 이중 헤드 디코더 및 50회 몬테카를로 NLL 스코어링 결합 전수 평가 완료. 짧은 시계열 군($L < 150$, 168개)에서 F1-Score가 고정 VAE 및 적응형 MSE VAE 대비 각각 **+7.89%p**, **+4.01%p** 이상 획기적으로 상승(F1 **0.3678**)하여 분산 붕괴 억제 시너지 효과를 성공적으로 증명함. 단, 긴 시계열군에서는 디코더 듀얼 헤드의 그라디언트 상호 간섭으로 소폭 하락(F1 0.3452)함을 발견함.
    - **소스코드**: [run_all_adaptive_recon_prob_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_recon_prob_evaluations.py)
12. **[우선순위 1] 시계열 길이 분기형 복합 적응 VAE (Hybrid Adaptive VAE) 구축**
    - **목적**: 짧은 군에는 적응형 복원확률 VAE(NLL)를 씌우고, 긴 군에는 적응형 MSE VAE를 씌워 전역 성능을 극대화(예상 F1 **0.3663** 돌파)하는 최종 파이프라인 수립.
13. **[우선순위 2] 잠재 공간 대조 학습 (Contrastive Latent Representation) 도입**
    - **목적**: VAE 디코더의 형상 복원 무결성을 유지하되, 정상 데이터의 잠재 공간상 응집력(Compactness)을 높여 이상 데이터 유입 시의 복원 오차 마진 극대화

---

## 🛠️ 세부 실험 설계 가이드

### 1. [완료] 시계열 데이터 증강 (Time Series Augmentation)
* **기법**: Jittering (std=0.03) 및 Scaling (0.9 ~ 1.1) 기법을 결합하여 데이터 복제 없이 500개의 흔들림 파형 생성.
* **증명**: 오라클 F1 성능이 수직 상승하여 이상치 변변 경계선을 효과적으로 정화함을 입증 완료.

### 2. [완료] 왜도(Skewness) 기반 임계치 자동 분기
* **기법**: 학습 에러 왜도(Skewness)를 계산하여 1.2 초과 시 Log-Normal, 0.2 미만 시 Gaussian Normal, 그 외 Gamma 분포로 적응형 모델을 분기 적합.
* **증명**: 고정 백분위수의 통계적 한계를 우회하고, 우측 비대칭 꼬리가 긴 오차 분포에 가장 알맞은 통계 경계를 매핑하여 F1 성능 상승 완료.

### 3. [완료] 소형 데이터셋 점수 역전(Score Inversion) 해결 설계
* **결론**: 극소형 데이터셋(N < 30)에서 딥러닝 과적합 제어를 위해 PCA-KDE 등으로 분기하는 하이브리드 파이프라인을 구축해 보았으나, 1D CNN의 비선형 시간 위상 변이 포착력과 데이터 증강(Jittering & Scaling) 효과가 전통적인 선형 기법 대비 소형 데이터셋 조건에서도 더 뛰어난 일반화 강인성을 제공함을 규명. 따라서 증강형 VAE 단독 모델이 최종적인 최적 아키텍처로 낙점됨.

### 4. [완료] 복원 확률 (Reconstruction Probability)
* **결론**: 디코더가 평균 $\mu_x$와 분산 $\sigma_x^2$를 동시 출력하도록 설계해 몬테카를로 음의 로그 우도(NLL)를 점수로 사용했으나, 분산 파라미터의 수렴 붕괴 문제로 노이즈에 매우 취약해져 F1 점수가 하락함. UCR과 같이 샘플 수가 불균일한 소형 벤치마크 환경에서는 등분산 가정이 훨씬 강건한 일반화를 유도함을 확인.

### 5. [완료] 극값 이론 (Extreme Value Theory, EVT) 기반 임계값 설정
* **기법**: 복원 에러의 상위 90% 이상을 임계 초과 영역(Excesses)으로 획정하고, 이 데이터에 Generalized Pareto Distribution (GPD)을 적합하여 꼬리 분포를 정밀 모사. 수렴 실패 시 왜도 적응형 임계치로 자동 Fallback 적용.
* **증명**: 일반적인 연속 확률 분포가 잡아내지 못하는 극한의 꼬리 경계면을 매핑하여 False Positive를 효과적으로 차단, 최종 F1-Score **0.3430**으로 성능 갱신 완료.

### 6. [완료] 설비 도메인 와이블(Weibull) 및 검벨(Gumbel) 임계치 검증
* **기법**: 전체 복원 오차에 2-parameter 와이블 분포를 직접 피팅하거나, 시계열을 블록화하여 블록 최댓값(Block Maxima)에 Gumbel 분포를 피팅하는 동적 분위수 역산 임계치 설계.
* **증명**: 설비 수명 분야의 표준 분포가 비지도 이상치 탐지 오차 임계치 선정 문제에서는 전체 분포 왜곡 문제 및 데이터 유실 문제를 초래하여, 오직 극한 꼬리만을 타겟팅하는 GPD(POT) 대비 열위함을 통계적으로 증명 완료.

### 7. [완료] 온라인 동적 임계값 및 EMA 필터링 검증
* **기법**: 테스트 데이터의 실시간 유입 상황을 윈도우 버퍼로 트래킹하여 슬라이딩 지수이동평균(EMA) 필터링 및 로컬 표준편차 볼린저 밴드 보정 임계값 매핑.
* **증명**: 짧고 급격한 이상 스파이크 점수를 평활화로 뭉개버려 재현율을 망가뜨리고, 윈도우 내 이상 에러가 누적됨에 따라 분산이 급팽창하여 뒤따르는 이상 감지를 완전히 무력화하는 Variance Inflation Feedback Loop의 통계적 역효과를 증명 완료. 이에 따라 고정 전역 EVT-GPD 전략이 비지도 실무에서 가장 우수한 일반화 성능을 제공함을 확인.

### 8. [완료] 시간-주파수 도메인 STFT 복원 손실 VAE 검증
* **기법**: 입력 파형과 복원 파형의 STFT Magnitude 간의 Frobenius Norm 및 Log L1 손실을 가중 합산한 하이브리드 복원 VAE 모델 설계 및 전수 평가.
* **증명**: 비주기성/기하형태 지배형 1차원 UCR 시계열에서는 주파수 변환 오차 유입이 오히려 디코더의 정상 형상 복원을 방해(Underfitting)하여 변별 한계를 떨어뜨림을 규명. 즉, 범용 이상 탐지에는 시간 도메인 복원 손실 VAE 단독 구조가 안정된 일반화를 냄을 규명함.

### 9. [완료] 주기성 판별 분기형 하이브리드 VAE 검증
* **기법**: 학습 데이터의 ACF Peak 탐색을 기반으로 주기성 유무를 자동 진단하여, 손실 함수 파라미터 $\lambda_{spec}$ 및 스코어링 수식을 선택적으로 분기 적용.
* **증명**: 단순 ACF 피크 탐색은 선형 추세(Trend)나 변위(Drift)를 주기성으로 오판하는 높은 오류율을 지녀, 비주기성 데이터에 STFT Loss 결합 교란을 전파하여 전체 평균 F1-Score의 하락(0.3433 -> 0.3239)을 초래함을 규명. 비지도 이상 탐지에서는 형태 복원 VAE를 전단 고정하는 것이 가장 강건함을 최종 입증 완료.

### 10. [완료] 고도화 주기성 판별 분기형 하이브리드 VAE 검증
* **기법**: 선형 디트렌딩 전처리 및 PSD 기반 스펙트럼 엔트로피, ACF 복합 로직 판별기를 내장하여 주기성 여부를 자동 분류 후 손실 함수 및 평가지표 분기 대조.
* **증명**: UCR 데이터셋처럼 시계열이 협소하고(Bin 개수 부족) Z-정규화된 구조에서는 선형 추세선 제거가 오히려 랜덤 노이즈의 위상 피크를 증폭시키는 아티팩트를 자아내어 90.81%의 기괴한 오탐 진단을 초래함을 규명. 인위적인 도메인 분기보다 형태 정보를 100% 보존하는 순수 시간축 MSE VAE가 최강의 일반화 강인함을 가짐을 입증 완료.

### 11. [완료] 적응형 복원 확률 VAE 기반 1D CNN 표현력 제약 극복 검증
* **기법**: 입력 시계열 길이 L에 비례해 토폴로지를 최적화하는 VAE 구조에 복원 확률(NLL 몬테카를로 샘플링)을 결합하여 이중 분기 디코더로 훈련.
* **증명**: 얕고 정화된 레이어 덕분에 분산 수렴 붕괴가 영구 방어되어 짧은 시계열 군($L < 150$, 168개)에서 F1-Score가 **0.3678**로 수직 상승하여 확률론적 스코어가 갖는 정밀 묘사 우위성을 실증 완료. 단, 긴 시계열군에서는 듀얼 헤드의 평균/분산 그라디언트 상호 간섭으로 인해 약간의 성능 교란이 존재함을 규명함.

