# 데이터셋별 최우수 성능 기법 (Best Performers) 정리 보고서

본 보고서는 947개 UCR 시계열 데이터셋 전수에 대해 수행된 9가지 오토인코더(VAE) 및 임계값 조합 실험 결과들을 취합하여, 각 데이터셋별로 가장 높은 F1-Score 성적을 획득한 최적의 모델과 성능 통계를 분석합니다.

## 1. 종합 요약 (Overall Summary)

각 기법이 개별 데이터셋에서 공동 1위를 포함하여 최고 성능을 기록한 횟수 및 비율입니다.

| 평가 기법 (임계값 / 손실함수 조합) | 최고 성능 달성 데이터셋 수 | 점유율 (%) |
| :--- | :---: | :---: |
| **VAE_Percentile_98** | 119 | 12.57% |
| **VAE_Skewness_Adaptive** | 216 | 22.81% |
| **VAE_EVT_GPD** | 216 | 22.81% |
| **VAE_Weibull** | 173 | 18.27% |
| **VAE_Gumbel** | 213 | 22.49% |
| **VAE_Online_Dynamic** | 85 | 8.98% |
| **STFT_VAE_EVT** | 191 | 20.17% |
| **Hybrid_VAE_EVT** | 222 | 23.44% |
| **Adv_Hybrid_VAE_EVT** | 219 | 23.13% |
| **전 기법 F1 0.0 기록 (탐지 실패)** | 130 | 13.73% |

> [!NOTE]
> 1위 수치가 중복 집계된 이유는 특정 데이터셋에서 복수의 기법(예: 왜도 적응형과 GPD-EVT)이 동일하게 최고 F1-Score를 달성했기 때문입니다.

## 2. 데이터셋별 최적 성능 분석 목록 (상위 100개 대표 예시)

전체 데이터셋 중 알파벳 순서 기준 상위 100개 데이터셋의 성능 요약표입니다.

| 데이터셋명 | 최고 F1-Score | 최고 성능 달성 기법 목록 |
| :--- | :---: | :--- |
| ACSF1_normal_0 | 1.0000 | STFT_VAE_EVT, Hybrid_VAE_EVT, Adv_Hybrid_VAE_EVT |
| ACSF1_normal_1 | 1.0000 | VAE_Weibull |
| ACSF1_normal_2 | 0.0000 | None (All 0) |
| ACSF1_normal_3 | 0.0000 | None (All 0) |
| ACSF1_normal_4 | 1.0000 | VAE_EVT_GPD, STFT_VAE_EVT, Hybrid_VAE_EVT |
| ACSF1_normal_5 | 0.0000 | None (All 0) |
| ACSF1_normal_6 | 0.6667 | VAE_EVT_GPD |
| ACSF1_normal_7 | 0.6667 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Weibull, VAE_Gumbel, STFT_VAE_EVT, Hybrid_VAE_EVT, Adv_Hybrid_VAE_EVT |
| ACSF1_normal_8 | 0.0000 | None (All 0) |
| ACSF1_normal_9 | 1.0000 | VAE_Online_Dynamic |
| Adiac_normal_1 | 0.2000 | VAE_Percentile_98 |
| Adiac_normal_10 | 0.5000 | VAE_EVT_GPD |
| Adiac_normal_11 | 0.0000 | None (All 0) |
| Adiac_normal_12 | 1.0000 | VAE_Weibull, VAE_Online_Dynamic |
| Adiac_normal_13 | 0.3333 | VAE_Percentile_98 |
| Adiac_normal_14 | 0.0000 | None (All 0) |
| Adiac_normal_15 | 0.0000 | None (All 0) |
| Adiac_normal_16 | 0.0000 | None (All 0) |
| Adiac_normal_17 | 0.2857 | VAE_Gumbel |
| Adiac_normal_18 | 0.6667 | STFT_VAE_EVT, Hybrid_VAE_EVT |
| Adiac_normal_19 | 0.4000 | Hybrid_VAE_EVT |
| Adiac_normal_2 | 1.0000 | Hybrid_VAE_EVT |
| Adiac_normal_20 | 0.0000 | None (All 0) |
| Adiac_normal_21 | 0.0000 | None (All 0) |
| Adiac_normal_22 | 0.0000 | None (All 0) |
| Adiac_normal_23 | 1.0000 | VAE_Online_Dynamic |
| Adiac_normal_24 | 0.0000 | None (All 0) |
| Adiac_normal_25 | 0.0000 | None (All 0) |
| Adiac_normal_26 | 0.5000 | VAE_EVT_GPD |
| Adiac_normal_27 | 1.0000 | VAE_EVT_GPD |
| Adiac_normal_28 | 0.6667 | VAE_Gumbel |
| Adiac_normal_29 | 0.1818 | VAE_EVT_GPD |
| Adiac_normal_3 | 0.0000 | None (All 0) |
| Adiac_normal_30 | 0.4000 | VAE_EVT_GPD |
| Adiac_normal_31 | 0.0000 | None (All 0) |
| Adiac_normal_32 | 0.0000 | None (All 0) |
| Adiac_normal_33 | 1.0000 | VAE_Percentile_98, VAE_Skewness_Adaptive |
| Adiac_normal_34 | 1.0000 | VAE_Skewness_Adaptive |
| Adiac_normal_35 | 1.0000 | VAE_Weibull, Adv_Hybrid_VAE_EVT |
| Adiac_normal_36 | 1.0000 | VAE_Weibull, VAE_Gumbel, Hybrid_VAE_EVT |
| Adiac_normal_37 | 1.0000 | VAE_EVT_GPD |
| Adiac_normal_4 | 0.2222 | VAE_EVT_GPD |
| Adiac_normal_5 | 0.6667 | Adv_Hybrid_VAE_EVT |
| Adiac_normal_6 | 1.0000 | STFT_VAE_EVT |
| Adiac_normal_7 | 1.0000 | STFT_VAE_EVT, Hybrid_VAE_EVT, Adv_Hybrid_VAE_EVT |
| Adiac_normal_8 | 1.0000 | STFT_VAE_EVT, Adv_Hybrid_VAE_EVT |
| Adiac_normal_9 | 1.0000 | STFT_VAE_EVT, Hybrid_VAE_EVT, Adv_Hybrid_VAE_EVT |
| ArrowHead_normal_0 | 0.0000 | None (All 0) |
| ArrowHead_normal_1 | 0.0000 | None (All 0) |
| ArrowHead_normal_2 | 0.0000 | None (All 0) |
| BME_normal_1 | 1.0000 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Weibull, VAE_Gumbel, Hybrid_VAE_EVT |
| BME_normal_2 | 1.0000 | VAE_Skewness_Adaptive, VAE_Weibull, Hybrid_VAE_EVT |
| BME_normal_3 | 0.6667 | VAE_EVT_GPD, VAE_Weibull |
| Beef_normal_1 | 1.0000 | STFT_VAE_EVT, Adv_Hybrid_VAE_EVT |
| Beef_normal_2 | 0.2500 | Adv_Hybrid_VAE_EVT |
| Beef_normal_3 | 1.0000 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_EVT_GPD |
| Beef_normal_4 | 0.6667 | Adv_Hybrid_VAE_EVT |
| Beef_normal_5 | 1.0000 | VAE_EVT_GPD, VAE_Gumbel, VAE_Online_Dynamic, Hybrid_VAE_EVT |
| BeetleFly_normal_1 | 0.6667 | STFT_VAE_EVT |
| BeetleFly_normal_2 | 0.6667 | STFT_VAE_EVT, Adv_Hybrid_VAE_EVT |
| BirdChicken_normal_1 | 0.5000 | VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Weibull, VAE_Gumbel |
| BirdChicken_normal_2 | 1.0000 | STFT_VAE_EVT |
| CBF_normal_1 | 0.5556 | Hybrid_VAE_EVT |
| CBF_normal_2 | 1.0000 | VAE_Gumbel |
| CBF_normal_3 | 0.8000 | VAE_Gumbel |
| Car_normal_1 | 0.3333 | VAE_Weibull |
| Car_normal_2 | 1.0000 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Gumbel |
| Car_normal_3 | 0.6667 | VAE_Weibull |
| Car_normal_4 | 0.6667 | VAE_Weibull |
| Chinatown_normal_1 | 0.0000 | None (All 0) |
| Chinatown_normal_2 | 0.3333 | VAE_Gumbel, Adv_Hybrid_VAE_EVT |
| ChlorineConcentration_normal_1 | 0.0000 | None (All 0) |
| ChlorineConcentration_normal_2 | 0.0952 | VAE_Gumbel |
| ChlorineConcentration_normal_3 | 0.1020 | STFT_VAE_EVT |
| CinCECGTorso_normal_1 | 0.6000 | VAE_EVT_GPD |
| CinCECGTorso_normal_2 | 0.9231 | VAE_Gumbel |
| CinCECGTorso_normal_3 | 1.0000 | VAE_Gumbel |
| CinCECGTorso_normal_4 | 0.9231 | VAE_Gumbel |
| Coffee_normal_0 | 0.0000 | None (All 0) |
| Coffee_normal_1 | 0.0000 | None (All 0) |
| Computers_normal_1 | 0.0000 | None (All 0) |
| Computers_normal_2 | 0.0000 | None (All 0) |
| CricketX_normal_1 | 1.0000 | VAE_EVT_GPD, Hybrid_VAE_EVT |
| CricketX_normal_10 | 1.0000 | Adv_Hybrid_VAE_EVT |
| CricketX_normal_11 | 0.0000 | None (All 0) |
| CricketX_normal_12 | 0.3333 | VAE_EVT_GPD |
| CricketX_normal_2 | 0.5000 | VAE_Weibull, VAE_Gumbel, STFT_VAE_EVT |
| CricketX_normal_3 | 0.2000 | VAE_Gumbel |
| CricketX_normal_4 | 0.6667 | STFT_VAE_EVT |
| CricketX_normal_5 | 0.0000 | None (All 0) |
| CricketX_normal_6 | 0.5000 | VAE_EVT_GPD |
| CricketX_normal_7 | 1.0000 | VAE_Skewness_Adaptive |
| CricketX_normal_8 | 0.5000 | VAE_Percentile_98, VAE_EVT_GPD |
| CricketX_normal_9 | 0.0000 | None (All 0) |
| CricketY_normal_1 | 1.0000 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Weibull, Hybrid_VAE_EVT |
| CricketY_normal_10 | 0.6667 | VAE_Percentile_98 |
| CricketY_normal_11 | 1.0000 | VAE_Percentile_98 |
| CricketY_normal_12 | 0.0000 | None (All 0) |
| CricketY_normal_2 | 1.0000 | VAE_Skewness_Adaptive, VAE_EVT_GPD, VAE_Weibull, VAE_Gumbel, Hybrid_VAE_EVT |
| CricketY_normal_3 | 0.4000 | VAE_Percentile_98, VAE_Skewness_Adaptive, VAE_Gumbel |

*(나머지 847개 데이터셋의 개별 매핑 결과는 메모리 및 파일에 안전하게 취합 및 적재되었습니다.)*
