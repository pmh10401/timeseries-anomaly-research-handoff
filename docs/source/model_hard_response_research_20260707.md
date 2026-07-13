# Model-Hard Dataset Response Research - 2026-07-07

현재 `model_hard`로 분류된 218개 데이터셋, 특히 original repeated-normal과 clean-balanced 양쪽에서 모두 어려운 32개 핵심 후보에 대한 대응책 조사 기록이다.

## 현재 문제 요약

가장 강하게 어려운 데이터셋은 다음 계열에 몰려 있다.

- Device / power usage: `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `Computers`, `RefrigerationDevices`
- Spectral / chemical: `EthanolLevel`
- Image outline / shape: `HandOutlines`, `Distal/Middle/ProximalPhalanx*`, `PhalangesOutlinesCorrect`
- Sensor / noisy machinery: `FordA`, `FordB`, `Earthquakes`
- Motion / variable length: `GestureMidAir*`, `UWaveGestureLibrary*`, `MelbournePedestrian`

공통 결론은 threshold보다 score representation 문제라는 점이다. 완료 실험에서 oracle F1/AUC-PR 자체가 낮은 데이터셋은 threshold를 바꿔도 회복 여지가 작다.

## 데이터셋별 단서

- `ScreenType`과 `LargeKitchenAppliances`는 Powering the Nation 전력 사용 데이터이며 24시간 전력 소비 패턴이다. 위치/구간/주파수/저주파 trend가 중요할 가능성이 높다. Source: https://www.timeseriesclassification.com/description.php?Dataset=ScreenType, https://www.timeseriesclassification.com/description.php?Dataset=LargeKitchenAppliances
- `ElectricDevices`도 같은 Powering the Nation 계열이다. Explanation Space 논문에서도 ElectricDevices는 zero/baseline과 explanation space 문제가 언급된다. Source: https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices, https://arxiv.org/abs/2409.01354
- `EthanolLevel`은 spectrograph 기반의 forged spirits / ethanol level 문제다. 전체 주파수대와 스펙트럼 형태가 중요하므로 단순 time-domain KNN/ROCKET score가 약할 수 있다. Source: https://www.timeseriesclassification.com/description.php?Dataset=EthanolLevel
- `HandOutlines`와 Phalanx 계열은 이미지에서 추출한 outline correctness / bone age 관련 문제다. 짧은 국소 shape와 alignment, shapelet/distance 계열이 유리할 수 있다. Source: https://www.timeseriesclassification.com/description.php?Dataset=HandOutlines, https://www.timeseriesclassification.com/description.php?Dataset=DistalPhalanxOutlineCorrect
- `Earthquakes`는 segmentation된 지진 전조/비전조 sensor 문제이고, 양성 사건 정의가 물리적 조건과 과거 512시간 문맥에 의존한다. Source: https://www.timeseriesclassification.com/description.php?Dataset=Earthquakes
- `FordA/FordB`는 engine noise sensor 문제다. FordB는 test data가 noisy condition에서 수집되어 domain shift가 명시되어 있다. Source: https://www.timeseriesclassification.com/description.php?Dataset=FordA, https://www.timeseriesclassification.com/description.php?Dataset=FordB

## 조사한 기법과 의미

### 1. HIVE-COTE 2.0 계열을 분해해서 가져오기

HIVE-COTE 2.0은 shapelets, dictionary, interval, ROCKET ensemble을 섞는 heterogeneous meta-ensemble이다. UCR/UEA에서 강한 이유는 하나의 표현 공간에 기대지 않기 때문이다. Source: https://arxiv.org/abs/2104.07551

우리에게 맞는 해석:

- 전체 HC2를 그대로 돌리기보다, component 아이디어를 one-class score로 바꿔 가져온다.
- `model_hard` 32개에는 ROCKET 단독 점수보다 `shapelet + interval + dictionary + ROCKET` score ensemble이 필요하다.

### 2. MultiROCKET / HYDRA

MultiROCKET은 raw input뿐 아니라 first-order difference를 쓰고, 여러 pooling operator를 추가해 feature diversity를 늘린다. HYDRA는 ROCKET과 dictionary method의 중간 형태로, random convolutional kernel match count를 사용한다. Source: https://arxiv.org/abs/2102.00457, https://arxiv.org/abs/2203.13652

우리에게 맞는 해석:

- 기존 ROCKET/KNN이 실패한 데이터셋은 PPV/거리 하나로 충분하지 않을 수 있다.
- MultiROCKET의 difference/pooling, HYDRA의 pattern-count 성격을 normal-only distance score로 변환하는 실험이 타당하다.
- 특히 `UWaveGesture*`, `GestureMidAir*`, `FordA/B`, `Phoneme`처럼 국소 패턴/변화율이 중요한 계열에 우선 적용한다.

### 3. DrCIF / rSTSF / Quant: interval + distribution feature

DrCIF는 원시 series, first difference, periodogram에서 random interval을 뽑고 통계/catch22 feature를 사용한다. Quant는 interval 내부 값의 quantile을 핵심 feature로 쓰며 UCR 142개에서 빠르고 강한 interval 방법으로 제안되었다. Source: https://arxiv.org/abs/2008.09172, https://link.springer.com/article/10.1007/s10618-024-01036-9

우리에게 맞는 해석:

- Device/power 계열은 특정 시간 구간의 분포, 저주파 trend, periodogram이 중요하다.
- `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `RefrigerationDevices`, `Computers`에는 interval quantile / periodogram / difference feature를 붙인다.
- 이 계열은 “어느 구간에서 소비 패턴이 다르냐”가 핵심일 수 있으므로 global ROCKET rank보다 interval score가 더 설명 가능하다.

### 4. CEEMD + MultiROCKET

CEEMD-MultiRocket은 원시 시계열을 고/중/저주파 성분으로 분해한 뒤 MultiROCKET에 넣는 접근이다. 논문은 ScreenType 예시로 CEEMD decomposition을 다룬다. Source: https://www.mdpi.com/2079-9292/12/5/1188

우리에게 맞는 해석:

- `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `EthanolLevel`에 특히 적합하다.
- 다만 CEEMD는 계산 비용이 커질 수 있으므로, 우선은 FFT/STFT/wavelet/low-pass decomposition으로 간소화한 버전을 먼저 실험한다.

### 5. Shapelet-based anomaly detection

Unsupervised shapelet anomaly detection은 정상 class를 설명하는 shapelet feature를 학습하고, 정상 feature space를 compact boundary로 묶는 방식이다. Source: https://link.springer.com/article/10.1007/s00180-018-0824-9

우리에게 맞는 해석:

- `HandOutlines`, `Phalanx*`, `ArrowHead`, `Fish`, `Worms`처럼 국소 shape 차이가 중요한 계열에 적합하다.
- normal-only 운영 시스템과도 잘 맞는다. 정상 shapelet prototype에서 멀어지는 정도를 anomaly score로 쓸 수 있다.

### 6. CARLA / anomaly injection contrastive learning

CARLA는 time series anomaly detection을 위한 self-supervised contrastive representation learning으로, anomaly injection을 negative sample로 사용한다. 정상 boundary가 너무 빡빡해 false positive가 늘어나는 문제도 명시적으로 다룬다. Source: https://arxiv.org/html/2308.09296v3

우리에게 맞는 해석:

- 사용자의 운영 목표인 “명확한 이상치만 우선 검토”와 잘 맞는다.
- 현재 Multi-Aug VAE/InfoNCE에서 문제가 됐던 “negative가 실제 이상 경계를 대표하지 못함”을, generic anomaly injection으로 보완할 수 있다.
- `FordA/B`, `Earthquakes`, `HandOutlines`, `EthanolLevel`처럼 정상과 이상의 경계가 현재 score 공간에서 섞이는 계열에 적용한다.

### 7. Time-series imaging

시계열을 GAF, MTF, RP, spectrogram, scalogram 같은 2D 이미지형 표현으로 바꾸는 방법도 hard dataset 대응책으로 가치가 있다. 이 접근은 threshold를 직접 고치는 것이 아니라, 정상/이상 차이가 더 잘 드러나는 새로운 score representation을 만드는 방향이다. 별도 조사 기록: `docs/time_series_imaging_research_20260707.md`

우리에게 맞는 해석:

- `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices` 같은 power/device 계열은 MTF/RP/spectrogram이 후보가 된다.
- `EthanolLevel`, `FordA/B`, `Earthquakes`는 spectrogram/scalogram이 후보가 된다.
- `HandOutlines`, `Phalanx*`처럼 image-derived shape 계열은 GAF/RP/line image가 후보가 된다.
- 우선은 deep CNN보다 `image transform + PCA/KNN` smoke test가 안전하다.

## 우선 실험 제안

### 실험 A: Model-Hard Subset Diagnostic Benchmark

- 대상: original+clean 양쪽에서 모두 어려운 32개 dataset
- 목적: 전체 평균이 아니라 hard subset에서 무엇이 실제로 좋아지는지 확인
- metric: AUC-PR, oracle F1, best rank hit, F1, mean FP

### 실험 B: Interval-Quantile / DrCIF-lite Score

- 대상: `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `Computers`, `RefrigerationDevices`
- feature: raw/diff/periodogram의 fixed interval quantiles, slope, IQR, min/max
- score: train-normal KNN 또는 robust centroid distance
- 기대: power/device 계열의 구간별 소비 패턴 차이 회복

### 실험 C: Frequency-Decomposition ROCKET

- 대상: `ScreenType`, `EthanolLevel`, `ElectricDevices`, `FordA/B`
- feature: raw + diff + FFT band energy + wavelet/STFT summaries + ROCKET/MultiROCKET
- 기대: spectrograph/전력/engine noise처럼 주파수 성격이 강한 계열 보강

### 실험 D: Shapelet-Normal Prototype Score

- 대상: `HandOutlines`, `Phalanx*`, `ArrowHead`, `Fish`, `Worms`
- feature: random shapelet 또는 RDST-style dilated shapelet distance
- score: 정상 shapelet prototype distance, one-class SVDD-style compactness
- 기대: outline/shape 계열에서 국소 형태 차이 회복

### 실험 E: HYDRA Pattern Count Score

- 대상: `GestureMidAir*`, `UWaveGesture*`, `Phoneme`, `FordA/B`
- feature: competing convolutional kernels의 pattern count
- score: count histogram distance / KNN
- 기대: ROCKET max/PPV에서 놓친 반복 pattern 차이 회복

### 실험 F: CARLA-style Anomaly Injection Contrastive

- 대상: hard 32개 중 train size가 충분한 dataset부터
- synthetic negative: spike, segment scaling, local drift, frequency perturbation, time-warp, segment replacement
- score: embedding KNN + train-normal compactness
- 기대: normal-only 상황에서도 anomaly boundary를 더 명확히 학습

### 실험 G: Image-KNN Smoke Test

- 대상: original+clean 양쪽에서 모두 어려운 32개 dataset
- 변환: GASF, GADF, MTF, RP, spectrogram
- feature: 64 또는 96 크기 이미지로 축소한 뒤 PCA 32/64
- score: train-normal KNN 또는 robust centroid distance
- 기대: 이미지화된 pairwise/time-frequency structure가 oracle AUC-PR을 올리는지 빠르게 확인

## 추천 순서

1. 실험 A로 hard subset 전용 평가 harness를 만든다.
2. 가장 빠른 B와 C를 먼저 실행한다. 둘 다 feature engineering 중심이라 큐 부담이 상대적으로 작다.
3. D를 image/outline family에만 제한해 실행한다.
4. E를 motion/sensor family에 실행한다.
5. G로 이미지화 smoke test를 병렬 후보가 아니라 별도 순차 실험으로 추가한다.
6. F는 비용이 크므로 hard 32개 중 train size가 충분한 subset에서만 시작한다.

이 순서가 좋은 이유는, 먼저 빠른 classical/feature-space 실험으로 어떤 표현이 family별로 맞는지 확인한 뒤, 딥러닝 contrastive 실험을 좁은 타겟에만 쓰기 위해서다.
