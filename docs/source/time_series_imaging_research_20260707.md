# Time-Series Imaging Research - 2026-07-07

`model_hard` 데이터셋 대응책 중 하나로, 1D 시계열을 2D 이미지 또는 이미지형 행렬로 변환한 뒤 CNN/ViT/autoencoder/거리 기반 모델을 적용하는 방향을 조사했다.

## 핵심 결론

시계열 이미지화는 threshold 문제를 직접 해결하는 방법이 아니다. 대신 현재 score representation이 약한 데이터셋에서 새로운 표현 공간을 만든다.

특히 다음 경우에 가치가 있다.

- 시간점 간 관계가 중요한 경우: GAF, MTF, RP
- 반복/주기/상태 전이가 중요한 경우: MTF, RP, multi-scale RP
- 주파수/노이즈/스펙트럼 구조가 중요한 경우: spectrogram, scalogram
- 이미지/윤곽에서 유래한 시계열인 경우: RP, GAF, line-image, shapelet image feature
- 운영 시스템에서 설명 가능성이 필요한 경우: anomalous image patch, recurrence block, frequency band로 근거를 보여줄 수 있음

주의할 점도 명확하다.

- 길이 `L` 시계열을 GAF/RP로 바꾸면 기본적으로 `L x L` 이미지가 되어 비용이 커진다.
- 이미지 autoencoder를 너무 강하게 만들면 anomaly까지 잘 복원해 anomaly score가 낮아질 수 있다.
- GAF/RP는 scaling, window size, downsampling에 민감하다.
- pretrained vision model은 빠르게 실험할 수 있지만, 시계열 의미를 자동으로 이해한다고 보장하면 안 된다.

## 주요 변환 방법

### 1. Gramian Angular Field, GAF

GAF는 시계열을 `[-1, 1]` 범위로 정규화한 뒤 polar coordinate로 바꾸고, 시간점 쌍 `(i, j)`의 각도 관계를 행렬로 만든다. 보통 `GASF`와 `GADF` 두 종류가 있다.

Source: https://arxiv.org/abs/1506.00327  
Implementation reference: https://pyts.readthedocs.io/en/stable/modules/image.html

우리에게 맞는 해석:

- 전체 모양과 시간점 간 상관 구조를 이미지로 만든다.
- `HandOutlines`, `Phalanx*`, `Fish`, `Worms`처럼 shape가 중요한 데이터에 먼저 테스트할 가치가 있다.
- `EthanolLevel`처럼 길이가 긴 데이터는 바로 `1751 x 1751`로 만들면 비싸므로 downsampling/windowing이 필요하다.
- GASF 또는 GADF 단독보다 `GASF + GADF` 2채널이 안전하다. PHM 2023 논문은 두 방식이 서로 놓치는 구조가 있어 channel direction으로 겹쳐 쓰는 개선을 제안했다.

### 2. Markov Transition Field, MTF

MTF는 값을 bin으로 나눈 뒤, 상태 간 전이 확률을 시간 위치 위에 펼친 이미지다.

Source: https://arxiv.org/abs/1506.00327  
Implementation reference: https://pyts.readthedocs.io/en/stable/modules/image.html

우리에게 맞는 해석:

- 값의 순서 자체보다 상태 전이 패턴이 중요할 때 유리하다.
- `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `Computers`, `RefrigerationDevices`처럼 장비/전력 사용 상태가 바뀌는 데이터에 맞다.
- 정상 설비에서는 recipe별 상태 전이가 안정적이라는 사용자 가정과도 잘 맞는다.

### 3. Recurrence Plot, RP

RP는 time-delay embedding을 만든 뒤, 궤적 간 거리를 이미지로 표현한다. 반복성, 주기성, 동역학 구조를 볼 수 있다.

Source: https://pyts.readthedocs.io/en/stable/modules/image.html  
Multi-scale signed RP source: https://www.sciencedirect.com/science/article/pii/S0031320321005653

우리에게 맞는 해석:

- 반복 패턴, 주기, 유사 구간의 재등장이 중요한 데이터에 적합하다.
- `UWaveGesture*`, `GestureMidAir*`, `MelbournePedestrian`, `ScreenType`, `ElectricDevices`에 후보가 된다.
- 기본 RP는 scale/length variability와 trend 방향 혼동 문제가 보고되어 있다. 그래서 `multi-scale signed RP` 또는 trend/diff channel을 같이 쓰는 것이 더 낫다.

### 4. Spectrogram

Spectrogram은 STFT로 시간-주파수 에너지를 이미지화한다. window size에 따라 시간 해상도와 주파수 해상도 사이 tradeoff가 생긴다.

Source: https://arxiv.org/html/2403.03611v4

우리에게 맞는 해석:

- `FordA/B`, `EthanolLevel`, `Phoneme`, `Earthquakes`처럼 주파수 성격이 있는 데이터에 좋다.
- 계산 비용이 CWT보다 낮아 첫 실험으로 적합하다.
- 운영 설명에서는 “어느 시간대, 어느 주파수 band가 정상과 달랐다”로 설명할 수 있다.

### 5. Scalogram, CWT

Scalogram은 wavelet transform 기반의 시간-주파수 이미지다. STFT보다 multiresolution 성격이 강해서, 고주파 transient와 저주파 trend를 동시에 보기 좋다.

Source: https://arxiv.org/html/2403.03611v4

우리에게 맞는 해석:

- `EthanolLevel`, `FordA/B`, `Earthquakes`, `ScreenType`처럼 느린 변화와 짧은 이벤트가 섞인 경우에 후보가 된다.
- STFT보다 무겁기 때문에 hard subset 또는 family-specific 실험으로 제한하는 편이 좋다.

### 6. 단순 line image / segment image / pretrained ViT

최근 연구는 시계열을 line plot, heatmap, GAF, RP 같은 이미지로 만들어 pretrained vision transformer에 넣는 방향도 검토한다.

Source: https://arxiv.org/html/2506.08641v2

우리에게 맞는 해석:

- 학습 데이터가 적을 때 pretrained image encoder embedding을 가져와 KNN/centroid distance를 만들 수 있다.
- 단, line plot은 축/스케일/렌더링 설정에 민감하다. 실무 운영용 기본 방법으로 바로 채택하기보다, 빠른 후보 실험으로 보는 것이 맞다.

## `model_hard` family별 추천

| Family | 대표 데이터셋 | 우선 이미지화 |
|---|---|---|
| Device / power usage | `ScreenType`, `LargeKitchenAppliances`, `ElectricDevices`, `Computers` | MTF, RP, spectrogram |
| Spectral / chemical | `EthanolLevel` | spectrogram, scalogram, GAF downsample |
| Outline / image-derived shape | `HandOutlines`, `Phalanx*`, `Fish`, `Worms` | GAF, RP, line image |
| Sensor / noisy machinery | `FordA`, `FordB`, `Earthquakes` | spectrogram, scalogram, RP |
| Motion / gesture | `UWaveGesture*`, `GestureMidAir*`, `MelbournePedestrian` | RP, multi-scale RP, GAF |

## 운영 시스템 관점의 anomaly score 설계

### 방법 A: 이미지 변환 + flatten feature + KNN/robust centroid

- train-normal만 이미지로 변환한다.
- 이미지를 downsample한 뒤 flatten하거나 PCA한다.
- test 이미지와 train-normal 이미지 feature 사이의 KNN distance를 anomaly score로 쓴다.

장점:

- 빠르다.
- 딥러닝 학습이 필요 없다.
- 기존 KNN threshold 정책과 연결하기 쉽다.

단점:

- 이미지 크기가 커지면 feature 차원이 커진다.
- family별 transform 선택이 중요하다.

### 방법 B: 이미지 변환 + CNN embedding + KNN

- GAF/MTF/RP/spectrogram 이미지를 작은 CNN에 넣어 embedding을 만든다.
- contrastive 또는 autoencoder 방식으로 train-normal 구조를 학습한다.
- embedding distance를 anomaly score로 쓴다.

장점:

- raw image flatten보다 안정적일 가능성이 높다.
- 대시보드에서 anomalous patch를 설명하기 쉽다.

단점:

- 학습 비용과 seed variance가 생긴다.
- train 수가 작은 데이터셋에서는 과적합 가능성이 있다.

### 방법 C: 이미지 autoencoder reconstruction

- train-normal 이미지로 autoencoder를 학습한다.
- test image reconstruction error를 anomaly score로 쓴다.

장점:

- normal-only 운영 방식과 잘 맞다.
- 시각적으로 설명하기 쉽다.

주의:

- PHM 2023 논문은 autoencoder reconstruction accuracy를 무조건 높이는 것이 anomaly detection에는 오히려 해로울 수 있음을 보고했다.
- 따라서 너무 강한 U-Net보다 작은 bottleneck 모델, SSIM/L1 조합, train-normal 기반 threshold가 필요하다.

## 추천 실험

### 실험 I1: Image-KNN Smoke Test

- 대상: `model_hard` 핵심 32개
- 변환: GASF, GADF, MTF, RP
- 이미지 크기: 64 또는 96
- score: flattened image PCA 32/64 + KNN distance
- 목적: 이미지화 자체가 oracle AUC-PR을 올리는지 빠르게 확인

### 실험 I2: Family-Specific Imaging

- power/device: MTF + RP
- spectral/sensor: spectrogram + scalogram
- outline/shape: GAF + RP
- score: robust centroid + KNN rank ensemble
- 목적: family별로 맞는 변환이 다른지 확인

### 실험 I3: GASF/GADF 2-channel Autoencoder

- 대상: train size가 충분한 hard subset
- 변환: GASF + GADF 2채널
- 모델: 작은 convolutional autoencoder
- score: L1 + SSIM residual
- 목적: normal-only 운영 시스템에서 시각적 근거를 함께 제공할 수 있는지 확인

### 실험 I4: Spectrogram/Scalogram Score

- 대상: `EthanolLevel`, `FordA/B`, `Earthquakes`, `Phoneme`
- 변환: STFT spectrogram 우선, 이후 CWT scalogram
- score: PCA/KNN 또는 small CNN embedding
- 목적: 주파수 기반 hard dataset 개선 가능성 확인

### 실험 I5: RP + Trend/Diff Channel

- 대상: `ScreenType`, `ElectricDevices`, `UWaveGesture*`, `GestureMidAir*`
- 변환: RP + signed RP 또는 RP + first-difference RP
- 목적: 기본 RP의 trend confusion을 줄이는지 확인

## 우선순위 제안

가장 먼저 할 만한 것은 `I1 Image-KNN Smoke Test`다. 이유는 빠르고, 기존 pipeline의 KNN/threshold 평가 구조를 거의 그대로 쓸 수 있기 때문이다.

그 다음은 `I2 Family-Specific Imaging`이다. image transform은 데이터 성격을 많이 타므로, 전체 평균 하나로 판단하면 좋은 후보를 버릴 위험이 있다.

딥러닝 autoencoder는 바로 전체 실행하지 않는 편이 좋다. 먼저 이미지 변환 자체가 AUC-PR/oracle F1을 올리는지 본 뒤, 좋아지는 family에만 CNN/AE를 붙이는 것이 안전하다.

## 참고 자료

- Wang and Oates, Imaging Time-Series to Improve Classification and Imputation: https://arxiv.org/abs/1506.00327
- pyts imaging time series documentation: https://pyts.readthedocs.io/en/stable/modules/image.html
- Multi-scale signed recurrence plot based TSC: https://www.sciencedirect.com/science/article/pii/S0031320321005653
- Time-series image encoding for anomaly detection with GAF and AE: https://papers.phmsociety.org/index.php/phmap/article/download/3760/2226
- Spectrogram vs scalogram discussion and workflow: https://arxiv.org/html/2403.03611v4
- Time-series representations in pretrained vision transformers: https://arxiv.org/html/2506.08641v2
