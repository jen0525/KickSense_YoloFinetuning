# KickSense YOLO Training

## ⚽️ 축구공 탐지를 위한 YOLO 모델 파인튜닝 프로젝트

이 저장소는 KickSense 프로젝트의 축구공 탐지 성능을 향상시키기 위해 YOLO 모델을 커스텀 데이터셋으로 파인튜닝하는 코드를 포함합니다.

## 📁 프로젝트 구조

```
KickSense_yolo-training/
├── README.md                    # 프로젝트 설명서
├── fine_tuning.ipynb           # 🧪 YOLO 파인튜닝 메인 노트북
├── data.yaml                   # 📊 YOLO 데이터셋 설정 파일
├── ball_yolom.pt              # 🎯 훈련된 YOLO Medium 모델
├── ball_yolos.pt              # 🎯 훈련된 YOLO Small 모델
├── train/                      # 📂 훈련 데이터셋
│   ├── images/                # 훈련용 이미지들
│   ├── labels/                # YOLO 형식 라벨 파일들
│   └── labels.cache          # 라벨 캐시 파일
├── val/                       # 📂 검증 데이터셋
│   ├── images/                # 검증용 이미지들
│   └── labels/                # 검증용 라벨 파일들
└── runs/                      # 🏃‍♂️ 훈련 결과 및 로그
```

## 🎯 프로젝트 목표

- **축구공 탐지 정확도 향상**: 다양한 촬영 환경에서의 축구공 탐지 성능 최적화
- **실시간 처리**: 모바일 및 웹 환경에서 실시간 분석이 가능한 경량화된 모델 개발
- **강건성 확보**: 다양한 조명, 각도, 배경에서도 안정적인 탐지 성능 보장

## 📊 데이터셋 정보

### 데이터셋 구성
- **클래스**: 1개 (축구공)
- **훈련 데이터**: 9,615개 이미지
- **검증 데이터**: 2,686개 이미지
- **라벨 형식**: bounding box 

### 데이터셋 특징
- 다양한 촬영 각도의 축구공 이미지
- 실제 킥 동작 중 캡처된 프레임들
- 다양한 조명 조건 및 배경 환경

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 의존성 설치
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
```

### 2. 훈련 실행

```python
# fine_tuning.ipynb 노트북 실행
# 또는 명령어로 직접 훈련
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8m.pt')  # YOLOv8 Medium 사용

# 훈련 실행
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=960,
    batch=6,
    lr0=0.0005,
    name='ball_detection'
)
```

### 3. 모델 평가

```python
# 훈련된 모델로 평가
model = YOLO('ball_yolom.pt')
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 📈 모델 성능

### ball_yolos.pt (YOLOv8s)
- **mAP@0.5**: 0.95
- **mAP@0.5:0.95**: 0.95
- **추론 속도**: ~15ms (GPU 기준)
- **모델 크기**: ~25MB
- **Best Epoch**: 훈련 과정에서 최적 성능 달성

### ball_yolom.pt (YOLOv8m)
- **추론 속도**: ~20ms (GPU 기준)
- **모델 크기**: ~50MB
- **안정적인 성능**: 다양한 환경에서 일관된 탐지 성능

### 평가 지표 요약
- **Precision**: 높은 정밀도로 false positive 최소화
- **Recall**: 뛰어난 재현율로 축구공 놓치는 경우 최소화
- **F1-score**: Precision과 Recall의 조화평균으로 전체적인 성능 평가

## 🔧 훈련 설정

### 하이퍼파라미터
```yaml
epochs: 50
batch_size: 6
learning_rate: 0.0005
image_size: 960
optimizer: AdamW
augmentation: true
```

### 모델 아키텍처
- **YOLOv8s**: 빠른 추론 속도를 위한 Small 버전
- **YOLOv8m**: 정확도와 속도의 균형을 맞춘 Medium 버전

### 데이터 증강
- 회전, 스케일링, 크롭
- 밝기 및 대비 조정
- 노이즈 추가
- 좌우 반전

## 📁 파일 설명

### `fine_tuning.ipynb`
- YOLO 모델 훈련의 전체 과정을 담은 주요 노트북
- 데이터 전처리부터 모델 평가까지의 완전한 파이프라인
- 훈련 과정 시각화 및 결과 분석

### `data.yaml`
- YOLO 훈련을 위한 데이터셋 설정 파일
- 훈련/검증 데이터 경로 및 클래스 정보 정의

### `ball_yolom.pt` / `ball_yolos.pt`
- 훈련 완료된 YOLO 모델 파일들
- KickSense 백엔드 서버에서 축구공 탐지에 사용

## 🔗 관련 프로젝트

- **[KickSense_SoccerKickAnalyzer](https://github.com/jen0525/KickSense_SoccerKickAnalyzer)**: 메인 백엔드 서버
- **KickSense_Flutter**: 모바일 앱 (예정)

## 📊 훈련 결과 분석

### 데이터셋 통계
- **총 훈련 이미지**: 9,615장
- **총 검증 이미지**: 2,686장
- **이미지 해상도**: 960x960 픽셀
- **배치 크기**: 6

### 손실 함수 변화
- Box Loss: 바운딩 박스 위치 정확도
- Class Loss: 분류 정확도
- DFL Loss: Distribution Focal Loss

### 최종 성능 지표
- **mAP@0.5**: 0.95 (95% 정확도)
- **mAP@0.5:0.95**: 0.95 (다양한 IoU 임계값에서 일관된 성능)
- **학습률**: 0.0005로 안정적 수렴
- **에포크**: 50회 훈련으로 최적 성능 달성

## 🛠️ 개발 환경

- **Framework**: Ultralytics YOLOv8
- **Python**: 3.8+
- **GPU**: CUDA 지원 권장
- **메모리**: 최소 8GB RAM

## 📝 사용 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여 방법

1. Fork 후 개선사항 작업
2. Pull Request 생성
3. 코드 리뷰 후 병합

## 📞 문의
- 이윤환 
- 이메일: lyh516@naver.com
- 프로젝트 관련 문의는 Issues 탭 활용

---

**⚽️ KickSense Team - AI로 축구 실력 향상을 돕습니다**