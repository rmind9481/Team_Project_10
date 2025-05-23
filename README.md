🎯 주제 선정 이유
기존의 선형 모델이 이미지의 공간적 구조를 학습하는 데 한계가 있음을 인식하였습니다.

CNN 구조를 활용하여 이미지의 위치, 형태, 색상 간의 비선형 관계를 효과적으로 학습하고자 하였습니다.

🔍 데이터 수집 및 전처리
데이터 수집: 초기 800장의 이미지를 수집하였으며, 이후 데이터 증강을 통해 3,000장으로 확장하였습니다.

클래스 수: 초기 4개에서 25개로 증가시켰습니다.

데이터 증강 기법: 회전, 반전, 밝기 조절 등 다양한 방식 적용

🛠 사용 기술 및 라이브러리
프로그래밍 언어: Python

딥러닝 프레임워크: pytorch

데이터 처리: NumPy, Pandas

🧪 모델 구조 및 하이퍼파라미터 실험
1. 계층 수 변화에 따른 성능 분석
계층 수	정확도	해석
2계층	59%	기준 모델, 매우 낮은 성능
3계층	73.2%	성능의 명확한 향상, 중요한 변화 지점
5계층	87.5%	최적 성능, 가장 효과적인 계층 수
7계층	86.5%	성능 소폭 하락, 과적합 및 학습 시간 증가 우려

2. 활성화 함수 실험
Sigmoid vs ReLU: ReLU가 정확도와 학습 속도에서 우수한 성능을 보였습니다.

3. 풀링 방식 실험
MaxPooling > AvgPooling > 없음: MaxPooling 사용 시 정확도 86.9%, 학습 시간 8분

4. 하이퍼파라미터 실험
Dropout: 0.3~0.5 중 0.3에서 가장 안정적인 성능

Epoch: 20~30 중 과도한 epoch는 오히려 성능 저하

Batch size: 16, 32, 64 중 64에서 시간 대비 효율 양호

Learning rate: 0.0001~0.001 중 0.0001이 가장 안정적인 성능

🔄 ResNet 구조 적용
기존 CNN 정확도: 60%

ResNet 정확도: 82.5%

확장된 데이터셋(3,000장, 25종 클래스)에서의 테스트 정확도: 96.68%

특징: Skip Connection을 통해 기울기 소실 문제 극복 및 병렬 연산 가능


프로젝트 구조
<pre>
'''
project/
├── data/
│   ├── raw_images/
│   └── augmented_images/
├── notebooks/
│   ├── cnn_baseline.ipynb
│   └── resnet_experiments.ipynb
├── models/
│   ├── cnn_model.h5
│   └── resnet_model.h5
├── images/
│   └── accuracy_comparison.png
└── README.md
'''
</pre>

📈 주요 결과
계층 수를 5로 설정하고 ReLU 활성화 함수와 MaxPooling을 적용한 모델이 가장 우수한 성능을 보였습니다.

ResNet 구조를 적용함으로써 테스트 정확도를 96.68%까지 향상시킬 수 있었습니다.

📌 결론 및 시사점
CNN 구조의 적절한 설계와 하이퍼파라미터 튜닝을 통해 이미지 분류 모델의 성능을 크게 향상시킬 수 있습니다.

ResNet과 같은 고급 구조를 활용하면 더 복잡한 데이터셋에서도 높은 정확도를 달성할 수 있습니다.
