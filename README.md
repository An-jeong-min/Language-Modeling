# Language Modeling
### 인공신경망과 딥러닝 HW3
### 기계정보공학과 24510091 안정민
#

### 과제 설명
- Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델 구축
- 구축할 언어 모델은 "many-to-many" 형태의 순환 신경망

### 요구 사항
1. 데이터 제공 파이프라인을 직접 작성하세요. `dataset.py` 템플릿에 코드를 작성하세요.
2. `model.py`에서 vanilla RNN과 LSTM 모델을 구현하세요. 모델 성능을 향상시키기 위해 레이어를 쌓는 것도 가능합니다.
3. `main.py`에서 모델을 학습시키는 코드를 작성하세요. 학습 과정 중 훈련 및 검증 데이터셋의 평균 손실 값을 모니터링해야 합니다.
4. (보고서) 훈련 및 검증 데이터셋에 대한 평균 손실 값을 그래프로 나타내세요. 검증 데이터셋에 대한 손실 값으로 vanilla RNN과 LSTM의 언어 생성 성능을 비교하세요.
5. `generate.py`를 작성하여 학습된 모델로 문자를 생성하세요. 가장 좋은 검증 성능을 보이는 모델을 선택하세요. 다른 시작 문자를 사용하여 각각 최소 100자의 5개 샘플을 생성하세요.
6. (보고서) 소프트맥스 함수의 온도 매개변수 *T*는 다음과 같이 작성할 수 있습니다:
    ```
    y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}  
    ```
   문자를 생성할 때 다른 온도를 시도해보고, 온도가 어떤 차이를 만드는지 그리고 향상된 결과를 생성하는데 왜 도움이 되는지 논의하세요.

### 파일 설명
#### dataset.py
이 코드는 Shakespeare 데이터셋을 불러와 문자 사전을 생성하고, 이를 인덱스로 변환한 후 시퀀스 길이 30의 입력-타겟 쌍으로 데이터를 나누어 torch.utils.data.Dataset 클래스를 상속받아 데이터셋을 정의합니다. 각 시퀀스는 입력과 타겟으로 구성되며, 모델 학습에 사용됩니다. __len__과 __getitem__ 메소드를 구현하여 데이터셋 크기와 인덱스에 따른 데이터를 반환합니다.
#### model.py
이 코드는 `CharRNN`과 `CharLSTM` 두 가지 문자 기반 신경망 모델을 정의합니다. `CharRNN` 클래스는 간단한 순환 신경망(RNN)을 구현하며, `CharLSTM` 클래스는 장단기 메모리 네트워크(LSTM)를 구현합니다. 각 모델은 입력 시퀀스를 임베딩하여 순환 레이어(RNN 또는 LSTM)를 통과시킨 후, 출력 레이어에서 예측 결과를 생성합니다. `init_hidden` 메소드는 주어진 배치 크기와 디바이스에 맞게 초기 은닉 상태를 생성합니다.
#### main.py
이 코드는 셰익스피어 데이터셋을 이용하여 문자 수준의 언어 모델을 훈련하는 스크립트입니다. `train` 함수는 모델을 훈련하고, `validate` 함수는 검증 데이터로 모델의 성능을 평가합니다. `main` 함수는 데이터셋을 로드하고, 훈련 및 검증 세트로 분할한 후, 지정된 RNN 또는 LSTM 모델을 선택하여 훈련을 수행합니다. 훈련 과정에서 손실 값을 기록하고, 각 에포크(epoch)마다 훈련 및 검증 손실을 출력합니다. 마지막으로, 훈련 및 검증 손실의 변화를 시각화하여 플롯을 생성합니다.
#### generate.py
이 코드는 훈련된 문자 수준 언어 모델을 사용하여 주어진 시드 문자와 온도 매개변수에 따라 새로운 문자를 생성합니다. `generate` 함수는 시드 문자를 받아 모델의 출력을 바탕으로 새로운 문자를 순차적으로 생성합니다. 온도 매개변수는 확률 분포를 조절하여 출력의 다양성을 조절합니다. 입력 시드 문자를 인덱스로 변환하고 모델에 전달하여 출력 로짓을 얻고, 이를 소프트맥스 함수로 변환하여 다음 문자를 샘플링합니다. 이 과정을 반복하여 지정된 길이의 문자를 생성합니다.

### 실행 방법
1. `dataset.py`, `model.py`, `main.py`, `generate.py` 파일을 같은 디렉토리에 저장합니다.
2. `main.py`를 실행하여 모델을 학습시킵니다.
3. 학습이 완료되면, `generate.py`를 실행하여 문자를 생성합니다.

#

### 모델 학습 결과
1) 
    batch_size = 32  
    seq_length = 30  
    hidden_size = 128  
    num_layers = 1  
    num_epochs = 5  
    learning_rate = 0.002  
    model_type = 'lstm'  
    device = torch.device('cpu')  
    
Epochs:  20%|██        | 1/5 [05:44<22:56, 344.14s/it]Epoch 1/5, Training Loss: 1.5358, Validation Loss: 1.3917  
Epochs:  40%|████      | 2/5 [11:42<17:36, 352.27s/it]Epoch 2/5, Training Loss: 1.3527, Validation Loss: 1.3286  
Epochs:  60%|██████    | 3/5 [16:34<10:50, 325.11s/it]Epoch 3/5, Training Loss: 1.3044, Validation Loss: 1.2913  
Epochs:  80%|████████  | 4/5 [21:50<05:21, 321.35s/it]Epoch 4/5, Training Loss: 1.2770, Validation Loss: 1.2718  
Epochs: 100%|██████████| 5/5 [26:59<00:00, 323.81s/it]Epoch 5/5, Training Loss: 1.2596, Validation Loss: 1.2609  

![image](https://github.com/An-jeong-min/Language-Modeling/assets/131511349/82762934-743e-4711-a9eb-a15bfcbfa36e)


2)
    batch_size = 32  
    seq_length = 30  
    hidden_size = 128  
    num_layers = 6  
    num_epochs = 10  
    learning_rate = 0.002  
    model_type = 'lstm'  # 'rnn' or 'lstm'  
    device = torch.device('cpu')  

  
이 과제에서는 Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델을 구축하고, vanilla RNN과 LSTM의 성능을 비교하며, 다양한 온도로 문자를 생성해보는 경험을 합니다. 이를 통해 언어 모델의 작동 원리를 이해하고, 모델의 성능을 개선하는 방법을 학습할 수 있습니다.
