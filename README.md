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
#### 1) CharRNN
    input_file = 'shakespeare_train.txt'  
    batch_size = 128  
    hidden_size = 256  
    num_layers = 2  
    learning_rate = 0.001  
    num_epochs = 20

    Epoch 1, Train Loss: 2.5935, Validation Loss: 2.1706  
    Epoch 2, Train Loss: 2.0706, Validation Loss: 1.9588  
    Epoch 3, Train Loss: 1.9171, Validation Loss: 1.8457  
    Epoch 4, Train Loss: 1.8248, Validation Loss: 1.7686  
    Epoch 5, Train Loss: 1.7626, Validation Loss: 1.7197  
    Epoch 6, Train Loss: 1.7172, Validation Loss: 1.6860  
    Epoch 7, Train Loss: 1.6802, Validation Loss: 1.6536  
    Epoch 8, Train Loss: 1.6522, Validation Loss: 1.6325  
    Epoch 9, Train Loss: 1.6276, Validation Loss: 1.6135  
    Epoch 10, Train Loss: 1.6040, Validation Loss: 1.5940
   
 ![image](https://github.com/An-jeong-min/Language-Modeling/assets/131511349/caae0dfb-e771-4093-b42a-9f6469233a4f)

#### 2) CharLSTM
    input_file = 'shakespeare_train.txt'  
    batch_size = 128  
    hidden_size = 256  
    num_layers = 2  
    learning_rate = 0.001  
    num_epochs = 10
    
    Epoch 1, Train Loss: 3.0445, Validation Loss: 2.4835  
    Epoch 2, Train Loss: 2.2962, Validation Loss: 2.1407  
    Epoch 3, Train Loss: 2.0608, Validation Loss: 1.9608  
    Epoch 4, Train Loss: 1.9191, Validation Loss: 1.8485  
    Epoch 5, Train Loss: 1.8276, Validation Loss: 1.7768  
    Epoch 6, Train Loss: 1.7598, Validation Loss: 1.7195  
    Epoch 7, Train Loss: 1.7097, Validation Loss: 1.6775  
    Epoch 8, Train Loss: 1.6687, Validation Loss: 1.6484  
    Epoch 9, Train Loss: 1.6350, Validation Loss: 1.6232  
    Epoch 10, Train Loss: 1.6051, Validation Loss: 1.6000  

   ![image](https://github.com/An-jeong-min/Language-Modeling/assets/131511349/55090548-cd8d-48c3-9cc1-d78e2791e97c)



### generate.py
#### 1) CharRNN 결과

Temperature: 0.5
Generated Text:
To be, or not to be, that is the question:
What is swords:
How shall the stain the son, he cannot the return of my lord, and with deeds, and t

Temperature: 1.0
Generated Text:
To be, or not to be, that is the question:
But of mar,
On the mothers spows,
To us, but, them officions, our kided,
Now to seft again, y.

COR

Temperature: 1.5
Generated Text:
To be, or not to be, that is the question: Premfepreat the haster
Hop. are have moke
I worchuets.

Varcius; your hing
All as,
ds blious two,
W


- Temperature: 0.5
생성된 텍스트는 원래 텍스트와 매우 유사하고 일관성이 높습니다. 그러나, 창의성이 떨어지고 단어와 표현이 반복되는 경향이 있습니다.

- Temperature: 1.0
텍스트가 적절한 다양성과 일관성을 유지하고 있습니다. 문장은 온도 0.5보다 덜 예측 가능하며, 더 창의적입니다. 이 값은 텍스트 생성에서 일반적으로 가장 좋은 결과를 낳는 온도입니다.

- Temperature: 1.5
텍스트가 매우 창의적이지만, 일관성이나 논리성이 떨어집니다. 단어들이 무작위로 보이며, 실제로 의미 있는 문장을 형성하지 못합니다.

- 결론
온도 1.0이 가장 좋은 결과를 제공합니다. 이는 생성된 텍스트가 적절한 다양성과 일관성을 유지하면서도 창의적이기 때문입니다. 온도 0.5는 너무 예측 가능하고 반복적이며, 온도 1.5는 너무 불규칙하고 비논리적입니다.

#### 2) CharLSTM 결과

- Temperature: 0.5  
" To be, or not to be, that is the question:
Than see for his more the lied aroung for as the sind,
And they purtory his many must he say by his"  
생성된 텍스트는 상대적으로 안정적이고 일관된 양상을 보입니다. 각 문자의 확률 분포가 보다 일반적인 분포에 가깝기 때문에 문장이 자연스럽고 읽기 쉽습니다. 하지만 너무 일반적일 수 있어서 특별한 내용이나 창의성이 부족할 수 있습니다.

- Temperature: 1.0  
'To be, or not to be, that is the question:
The poirs Hose how I ward,
This both the purnow I saided towers, let not childed bebeal Senting,
I '  
온도가 조금 높은 경우, 생성된 텍스트는 좀 더 다양한 문자를 포함하고 있습니다. 이는 텍스트가 더 다양한 방식으로 변형되고 새로운 아이디어나 표현이 등장할 가능성이 있음을 의미합니다. 하지만 가끔은 문맥에서 벗어난 이상한 문구나 잘못된 단어들이 등장할 수 있습니다.

- Temperature: 1.5  
"To be, or not to be, that is the question:
Hlelse
My strasm
Wca! I good I show of Care.
But you uphWelnaine gons in oliard of::
Benen every we"  
온도가 매우 높은 경우, 생성된 텍스트는 매우 다양하고 창의적이지만 종종 이해하기 어려운 문장이 생성될 수 있습니다. 일부 단어는 문맥과 상관없이 사용되거나, 문법적으로 부정확할 수 있습니다. 따라서 이 경우에는 높은 온도로 인해 텍스트의 일부가 무작위로 선택되고 조합되기 때문에 생성된 텍스트가 이해하기 어려울 수 있습니다.




이 과제에서는 Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델을 구축하고, vanilla RNN과 LSTM의 성능을 비교하며, 다양한 온도로 문자를 생성해보는 경험을 합니다. 이를 통해 언어 모델의 작동 원리를 이해하고, 모델의 성능을 개선하는 방법을 학습할 수 있습니다.
