# Language Modeling
### 인공신경망과 딥러닝 HW3
### 기계정보공학과 24510091 안정민
#

### 1) 과제 설명
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

### 2) 파일 설명
#### dataset.py
이 코드는 Shakespeare 데이터셋을 불러와 문자 사전을 생성하고, 이를 인덱스로 변환한 후 시퀀스 길이 30의 입력-타겟 쌍으로 데이터를 나누어 torch.utils.data.Dataset 클래스를 상속받아 데이터셋을 정의합니다. 각 시퀀스는 입력과 타겟으로 구성되며, 모델 학습에 사용됩니다. __len__과 __getitem__ 메소드를 구현하여 데이과
#### 1) CharRNN 결과

- **Temperature: 0.5**

> " To be, or not to be, that is the question:  
> What is swords:  
> How shall the stain the son, he cannot the return of my lord, and with deeds, and t"  

생성된 텍스트는 원래 텍스트와 매우 유사하고 일관성이 높습니다. 그러나, 창의성이 떨어지고 단어와 표현이 반복되는 경향이 있습니다.

- **Temperature: 1.0**

> " To be, or not to be, that is the question:  
> But of mar,  
> On the mothers spows,  
> To us, but, them officions, our kided,  
> Now to seft again, y."  

텍스트가 적절한 다양성과 일관성을 유지하고 있습니다. 문장은 온도 0.5보다 덜 예측 가능하며, 더 창의적입니다. 이 값은 텍스트 생성에서 일반적으로 가장 좋은 결과를 낳는 온도입니다.

- **Temperature: 1.5**

> "Generated Text:  
> To be, or not to be, that is the question: Premfepreat the haster  
> Hop. are have moke  
> I worchuets.  
> Varcius; your hing  
> All as,  
> ds blious two,  
> W"  

텍스트가 매우 창의적이지만, 일관성이나 논리성이 떨어집니다. 단어들이 무작위로 보이며, 실제로 의미 있는 문장을 형성하지 못합니다.


#### 2) CharLSTM 결과

- **Temperature: 0.5**

> " To be, or not to be, that is the question:  
> Than see for his more the lied aroung for as the sind,  
> And they purtory his many must he say by his"  

생성된 텍스트는 상대적으로 안정적이고 일관된 양상을 보입니다. 각 문자의 확률 분포가 보다 일반적인 분포에 가깝기 때문에 문장이 자연스럽고 읽기 쉽습니다.  
하지만 너무 일반적일 수 있어서 특별한 내용이나 창의성이 부족할 수 있습니다.  

- **Temperature: 1.0**
  
> "To be, or not to be, that is the question:  
> The poirs Hose how I ward,  
> This both the purnow I saided towers, let not childed bebeal Senting,  
> I "  

온도가 조금 높은 경우, 생성된 텍스트는 좀 더 다양한 문자를 포함하고 있습니다. 이는 텍스트가 더 다양한 방식으로 변형되고 새로운 아이디어나 표현이 등장할 가능성이 있음을 의미합니다.  
 하지만 가끔은 문맥에서 벗어난 이상한 문구나 잘못된 단어들이 등장할 수 있습니다.  

- **Temperature: 1.5**
  
> "To be, or not to be, that is the question:  
> Hlelse  
> My strasm  
> Wca! I good I show of Care.  
> But you uphWelnaine gons in oliard of::  
> Benen every we"   

온도가 매우 높은 경우, 생성된 텍스트는 매우 다양하고 창의적이지만 종종 이해하기 어려운 문장이 생성될 수 있습니다.  
일부 단어는 문맥과 상관없이 사용되거나, 문법적으로 부정확할 수 있습니다. 따라서 이 경우에는 높은 온도로 인해 텍스트의 일부가 무작위로 선택되고 조합되기 때문에 생성된 텍스트가 이해하기 어려울 수 있습니다.  




이 과제에서는 Shakespeare 데이터셋을 사용하여 문자 단위 언어 모델을 구축하고, vanilla RNN과 LSTM의 성능을 비교하며, 다양한 온도로 문자를 생성해보는 경험을 합니다. 이를 통해 언어 모델의 작동 원리를 이해하고, 모델의 성능을 개선하는 방법을 학습할 수 있습니다.
