## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## generate.py

import torch
import numpy as np
import csv
from model import CharLSTM  # model.py에서 CharLSTM을 임포트

def generate(model, seed_characters, temperature, length=100):
    """ 문자를 생성합니다.

    Args:
        model: 학습된 모델
        seed_characters: 시작 문자들
        temperature: 온도
        length: 생성할 시퀀스의 길이

    Returns:
        samples: 생성된 문자들
    """
    model.eval()
    samples = seed_characters

    # 시작 문자를 인덱스로 변환
    with torch.no_grad():
        for _ in range(length):
            input_indices = [model.dataset.char_to_idx[ch] for ch in samples[-len(seed_characters):]]
            input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(next(model.parameters()).device)

            # 모델을 통해 예측된 출력 로짓을 얻습니다.
            hidden = model.init_hidden(1, next(model.parameters()).device)
            output, hidden = model(input_tensor, hidden)

            # 마지막 예측 문자의 로짓을 가져와 온도 스케일링 적용
            logits = output[-1, :] / temperature

            # 소프트맥스를 적용하여 확률을 얻습니다.
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            # 확률 분포로부터 문자를 샘플링합니다.
            sampled_index = np.random.choice(len(model.dataset.chars), p=probabilities)
            sampled_char = model.dataset.idx_to_char[sampled_index]

            samples += sampled_char

    return samples

def save_samples_to_csv(samples_list, filename):
    """ 샘플들을 CSV 파일로 저장합니다.

    Args:
        samples_list: 생성된 문자 리스트
        filename: 출력 CSV 파일 이름
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Generated Text'])
            for samples in samples_list:
                writer.writerow([samples])
        print(f"Samples successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving samples to {filename}: {e}")

# 예제 사용법:
# 검증 성능이 가장 좋은 모델을 로드합니다.
device = torch.device('cpu')  # 또는 'cuda'
input_size = 65  # 실제 데이터셋의 문자 집합 크기로 설정
hidden_size = 128
output_size = input_size  # 문자 집합 크기와 동일
num_layers = 6


# 모델 초기화 및 가중치 로드
model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
model_path = 'model.pth'  # 모델 파일 경로
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 5개의 서로 다른 시작 문자 설정
seed_characters_list = [
    "one",
    "two",
    "three",
    "four",
    "five"
]

# 모델의 데이터셋 설정 (dataset 객체가 model에 포함되어 있어야 합니다)
# 예를 들어, 모델이 학습된 데이터셋의 정보를 가지고 있어야 합니다.
class DummyDataset:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz ")
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

model.dataset = DummyDataset()

# 생성할 샘플들의 리스트 초기화
generated_samples_list = []
temperature = 1.0
length = 100

# 각 시작 문자에 대해 샘플 생성
for seed_characters in seed_characters_list:
    generated_samples = generate(model, seed_characters, temperature, length)
    generated_samples_list.append(generated_samples)

# 생성된 샘플들을 CSV 파일로 저장
save_samples_to_csv(generated_samples_list, 'generated_samples.csv')
