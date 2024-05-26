## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## dataset.py

import torch
from torch.utils.data import Dataset



class Shakespeare(Dataset):
    
    """Shakespeare dataset

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """
    
    ## 입력 파일 로드 및 문자 딕셔너리 생성
    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.chars = sorted(set(self.text))
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        self.text_as_int = [self.char2idx[char] for char in self.text] ## 문자 인덱스 목록 생성
        self.seq_length = 30
    
    ## 시퀀스 길이 30의 청크로 데이터 나누기 및 타겟 생성
    def __len__(self):
        return len(self.text_as_int) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        chunk = self.text_as_int[start_idx:end_idx]
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)
        return input_seq, target_seq



if __name__ == '__main__':
    dataset = Shakespeare(input_file='shakespeare_train.txt')
    for i in range(3):
        input_seq, target_seq = dataset[i]


