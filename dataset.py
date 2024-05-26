## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## dataset.py

import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30.
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load input file
        with open(input_file, 'r') as f:
            self.text = f.read()

        # Create a dictionary of unique characters
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # Convert text to indices
        self.text_indices = [self.char_to_idx[ch] for ch in self.text]

        # Set sequence length
        self.seq_length = 30

        # Create input-target pairs
        self.data = []
        for i in range(len(self.text_indices) - self.seq_length):
            input_seq = self.text_indices[i:i + self.seq_length]
            target_seq = self.text_indices[i + 1:i + self.seq_length + 1]
            self.data.append((input_seq, target_seq))

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input and target sequences
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

if __name__ == '__main__':
    # Test code to verify implementations
    dataset = Shakespeare('shakespeare_train.txt')
    print(f'Dataset size: {len(dataset)}')
    print('Sample input and target:')
    for i in range(5):
        input_seq, target_seq = dataset[i]
        input_text = ''.join([dataset.idx_to_char[idx.item()] for idx in input_seq])
        target_text = ''.join([dataset.idx_to_char[idx.item()] for idx in target_seq])
        print(f'Input: {input_text}')
        print(f'Target: {target_text}\n')
