## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## model.py


import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        # Define output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Forward pass through RNN layer
        output, hidden = self.rnn(input, hidden)
        # Reshape output to fit the fully connected layer
        output = output.contiguous().view(-1, self.hidden_size)
        # Forward pass through fully connected layer
        output = self.fc(output)
        # Reshape output to match the expected shape
        output = output.view(input.size(0), -1, self.output_size)
        return output, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers * 1, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.n_layers * 1, batch_size, self.hidden_size).to(device)
        return hidden, cell

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        # Define output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Forward pass through LSTM layer
        output, hidden = self.lstm(input, hidden)
        # Reshape output to fit the fully connected layer
        output = output.contiguous().view(-1, self.hidden_size)
        # Forward pass through fully connected layer
        output = self.fc(output)
        # Reshape output to match the expected shape
        output = output.view(input.size(0), -1, self.output_size)
        return output, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state with zeros
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device), 
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))