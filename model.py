## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## model.py


import torch
import torch.nn as nn

class CharRNN(nn.Module):
    """ Vanilla RNN Model """
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

class CharLSTM(nn.Module):
    """ LSTM Model """
    def __init__(self, vocab_size, hidden_dim, output_size, n_layers):  # 수정된 생성자
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # 수정된 output_size

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size, device):
        device = next(self.parameters()).device
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), 
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
