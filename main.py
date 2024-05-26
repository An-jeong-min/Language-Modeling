## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import dataset
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output, hidden = model(inputs, model.init_hidden(inputs.size(0), device))
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            output, hidden = model(inputs, model.init_hidden(inputs.size(0), device))
            loss = criterion(output, targets.view(-1))
            
            total_loss += loss.item()
    
    val_loss = total_loss / len(val_loader)
    return val_loss

def main():
    batch_size = 32
    seq_length = 30
    hidden_size = 128
    num_layers = 6
    num_epochs = 10
    learning_rate = 0.002
    model_type = 'lstm'  # 'rnn' or 'lstm'
    device = torch.device('cpu') 

    # Load dataset
    dataset_path = 'shakespeare_train.txt'
    shakespeare_dataset = dataset.Shakespeare(dataset_path)

    # Split data into training and validation sets
    dataset_size = len(shakespeare_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(shakespeare_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(shakespeare_dataset, batch_size=batch_size, sampler=val_sampler)

    input_size = len(shakespeare_dataset.chars)
    output_size = input_size

    if model_type == 'rnn':
        model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    else:
        model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)

        train_losses.append(trn_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
