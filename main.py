## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import dataset
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

# =============================================================================
## CharRNN
# def train(model, trn_loader, device, criterion, optimizer):
#     model.train()
#     trn_loss = 0
#     
#     for inputs, targets in trn_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         hidden = model.init_hidden(inputs.size(0)).to(device)
#         outputs, hidden = model(inputs, hidden)
#         loss = criterion(outputs, targets.view(-1))
#         loss.backward()
#         optimizer.step()
#         trn_loss += loss.item()
#     
#     return trn_loss / len(trn_loader)
# =============================================================================

## CharLSTM
def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    
    return trn_loss / len(trn_loader)



# =============================================================================
## CharRNN
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     val_loss = 0
#     
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             hidden = model.init_hidden(inputs.size(0)).to(device)
#             outputs, hidden = model(inputs, hidden)
#             loss = criterion(outputs, targets.view(-1))
#             val_loss += loss.item()
#     
#     return val_loss / len(val_loader)
# =============================================================================

## CharLSTM
def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))  # init_hidden에서 device 설정 필요 없음
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

# =============================================================================
# ## charRNN
# def main():
#     input_file = 'shakespeare_train.txt'
#     batch_size = 128
#     hidden_size = 256
#     num_layers = 2
#     learning_rate = 0.001
#     num_epochs = 10
# 
#     dataset_obj = dataset.Shakespeare(input_file)
#     dataset_size = len(dataset_obj)
#     indices = list(range(dataset_size))
#     split = int(np.floor(0.2 * dataset_size))
#     np.random.shuffle(indices)
#     train_indices, val_indices = indices[split:], indices[:split]
# 
#     train_sampler = SubsetRandomSampler(train_indices)
#     val_sampler = SubsetRandomSampler(val_indices)
# 
#     train_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=train_sampler)
#     val_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=val_sampler)
# 
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = CharLSTM(len(dataset_obj.chars), hidden_size, num_layers).to(device)
# 
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     
#     train_losses = []
#     val_losses = []
# 
#     for epoch in range(num_epochs):
#         trn_loss = train(model, train_loader, device, criterion, optimizer)
#         val_loss = validate(model, val_loader, device, criterion)
#         
#         train_losses.append(trn_loss)
#         val_losses.append(val_loss)
#         
#         print(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
# 
#     # 로스 그래프 그리기
#     plt.figure()
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.show()
# 
#     torch.save(model.state_dict(), 'CharLSTM.pth')
# =============================================================================



def main():
    input_file = 'shakespeare_train.txt'
    batch_size = 128
    hidden_size = 256
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 10

    dataset_obj = dataset.Shakespeare(input_file)
    dataset_size = len(dataset_obj)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=val_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharLSTM(len(dataset_obj.chars), hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 로스 그래프 그리기
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'CharLSTM.pth')


if __name__ == '__main__':
    main()


