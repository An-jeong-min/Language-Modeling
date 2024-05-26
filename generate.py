## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## generate.py

import torch
import torch.nn.functional as F
from model import CharRNN, CharLSTM
import dataset


# =============================================================================
## CharRNN
# def generate(model, seed_characters, temperature, char2idx, idx2char, length=100):
#     
#     """ Generate characters
# 
#     Args:
#         model: trained model
#         seed_characters: seed characters
#         temperature: T
#         char2idx: character to index mapping
#         idx2char: index to character mapping
#         length: number of characters to generate
# 
#     Returns:
#         samples: generated characters
#     """
#     
#     model.eval()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     
#     hidden = model.init_hidden(1).to(device)
#     input_seq = torch.tensor([char2idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
#     
#     samples = seed_characters
#     
#     with torch.no_grad():
#         for _ in range(length):
#             output, hidden = model(input_seq, hidden)
#             output = output.squeeze() / temperature  
#             probabilities = F.softmax(output, dim=-1)
#             char_idx = torch.multinomial(probabilities, 1)[-1].item() 
#             
#             samples += idx2char[char_idx]
#             input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
#     
#     return samples
# 
# 
# 
# if __name__ == '__main__':
#     
#     input_file = 'shakespeare_train.txt'
#     dataset_obj = dataset.Shakespeare(input_file)
#     
#     model = CharLSTM(len(dataset_obj.chars), hidden_size=256, num_layers=2)
#     model.load_state_dict(torch.load('CharLSTM.pth'))
# 
#     seed_characters = "To be, or not to be, that is the question:"
#     temperatures = [0.5, 1.0, 1.5]
# 
#     for temperature in temperatures:
#         generated_text = generate(model, seed_characters, temperature, dataset_obj.char2idx, dataset_obj.idx2char)
#         print(f"Temperature: {temperature}\nGenerated Text:\n{generated_text}\n")
# 
# =============================================================================

## charLSTM
def generate(model, seed_characters, temperature, char2idx, idx2char, length=100):
    
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        char2idx: character to index mapping
        idx2char: index to character mapping
        length: number of characters to generate

    Returns:
        samples: generated characters
    """
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # hidden state initialization
    hidden = model.init_hidden(1)  # modified to remove device
    input_seq = torch.tensor([char2idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    
    samples = seed_characters
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output.squeeze() / temperature  
            probabilities = F.softmax(output, dim=-1)
            char_idx = torch.multinomial(probabilities, 1)[-1].item() 
            
            samples += idx2char[char_idx]
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return samples

if __name__ == '__main__':
    
    input_file = 'shakespeare_train.txt'
    dataset_obj = dataset.Shakespeare(input_file)
    
    model = CharLSTM(len(dataset_obj.chars), hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load('CharLSTM.pth'))

    seed_characters = "To be, or not to be, that is the question:"
    temperatures = [0.5, 1.0, 1.5]

    for temperature in temperatures:
        generated_text = generate(model, seed_characters, temperature, dataset_obj.char2idx, dataset_obj.idx2char)
        print(f"Temperature: {temperature}\nGenerated Text:\n{generated_text}\n")
