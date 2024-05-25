## 인공신경망과 딥러닝 과제 #3
## 기계정보공학과 24510091 안정민
## generate.py


import torch
import numpy as np

def generate(model, seed_characters, temperature, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        length: length of generated sequence

    Returns:
        samples: generated characters
    """

    model.eval()
    samples = seed_characters

    # Convert seed characters to indices
    with torch.no_grad():
        for _ in range(length):
            input_indices = [model.dataset.char_to_idx[ch] for ch in samples]
            input_tensor = torch.LongTensor(input_indices).unsqueeze(0)

            # Forward pass through the model to get the output logits
            output, _ = model(input_tensor, model.init_hidden(1))

            # Get the last predicted character's logits and apply temperature scaling
            logits = output[:, -1, :] / temperature

            # Apply softmax to obtain probabilities
            probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()

            # Sample a character from the probability distribution
            sampled_index = np.random.choice(len(model.dataset.chars), p=probabilities)
            sampled_char = model.dataset.idx_to_char[sampled_index]

            samples += sampled_char

    return samples