import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """ Shakespeare dataset for character-level language modeling.
    """
    def __init__(self, file_path, seq_length=50):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        unique_chars = sorted(list(set(text)))
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        
        # Convert characters to indices
        self.data_indices = [self.char_to_index[char] for char in text]
        
        # Prepare sequences and targets
        self.seq_length = seq_length
        self.input_sequences = []
        self.target_sequences = []
        for i in range(len(self.data_indices) - seq_length):
            self.input_sequences.append(self.data_indices[i:i+seq_length])
            self.target_sequences.append(self.data_indices[i+1:i+1+seq_length])

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return torch.tensor(self.input_sequences[index]), torch.tensor(self.target_sequences[index])

if __name__ == '__main__':
    file_path = '/home/idsl/sangbeom/jh/hw/shakespeare_train.txt'
    dataset = ShakespeareDataset(file_path)
    print(dataset[0])
