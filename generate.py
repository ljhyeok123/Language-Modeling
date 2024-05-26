import torch
from model import CharRNN, CharLSTM
from dataset import ShakespeareDataset

def generate(model, seed_chars, temperature, idx_to_char, char_to_idx, length=100):
    model.eval()
    device = next(model.parameters()).device
    input = torch.tensor([char_to_idx[ch] for ch in seed_chars], dtype=torch.long).unsqueeze(0).to(device)
    hidden_state = model.init_hidden(1)
    hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
    predicted_chars = seed_chars
    vocab_size = len(idx_to_char)
    
    for _ in range(length):
        output, hidden_state = model(input, hidden_state)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Ensure the generated index is within the valid range
        if top_i.item() >= vocab_size:
            top_i = torch.tensor([vocab_size - 1]).to(device)
        
        predicted_char = idx_to_char[top_i.item()]
        predicted_chars += predicted_char
        input = torch.tensor([[top_i]], dtype=torch.long).to(device)
    
    return predicted_chars

def main():
    # Load the dataset to get character mappings
    dataset = ShakespeareDataset('/home/idsl/sangbeom/jh/hw/shakespeare_train.txt')
    idx_to_char = dataset.index_to_char
    char_to_idx = dataset.char_to_index
    
    input_size = len(char_to_idx)  # Should match the vocab size used during training
    hidden_size = 64  # 히든 크기를 줄임
    output_size = input_size
    num_layers = 3  # 레이어 수를 3으로 설정

    # Load the model with the best validation performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    model.load_state_dict(torch.load('/home/idsl/sangbeom/jh/hw/best_lstm_model.pth'))
    
    # Seed characters for generating samples
    seed_chars_list = ['Once ', 'When ', 'While ', 'Although ', 'However ']
    temperature = 0.8
    generated_samples = []

    for seed_chars in seed_chars_list:
        sample = generate(model, seed_chars, temperature, idx_to_char, char_to_idx, length=100)
        generated_samples.append(sample)
        print(f'Seed: "{seed_chars}"')
        print(sample)
        print()

if __name__ == '__main__':
    main()
