import torch
import torch.optim as optim
import torch.nn as nn
from dataset import ShakespeareDataset
from model import CharRNN, CharLSTM
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

def train(model, data_loader, device, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device).view(-1)
        hidden_state = model.init_hidden(inputs.size(0))
        optimizer.zero_grad()
        with autocast():
            output, _ = model(inputs, hidden_state)
            loss = criterion(output, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1)
            hidden_state = model.init_hidden(inputs.size(0))
            with autocast():
                output, _ = model(inputs, hidden_state)
                loss = criterion(output, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ShakespeareDataset('/home/idsl/sangbeom/jh/homework/shakespeare_train.txt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    vocab_size = len(dataset.char_to_index)
    hidden_dim = 64
    output_dim = vocab_size
    num_layers = 3
    
    rnn = CharRNN(vocab_size, hidden_dim, output_dim, num_layers).to(device)
    lstm = CharLSTM(vocab_size, hidden_dim, output_dim, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(rnn.parameters())
    optimizer_lstm = optim.Adam(lstm.parameters())
    
    scaler = GradScaler()

    num_epochs = 10
    best_val_loss_rnn = float('inf')
    best_val_loss_lstm = float('inf')

    for epoch in range(num_epochs):
        # RNN 모델 학습 및 검증
        train_loss_rnn = train(rnn, train_loader, device, criterion, optimizer_rnn, scaler)
        val_loss_rnn = validate(rnn, val_loader, device, criterion)
        print(f'Epoch {epoch+1}, Train Loss RNN: {train_loss_rnn:.4f}, Val Loss RNN: {val_loss_rnn:.4f}')
        
        if val_loss_rnn < best_val_loss_rnn:
            best_val_loss_rnn = val_loss_rnn
            torch.save(rnn.state_dict(), '/home/idsl/sangbeom/jh/hw/best_rnn_model.pth')
    
    for epoch in range(num_epochs):
        # LSTM 모델 학습 및 검증
        train_loss_lstm = train(lstm, train_loader, device, criterion, optimizer_lstm, scaler)
        val_loss_lstm = validate(lstm, val_loader, device, criterion)
        print(f'Epoch {epoch+1}, Train Loss LSTM: {train_loss_lstm:.4f}, Val Loss LSTM: {val_loss_lstm:.4f}')
        
        if val_loss_lstm < best_val_loss_lstm:
            best_val_loss_lstm = val_loss_lstm
            torch.save(lstm.state_dict(), '/home/idsl/sangbeom/jh/hw/best_lstm_model.pth')

if __name__ == '__main__':
    main()
