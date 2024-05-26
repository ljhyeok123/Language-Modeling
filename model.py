import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state):
        x = self.embedding(x)
        out, hidden_state = self.rnn(x, hidden_state)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, num_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state):
        x = self.embedding(x)
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
