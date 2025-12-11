# Sequence models
import torch
import torch.nn as nn


class RecurrentNeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        x = self.rnn(input)
        y_hat = self.fc(x)

        return y_hat


class LSTMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        x = self.rnn(input)
        y_hat = self.fc(x)

        return y_hat


class GRUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        x = self.rnn(input)
        y_hat = self.fc(x)

        return y_hat
