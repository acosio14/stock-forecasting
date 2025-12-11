# Sequence models
import torch
import torch.nn as nn
import torch.optim


class SimpleRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        x = self.rnn(input)
        y_hat = self.fc(x)

        return y_hat


class SimpleLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        x = self.rnn(input)
        y_hat = self.fc(x)

        return y_hat


class myLSTM():
    """LSTM from scratch."""


def train_model(model, num_epochs, num_batches, X, y):
    """Train Neural Network."""

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for iterable in iteration:
        optimizer.zero_grad()

        y_hat = model(X,y)
        loss = loss_function(y_hat, y)

        loss.backward()
        optimizer.step()

        total_loss += loss




