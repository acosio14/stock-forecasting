# Sequence models
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from numpy.typing import NDArray


class SimpleRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):  # x (batch, seq_len, input_size)
        output, hx = self.rnn(input)
        y_hat = self.fc(hx)

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



def generate_batches(data, data_size, batch_size):
    idx = np.arange(0,data_size,batch_size)
    batches = np.array_split(data, idx[1:])
    
    return batches


def train_model(model, num_epochs, batch_size, learning_rate, features, targets):
    """Train Neural Network."""
    data_size = len(features)
    if data_size != len(targets):
        raise ValueError("feature and targets not same length.")
    
    num_batches = -(-data_size//batch_size)
    
    X_batches = generate_batches(features, data_size, batch_size)
    y_batches = generate_batches(targets, data_size, batch_size)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device="mps")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in range(num_batches):
            X_train = torch.from_numpy(X_batches[batch].astype(np.float32)).to(device="mps")
            y_train = torch.from_numpy(y_batches[batch].astype(np.float32)).to(device="mps")
            
            optimizer.zero_grad()

            y_hat = model(X_train)
            loss = loss_function(y_hat, y_train)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_train_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}")
        print(f"Total loss: {average_train_loss}")



