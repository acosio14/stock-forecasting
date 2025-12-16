# Sequence models
import torch
import torch.nn as nn
from torch.nn import functional as F
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
        y_hat = self.fc(hx[-1])

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


def generate_x_y_batches(features, targets, batch_size):
    data_size = len(features)
    if data_size != len(targets):
        raise ValueError("feature and targets not same length.")
    
    X_batches = generate_batches(features, data_size, batch_size)
    y_batches = generate_batches(targets, data_size, batch_size)

    return X_batches, y_batches


def train_model(model, num_epochs, batch_size, learning_rate, features, targets):
    """Train Neural Network."""
    
    X_batches, y_batches = generate_x_y_batches(features, targets, batch_size)
    num_batches = len(X_batches)
    
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device="mps")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in range(num_batches):
            X_train = torch.from_numpy(X_batches[batch].astype(np.float32)).to(device="mps")
            y_train = torch.from_numpy(y_batches[batch].astype(np.float32)).to(device="mps")
            
            optimizer.zero_grad()

            y_pred = model(X_train)
            loss = F.mse_loss(y_pred, y_train)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_train_loss = total_loss / num_batches
        losses.append(average_train_loss)

        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {average_train_loss}")
    
    return losses
    

@torch.no_grad()
def evaluate_model(model, num_epochs, batch_size, features, targets):

    X_batches, y_batches = generate_x_y_batches(features, targets, batch_size)
    num_batches = len(X_batches)

    losses = []
    all_pred = []
    actual_targets = []

    model.eval()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in range(num_batches):
            X_test = torch.from_numpy(X_batches[batch].astype(np.float32)).to(device="mps")
            y_test = torch.from_numpy(y_batches[batch].astype(np.float32)).to(device="mps")

            y_pred = model(X_test)
            loss = F.mse_loss(y_pred, y_test)

            total_loss += loss.item()

            all_pred.append(y_pred.cpu().numpy())
            actual_targets.append(y_test.cpu().numpy)
                
        average_eval_loss = total_loss/num_batches
        losses.append(average_eval_loss)

        print(f"Epoch {epoch + 1}")
        print(f"Eval Loss: {average_eval_loss}")
    
    return all_pred, actual_targets, losses
