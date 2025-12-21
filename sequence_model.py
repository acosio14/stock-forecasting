import numpy as np
import torch
import torch.optim
from torch import nn
from torch.nn import functional as F


class SimpleRNN(nn.Module):
    """Create Simple RNN Model."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        _, hx = self.rnn(x)
        return self.fc(hx[-1])


class SimpleLSTM(nn.Module):
    """create Simple LSTM Model."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        _, (hx, _) = self.lstm(x)
        return self.fc(hx[-1])


def generate_batches(data: np.array, data_size: int, batch_size: int) -> np.array:
    """Split data into batches."""
    idx = np.arange(0, data_size, batch_size)
    return np.array_split(data, idx[1:])


def generate_x_y_batches(
    features: np.array, targets: np.array, batch_size: int
) -> tuple[np.array, np.array]:
    """Split data inot x and y batches using generate_batches."""
    data_size = len(features)
    if data_size != len(targets):
        raise ValueError("feature and targets not same length.")

    x_batches = generate_batches(features, data_size, batch_size)
    y_batches = generate_batches(targets, data_size, batch_size)

    return x_batches, y_batches


def train_model(
    model: SimpleRNN | SimpleLSTM,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
) -> tuple[list[float], list[float], tuple[SimpleRNN | SimpleLSTM, int]]:
    """Train Neural Network."""
    X_batches, y_batches = generate_x_y_batches(X_train, y_train, batch_size)
    num_batches = len(X_batches)

    train_losses = []
    val_losses = []
    lowest_val_loss = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device="mps")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in range(num_batches):
            X_train = torch.from_numpy(X_batches[batch].astype(np.float32)).to(
                device="mps",
            )
            y_train = torch.from_numpy(y_batches[batch].astype(np.float32)).to(
                device="mps",
            )

            optimizer.zero_grad()

            y_pred = model(X_train)
            loss = F.mse_loss(y_pred, y_train)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / num_batches
        train_losses.append(train_loss)

        val_loss = validate_model(model, batch_size, X_val, y_val)
        val_losses.append(val_loss)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = (model, epoch+1)

        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

    return train_losses, val_losses, best_model


@torch.no_grad()
def validate_model(
    model: SimpleRNN | SimpleLSTM, batch_size: int, features: np.array, targets: np.array
) -> float:
    """Validate Model using MSE loss."""
    X_batches, y_batches = generate_x_y_batches(features, targets, batch_size)
    num_batches = len(X_batches)

    model.eval()
    total_loss = 0
    for batch in range(num_batches):
        X_val = torch.from_numpy(X_batches[batch].astype(np.float32)).to(device="mps")
        y_val = torch.from_numpy(y_batches[batch].astype(np.float32)).to(device="mps")

        y_pred = model(X_val)
        loss = F.mse_loss(y_pred, y_val)

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test_model(
    model: SimpleRNN | SimpleLSTM, X_test: np.array, y_test: np.array
) -> torch.tensor:
    """Make predictions on test data."""
    model.eval()

    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device="mps")
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device="mps")

    y_pred = model(X_test)

    return y_pred.cpu()

def calculate_regression_metrics(
    y_pred: torch.tensor, y_true: torch.tensor
) -> tuple[float, float, float, float]:
    """Calculate regression metrics from model predictions."""
    sum_of_squares_residuals = np.sum(np.pow(y_true - y_pred,2))
    total_sum_of_squares = np.sum(np.pow(y_true - np.mean(y_true),2))
    r2_score = np.round( 1 - (sum_of_squares_residuals / total_sum_of_squares) , 3)

    mse = np.round(
        np.mean(np.pow(y_true - y_pred,2)),
    )

    rmse = np.round(np.sqrt(mse),3)

    mae = np.round(np.mean(np.abs(y_true - y_pred)),3)

    return mse, rmse, mae, r2_score
