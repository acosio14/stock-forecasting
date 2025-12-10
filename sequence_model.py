# Sequence models
import torch
import torch.nn as nn
#from torch.nn import RNN


class RecurrentNeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1)
        self.fc = nn.Linear(in_features=32, out_features=1)

    
    def forward(sefl,input):
        ...