import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_dim, n_layers, lr):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        out, h = self.rnn(x, h0)
        return out, h
    
    def train(self, x, y):
        self.optimizer.zero_grad()
        pred, hidden = self.rnn(x)
        loss = self.loss_fn(pred, y.view(-1).long())
        loss.backward()
        self.optimizer.step()

    def loss_fn(self, pred_y, true_y, fn='RMSE'):
        if fn=='RMSE':
            criterion = nn.MSELoss()
            return torch.sqrt(criterion(pred_y, true_y))
        return 0
