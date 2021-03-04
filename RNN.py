import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, lr):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.trained = False
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.optimizer = optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        out, h = self.rnn(x, h0)
        return out, h
    
    def train(self, x, y, epoch=10, lr=.01):
        for _ in range(epoch):
            self.optimizer.zero_grad()
            pred, hidden = self.rnn(x)
            loss = loss_fn(pred, y.view(-1).long())
            loss.backward()
            optimizer.step()

    def eval(self, x):
        if not self.trained:
            return -1*float('inf')
        return self.rnn(x)

    def loss_fn(self, pred_y, true_y, fn='RMSE'):
        if fn=='RMSE':
            crieterion = nn.MSELoss()
            return torch.sqrt(criterion(pred_y, true_y))
        return 0
