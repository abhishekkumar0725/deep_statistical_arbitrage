import torch
import torch.optim
import torch.nn as nn
from wavenet.networks import WaveNet as WaveNetModule

class WaveNet:
    def __init__(self, input_size, layer_size, stack_size, res_channels, lr):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.trained = False
        self.lr = lr
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)
        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

    def foward(self, inputs):
        outputs = self.net(inputs)
        return outputs

    def train(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.net(x)
        loss = self.loss_fn(pred.view(-1, self.in_channels), y.long().view(-1))
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def loss_fn(self, pred_y, true_y, fn='RMSE'):
        if fn=='RMSE':
            criterion = nn.MSELoss()
            return torch.sqrt(criterion(pred_y, true_y))
        return 0
    