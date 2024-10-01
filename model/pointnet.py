import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from tqdm import trange

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.hidden_neurons = layers
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.0)
            self.hidden_layers.append(
                layer
            )

    def forward(self, x):
        for hidden_layer in self.hidden_layers[:-1]:
            x = self.activation(hidden_layer(x))
        return self.hidden_layers[-1](x)

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=3,neuron=100, layers=[100,1000,200]):
        super(PointNetEncoder, self).__init__()
        self.neuron = neuron
        self.conv1 = torch.nn.Conv1d(channel, self.neuron, 1)
        self.conv2 = torch.nn.Conv1d(self.neuron, self.neuron, 1)
        self.conv3 = torch.nn.Conv1d(self.neuron, self.neuron, 1)

        # self.mlp = MLP([self.neuron, self.neuron, 1175])
        self.mlp = MLP(layers)

        self.bn1 = nn.ReLU()
        self.bn2 = nn.ReLU()
        self.bn3 = nn.ReLU()
        self.global_feat = global_feat
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9 ** (1 / 2000))
        self.loss_log = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.neuron)
        return self.mlp(x)