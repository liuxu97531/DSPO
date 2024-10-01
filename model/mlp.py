import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.GELU())
        # linear_layers.append(nn.Dropout(0.2))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')

    def forward(self, x):
        return self.layers(x)


class PolyMLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(PolyMLP, self).__init__()
        linear_layers = []
        inject_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                nn.GELU()
            ))
            inject_layers.append(nn.Sequential(
                nn.Linear(layers[0], layers[i + 1]),
                nn.GELU()
            ))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.ModuleList(linear_layers)
        self.inject_layers = nn.ModuleList(inject_layers)

    def forward(self, x):
        x_in = x
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x) * self.inject_layers[i](x_in)
        return self.layers[-1](x)


if __name__ == '__main__':
    net = MLP([232, 128, 1280, 4800, 21504])
    print(net)
    x = torch.randn(2, 59, 232)
    print(net(x).shape)
