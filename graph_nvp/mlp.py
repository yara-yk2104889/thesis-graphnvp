import math
import torch.nn as nn

def lecun_linear(in_features, out_features, bias=True):
    """nn.Linear initialized like chainer's L.Linear default (as the official code):
    weights ~ N(0, 1 / in_features) (LeCun normal), bias 0"""
    linear = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(linear.weight, std=1. / math.sqrt(in_features))
    if bias:
        nn.init.zeros_(linear.bias)
    return linear

class MLP(nn.Module):
    def __init__(self, units, in_size=None):
        super(MLP, self).__init__()
        assert isinstance(units, (tuple, list))
        assert len(units) >= 1
        n_layers = len(units)

        units_list = [in_size] + list(units)
        layers = []
        for i in range(n_layers):
            layers.append(lecun_linear(units_list[i], units_list[i+1]))
            # ReLU between hidden layers, last layer linear (as in the official Chainer MLP)
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, x):
        output = self.layers(x)
        return output
