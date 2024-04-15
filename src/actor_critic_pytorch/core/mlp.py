import torch
import torch.nn as nn

from typing import List


class MLP(nn.Module):
    """
    A simple feedforward neural network with configurable hidden layers and activation functions.
    """

    _input_dim: int
    _hidden_sz: int
    _n_layers: int
    _output_dim: int
    _activation: str
    _device: torch.device
    _net: nn.Sequential

    def __init__(
        self,
        input_dim: int,
        hidden_sz: int,
        n_layers: int,
        output_dim: int,
        activation: str = "relu",
        device: torch.device = torch.device("cpu"),
    ):
        super(MLP, self).__init__()

        self._input_dim = input_dim
        self._hidden_sz = hidden_sz
        self._n_layers = n_layers
        self._output_dim = output_dim
        self._activation = activation
        self._device = device

        layers = []
        layers.append(self._create_linear_layer(input_dim, hidden_sz, activation))
        for _ in range(n_layers - 2):
            layers.append(self._create_linear_layer(hidden_sz, hidden_sz, activation))
        layers.append(self._create_linear_layer(hidden_sz, output_dim, "linear"))

        self._net = nn.Sequential(*layers).to(device)

        print("*" * 80)
        print(self._net)
        print("*" * 80)

    def _create_linear_layer(self, in_dim: int, out_dim: int, activation: str):
        if activation == "relu":
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        elif activation == "tanh":
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
        elif activation == "sigmoid":
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        elif activation == "linear":
            return nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Invalid activation, got {activation}")

    def forward(self, x):
        x = x.to(self._device)
        return self._net(x)
