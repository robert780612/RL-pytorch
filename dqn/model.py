import random

import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.ModuleList([
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
        ])
        self.fc = nn.ModuleList([
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions)
        ])

    def forward(self, x):
        x = x / 255
        assert x.max().item() <= 1.0
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc:
            x = layer(x)
        return x

    def feature_size(self):
        x = torch.zeros(1, *self.input_shape)
        for layer in self.features:
            x = layer(x)
        return x.view(1, -1).size(1)

