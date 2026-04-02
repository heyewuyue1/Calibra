from torch import nn
from models.LRU.TreeLRU import TreeLRU, TreeLayerNorm, TreeActivation, DynamicPooling


class TreeLRUNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.model = nn.Sequential(
            TreeLRU(in_features, 128, 64),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            TreeLRU(128, 64, 64),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, tree):
        x = tree
        for layer in self.model:
            x = layer(x)
        return x
