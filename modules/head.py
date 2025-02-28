import torch
from torch import nn

class FlattenHead2(nn.Module):
    """
    FlattenHead
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead2, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
