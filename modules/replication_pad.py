import torch
from torch import nn

class ReplicationPad1d2(nn.Module):
    def __init__(self, padding):
        super(ReplicationPad1d2, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        print('ReplicationPad1d output')
        return output