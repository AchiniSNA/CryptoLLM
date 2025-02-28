import torch
from torch import nn
from modules.replication_pad import ReplicationPad1d2

class TokenEmbedding2(nn.Module):
    """
    TokenEmbedding
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding2, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding2(nn.Module):
    """
    PatchEmbedding
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding2, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d2((0, stride))
        self.value_embedding = TokenEmbedding2(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
