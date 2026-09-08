import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size = 1, padding = 0, bias = False)
        self.bn = nn.BatchNorm1d(channel, momentum = 0.01, eps = 0.001)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2),2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2) 
        return x

