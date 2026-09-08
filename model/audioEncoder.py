import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeChanAdapterSmooth(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(32, 128, kernel_size=1, bias=True)
        self.smooth = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)

    #def forward(self, x, Tv: int):
    def forward(self, x):
        x = x.transpose(1, 2)                              # [B,32,Ta]
        x = self.proj(x)                                   # [B,128,Ta]
        #x = F.interpolate(x, size=Tv, mode='linear', align_corners=False)  # [B,128,Tv]
        x = self.smooth(x); x = self.bn(x); x = self.act(x)# [B,128,Tv]
        return x.transpose(1, 2)                           # [B,Tv,128]
