import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self, channel):
        super(Detector, self).__init__()
        h = channel // 4
        self.gru_forward = nn.GRU(input_size = channel, hidden_size = h, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = h, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.drop = nn.Dropout(0.3)
        self.proj = nn.Linear(2 * h, channel)
        self.__init_weight()

    def forward(self, x):
        x1, _ = self.gru_forward(self.drop(x))
        x = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x))
        x2 = torch.flip(x2, dims=[1])
        y = torch.cat([x1, x2], dim=-1)                      # [B,T,2h]
        y = self.proj(y)  
        return y

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
