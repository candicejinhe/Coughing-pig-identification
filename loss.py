import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
    def __init__(self, w_neg=1.2, w_pos=1.6, label_smoothing=0.02):
        """
        轻度惩罚FP：w_neg 稍大于 1（如 1.1~1.3），但 w_pos 仍 ≥ w_neg（如 1.5~1.8），
        这样能抑制FP的同时尽量不损失召回。
        label_smoothing 取很小的值（0.0~0.05），提升稳定性与校准性。
        """
        super(lossAV, self).__init__()
        #self.FC = nn.Linear(256, 2)
        self.FC = nn.Linear(256, 2) #用GRU时进行修改
        # 将权重注册为 buffer，避免每次 forward 重新创建张量
        self.register_buffer("ce_weight", torch.tensor([w_neg, w_pos], dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=self.ce_weight, label_smoothing=label_smoothing)

    def forward(self, x, labels=None):
        x = x.squeeze(1)
        logits = self.FC(x)                           # [B, 2]
        if labels is None:
            # 推理：返回正类(1)概率的一维 numpy
            prob = F.softmax(logits, dim=-1)[:, 1]
            return prob.detach().cpu().numpy()
        else:
            nloss = self.criterion(logits, labels)
            prob = F.softmax(logits, dim=-1)          # [B, 2]
            predLabel = torch.argmax(prob, dim=-1)    # 修正：使用 argmax，而不是 round
            correctNum = (predLabel == labels).sum().float()
            return nloss, prob, logits, correctNum


class lossA(nn.Module):
    def __init__(self, w_neg=1.0, w_pos=1.0, label_smoothing=0.0):
        super(lossA, self).__init__()
        self.FC = nn.Linear(128, 2)
        self.register_buffer("ce_weight", torch.tensor([w_neg, w_pos], dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=self.ce_weight, label_smoothing=label_smoothing)

    def forward(self, x, labels):
        x = x.squeeze(1)
        logits = self.FC(x)
        nloss = self.criterion(logits, labels)
        return nloss


class lossV(nn.Module):
    def __init__(self, w_neg=1.0, w_pos=1.0, label_smoothing=0.0):
        super(lossV, self).__init__()
        self.FC = nn.Linear(128, 2)
        self.register_buffer("ce_weight", torch.tensor([w_neg, w_pos], dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=self.ce_weight, label_smoothing=label_smoothing)

    def forward(self, x, labels, detail=False):
        x = x.squeeze(1)
        logits = self.FC(x)
        nloss = self.criterion(logits, labels)
        if not detail:
            return nloss
        prob = F.softmax(logits, dim=-1)
        predLabel = torch.argmax(prob, dim=-1)
        correctNum = (predLabel == labels).sum().float()
        return nloss, prob, logits, correctNum
