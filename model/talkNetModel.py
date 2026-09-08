import torch
import torch.nn as nn

from model.audioEncoder      import TimeChanAdapterSmooth
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import attentionLayer
from model.fusion import Fusion
from model.biGRU import Detector
# from model.Classifier import Detector
#from model.modality_attention import ModalityAttention #模态融合
# from model.CrossModalFusion import CrossModalFusion
#from model.mstcn import MSTCNppFeat
class talkNetModel(nn.Module):
    def __init__(self, visualOnly=False, fusionMode='channel_attn'):
        super(talkNetModel, self).__init__()
        self.visualOnly = visualOnly
        self.fusionMode = fusionMode  # channel_attn = model/fusion.py 通道注意力
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        if visualOnly:
            self.detectorV = Detector(128)
            self.selfV = attentionLayer(d_model = 128, nhead = 8, dropout=0.1)
            return

        # Audio Temporal Encoder
        self.audioEncoder  = TimeChanAdapterSmooth()
        self.fusion = Fusion(256)         # 通道注意力（model/fusion.py）
        self.concat_proj = nn.Sequential( # 用于 concat_only 基线
            nn.Linear(256, 256), nn.ReLU())
        self.detector = Detector(256)
        self.selfAV = attentionLayer(d_model = 256, nhead = 8, dropout=0.1)
        print("[talkNetModel] fusionMode =", self.fusionMode)
        # Assuming self.selfAV is the attention layer, we can add the biGRU here
        #self.biGRU = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x
    
    def forward_visual_only_backend(self, x):
        x = self.detectorV(x)
        x = self.selfV(src = x, tar = x)
        B, T, C = x.shape
        x = x.reshape(B*T, C)
        return x

    def forward_audio_frontend(self, x):    
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)      
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):  
        # x1=audio[B,T,128], x2=visual[B,T,128]
        if self.fusionMode == 'channel_attn':
            # 使用 model/fusion.py 的通道注意力：cat→1x1Conv→BN→Sigmoid→逐通道加权
            x = self.fusion(x1, x2)
        else:
            # 基线：直接拼接（仅用于消融对比）
            x = torch.cat((x1, x2), dim=-1)
            if hasattr(self, 'concat_proj'):
                x = self.concat_proj(x)
        x = self.detector(x)
        x = self.selfAV(src = x, tar = x)
        B, T, C = x.shape
        x = x.reshape(B*T, C)
        return x    

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x
    

