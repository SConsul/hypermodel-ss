import torch.nn as nn
from models.feature_extractor import _Encoder
from models.heads import TargetHead,PseudoHeads

class HydraNet(nn.Module):
    def __init__(self,num_heads:int, num_features: int,num_classes:int,pretrained:bool):
        super().__init__()
        self.num_heads = num_heads
        self.enc = _Encoder(pretrained=pretrained)
        self.tHead = TargetHead(num_features, num_classes)
        self.pHeads = PseudoHeads(self.num_heads,num_features, num_classes)
    
    def forward(self,x):
        x = self.enc(x)
        t_out = self.tHead(x)
        p_out = self.pHeads(x)
        return t_out,p_out