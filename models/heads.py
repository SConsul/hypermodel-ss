from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F

import re

class PseudoHeads(nn.Module):
    def __init__(self,num_heads:int, num_features: int,num_classes:int):
        super().__init__()

        self.num_heads = num_heads
        
        self.heads = []
        for _ in range(self.num_heads):
            self.heads.append(nn.Linear(num_features, num_classes))
        self.heads = nn.ModuleList(self.heads)
    
    def forward(self,x):
        head_out = [head(x) for head in self.heads]
        return head_out

class TargetHead(nn.Module):
    def __init__(self, num_features: int,num_classes:int,pretrained:bool):
        super().__init__()
        self.pretrained = pretrained
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if self.pretrained:
            pattern = re.compile(
                r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
            )
            model_dict = self.state_dict()
            state_dict = torch.load("./pretrained/densenet121-a639ec97.pth")
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            print("num weight transfered=",len(state_dict.keys()))
            for key in state_dict.keys():
                print(key)
            self.load_state_dict(state_dict)


    def forward(self,x):
        return self.classifier(x)



if __name__=="__main__":
    tHead = TargetHead(num_features=1024, num_classes=1000,pretrained=True)
    pHeads = PseudoHeads(num_heads=2,num_features=1024, num_classes=1000)