import torch
from hydranet import HydraNet


net = HydraNet(num_heads=2, num_features=1024,num_classes=1000,pretrained=False)
print(net.num_heads)
print(net.tHead.parameters)
