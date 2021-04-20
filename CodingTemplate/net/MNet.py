import torch.nn as nn
import numpy as np 

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.linear = nn.Linear(3,4) #test
    def forward(self,x):
        out = x
        return out
