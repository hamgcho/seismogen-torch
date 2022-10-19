import seisbench.data as sbd
import seisbench
import numpy as np
import torch
from torch import nn

class GenBlock(torch.nn.Module):
    '''
        this block gets a 1-D array (length 400 latent variable Z[:400] ~ Normal(0,1))
    '''
    def __init__(self, device:str):
        super(GenBlock, self).__init__(self)
        self.z_tconv = torch.nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=128, stride=4, padding=62)
        self.y_linear = torch.nn.Linear(in_features=1, out_features=400)
        self.conv1 = torch.nn.Conv1d(in_channels=32, out_channels)
        self.conv2 = torch.nn.Conv1d()
        self.conv3 = torch.nn.Conv1d()
    
    def forward(self, y:int):
        z = np.random.normal(size=(1, 400))
        z = torch.Tensor(z)
        z = self.tconv(z)
        
        y = torch.Tensor([[torch.float(y)]])
        x = torch.cat([y, z], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
    
class GenNetwork(torch.nn.Module):
    def __init__(self, ):
        super(GenNetwork, self).__init__(self)
        pass
        
    def forward(self, z):
        pass

