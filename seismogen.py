from email.generator import Generator
import seisbench.data as sbd
import seisbench
import numpy as np
import torch
from torch import nn

class GenBlock(torch.nn.Module):
    '''
        this block gets a 1-D array (length 400 latent variable Z[:400] ~ Normal(0,1))
    '''
    def __init__(self, length:int=1600):
        super(GenBlock, self).__init__()
        self.z_tconv = torch.nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=128, stride=4)
        
        # TODO :  the detail of whether how the size is re-operated so that it is of size 1600.
        
        self.z_upsample = torch.nn.Upsample(size=length)
        self.z_relu = torch.nn.ReLU()
        self.z_BN = torch.nn.BatchNorm1d(1)
        
        
        self.conv1 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=128, stride=1)
        self.upsample1 = torch.nn.Upsample(size=length)
        self.relu1 = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(32)
        
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=128, stride=1)
        self.upsample2 = torch.nn.Upsample(size=length)
        self.relu2 = torch.nn.ReLU()
        self.BN2 = torch.nn.BatchNorm1d(32)
        
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=128, stride=1)
        self.upsample3 = torch.nn.Upsample(size=length)
        
        
        
    def forward(self, processed_y:torch.Tensor):
        '''
            input y corresponds to the concept of label, which tells whether it is a positive/negative sample.
            
        '''
        z = np.random.normal(size=(1, 400))
        z = torch.Tensor([z])
        z = self.z_tconv(z)
        z = self.z_upsample(z)
        
        y = processed_y
        x = torch.cat([y, z], dim=1)
        
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.relu1(x)
        x = self.BN1(x)
        
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.relu2(x)
        x = self.BN2(x)
        
        x = self.conv3(x)
        x = self.upsample3(x)

        return x
class GenNetwork(torch.nn.Module):
    def __init__(self, length:int=1600):
        super(GenNetwork, self).__init__()
        
        self.y_linear = torch.nn.Linear(in_features=1, out_features=400)
        self.y_tconv = torch.nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=128, stride=4)
        self.y_upsample = torch.nn.Upsample(size=length)
        self.y_relu = torch.nn.ReLU()
        self.y_BN = torch.nn.BatchNorm1d(16)
        
        self.gblocks = [GenBlock() for _ in range(3)]
        
        
        
    def forward(self, y:int):
        y = torch.tensor([[[float(y)]]])
        y = self.y_linear(y)
        y = self.y_tconv(y)
        y = self.y_upsample(y)
        y = self.y_relu(y)
        y = self.y_BN(y)
        
        res = []
        for gblock in self.gblocks:
            res.append(gblock(y))
        res = torch.cat(res, dim=1)
        return res

def adaptive_cutoff():
    '''
        given data, infer the cutoff frequency.
    '''
    pass


class Discriminator(torch.nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        
        # feature extraction
        
        self.x_low_block = torch.nn.Sequential(
            torch.nn.Conv1D(in_channels=1, out_channels=32, kernel_size=16, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1D(in_channels=32, out_channels=32, kernel_size=16, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(size=800)
        )
        
        self.x_high_block = torch.nn.Sequential(
            torch.nn.Conv1D(in_channels=1, out_channels=32, kernel_size=16, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1D(in_channels=32, out_channels=32, kernel_size=16, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(size=800)
        )
        
        self.y_block = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=800),
            torch.nn.Conv1D(in_channels=1, out_channels=32, kernel_size=16, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(size=800)
        )
        
        # sample critic
         
        self.sample_critic = torch.nn.Sequential(
            torch.nn.Conv1D(in_channels=96, out_channels=48, kernel_size=16, stride=3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1D(in_channels=48, out_channels=32, kernel_size=16, stride=3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1D(in_channels=32, out_channels=24, kernel_size=16, stride=3),
            torch.nn.LeakyReLU(),
        )
    
    def decomposer(self, x:np.array, T:float=3.75)->tuple:
        '''
            with fft, get a spectrogram
            with the freq-component information and cut-off frequency T, decompose it in frequency domain
            And then with inverse fourire transformation, recover the x_low, x_high

            input : 
                np.array signal x
                float T : hyperparameter, cut-off frequency
                TODO : implement adaptive-filter inferrer and use the inference by it as the input in here.
            output : 
                (x_low, x_high) output each as torch.Tensor-formatted
        '''
    
    def forward(self, x:np.array, y:int):
        '''
            1. Feature extraction
                - signal decomposition is included
                - how? fft
            2. Sample critic
            
        '''
        
        # stage 1 : Feature extraction
        
        x_low, x_high = self.decomposer(x)
        
        x_low = self.x_low_block(x_low)
        x_low = self.x_high_block(x_high)
        y = torch.Tensor([[[float(y)]]])
        y = self.y_block(y)
        
        feature = torch.cat([x_low, x_high, y], dim=1)
        
        # stage 2 : Sample Critic 
        
        res = self.sample_critic(feature)
        res = torch.mean(res)
        return res

if __name__ == '__main__':
    G = GenNetwork()
    synthetic = G(0)
    print(synthetic.shape)