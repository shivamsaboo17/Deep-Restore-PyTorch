import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    
    def __init__(self, ni, no, ks, stride=1, pad=1, use_act=True):
        
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(ni, no, ks, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(no)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        op = self.bn(self.conv(x))
        return self.act(op) if self.use_act else op


class ResBlock(nn.Module):
    
    def __init__(self, ni, no, ks):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(ni, no, ks)
        self.block2 = ConvBlock(ni, no, ks, use_act=False)

    def forward(self, x):
        return x + self.block2(self.block1(x))
    

class SRResnet(nn.Module):

    def __init__(self, input_channels, output_channels, res_layers=16):
        super(SRResnet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        _resl = [ResBlock(output_channels, output_channels, 3) for i in range(res_layers)]
        self.resl = nn.Sequential(*_resl)

        self.conv2 = ConvBlock(output_channels, output_channels, 3, use_act=False)
        self.conv3 = nn.Conv2d(output_channels, input_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, input):
        _op1 = self.act(self.conv1(input))
        _op2 = self.conv2(self.resl(_op1))
        op = self.conv3(torch.add(_op1, _op2))
        return op
