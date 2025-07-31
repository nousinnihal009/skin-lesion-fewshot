import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class ProtoNetEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64):
        super(ProtoNetEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels)
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)
