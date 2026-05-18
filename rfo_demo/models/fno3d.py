import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    @staticmethod
    def compl_mul3d(input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1)
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2)
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3)
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4)
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 3
        self.fc0 = nn.Linear(5, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = nn.BatchNorm3d(self.width)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.bn2 = nn.BatchNorm3d(self.width)
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, t):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, t, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])
        x = F.gelu(self.bn0(self.conv0(x) + self.w0(x)))
        x = F.gelu(self.bn1(self.conv1(x) + self.w1(x)))
        x = self.bn2(self.conv2(x) + self.w2(x))
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1, 1).repeat(batchsize, 1, size_y, size_z, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1, 1).repeat(batchsize, size_x, 1, size_z, 1)
        gridz = torch.linspace(0, 1, size_z, device=device).reshape(1, 1, 1, size_z, 1).repeat(batchsize, size_x, size_y, 1, 1)
        return torch.cat((gridx, gridy, gridz), dim=-1)
