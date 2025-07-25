import torch
from torch import nn
from torch.nn import functional as F

# Traditional UNet

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=7, stride=1, padding=3, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_channel),
            # nn.Tanh(),
            nn.Conv1d(out_channel, out_channel, kernel_size=7, stride=1, padding=3, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_channel)
            # nn.Tanh()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv1d(channel, channel // 2, kernel_size=1, stride=1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        if out.size(2) < feature_map.size(2):
            out = F.pad(out, (0, feature_map.size(2) - out.size(2)))
        return torch.cat((out, feature_map), dim=1) 


class UNetWithAll(nn.Module):
    def __init__(self):
        super(UNetWithAll, self).__init__()
        self.c1 = Conv_Block(3, 16)
        self.d1 = DownSample(16)
        self.c2 = Conv_Block(16, 32)
        self.d2 = DownSample(32)
        self.c3 = Conv_Block(32, 64)
        self.d3 = DownSample(64)
        self.c4 = Conv_Block(64, 128)
        self.u1 = UpSample(128)
        self.c5 = Conv_Block(128, 64)
        self.u2 = UpSample(64)
        self.c6 = Conv_Block(64, 32)
        self.u3 = UpSample(32)
        self.c7 = Conv_Block(32, 16)
        self.linear = nn.Linear(400, 400)
        self.out = nn.Conv1d(16, 3, kernel_size=7, stride=1, padding=3, dilation=1)

        self.eta = nn.Parameter(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32), requires_grad=True)

    def forward(self, x1): # x1 shape [batch, 3, layers]
        f1 = self.c1(x1) # [batch, 16, layers]
        f2 = self.c2(self.d1(f1)) # [batch, 32, layers//2]
        f3 = self.c3(self.d2(f2)) # [batch, 64, layers//4]
        f4 = self.c4(self.d3(f3)) # [batch, 128, layers//8]

        O1 = self.c5(self.u1(f4, f3))
        O2 = self.c6(self.u2(O1, f2))
        O3 = self.c7(self.u3(O2, f1))

        output1 = self.out(O3)
        eta = self.eta.view(1, 3, 1)
        output2 = output1.squeeze(1)
        return output2*eta
