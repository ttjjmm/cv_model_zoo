import torch
import torch.nn as nn
import torch.nn.functional as F



class SubBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SubBlocks, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=out_channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    @staticmethod
    def __fuse_conv_bn():
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(MobileOneBlock, self).__init__()
        self.k_dep_blocks = nn.ModuleList(
           SubBlocks(in_channels=in_channels,
                     out_channels=in_channels,
                     kernel_size=3) for _ in range(k)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.k_poi_blocks = nn.ModuleList(
            SubBlocks(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1) for _ in range(k)
        )
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # for depthwise blocks
        identity = x
        for m in self.k_dep_blocks:
            x = m(x)
        x = x + self.conv1x1(identity) + self.bn1(identity)
        x = F.relu(x, inplace=True)
        # for pointwise blocks
        identity = x
        for m in self.k_poi_blocks:
            x = m(x)
        x = x + self.bn2(identity)
        x = F.relu(x, inplace=True)
        return x



if __name__ == '__main__':
    inp = torch.randn((4, 8, 40, 40))
    m = MobileOneBlock(8, 8, 2)
    print(m(inp).shape)

