import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


class ConvBnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, stride=1):
        super(ConvBnLayer, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, k_blocks=4):
        super(MobileOneBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_deploy = False
        # depth-wise convs
        self.bn1 = nn.BatchNorm2d(in_channels) if (in_channels == out_channels and stride == 1) else None
        self.dw_conv = nn.ModuleList(
           ConvBnLayer(in_channels=in_channels,
                       out_channels=in_channels,
                       kernel_size=kernel_size,
                       groups=in_channels,
                       stride=stride) for _ in range(k_blocks)
        )

        self.conv1x1 = ConvBnLayer(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   groups=in_channels)

        # point-wise convs
        self.bn2 = nn.BatchNorm2d(in_channels) if (in_channels == out_channels and stride == 1) else None
        self.pw_conv = nn.ModuleList(
            ConvBnLayer(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1) for _ in range(k_blocks)
        )
        # activation
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.is_deploy:
            x = self.act(self.dw_conv(x))
            x = self.act(self.pw_conv(x))
            return x
        else:
            identity = x
            x = sum([conv(x) for conv in self.dw_conv])
            if self.bn1 is not None:
                x = self.act(x + self.conv1x1(identity) + self.bn1(identity))
            else:
                x = self.act(x + self.conv1x1(identity))
            identity = x
            x = sum([conv(x) for conv in self.pw_conv])
            if self.bn2 is not None:
                x = self.act(x + self.bn2(identity))
            else:
                x = self.act(x)
            return x

    def deploy(self):
        self.is_deploy = True
        k_dw, b_dw = self._get_equivalent_kernel_bias(self.dw_conv, self.kernel_size)
        k_conv1x1, b_conv1x1 = self._fuse_bn_tensor(self.conv1x1)
        k_bn, b_bn = self._fuse_bn_tensor(self.bn1, groups=self.in_channels)
        k_dw, b_dw = k_dw + k_bn + self._pad_tensor(k_conv1x1, self.kernel_size), b_dw + b_bn + b_conv1x1
        self.dw_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=(self.kernel_size, self.kernel_size),
                                 stride=(self.stride, self.stride),
                                 padding=(self.kernel_size // 2, self.kernel_size // 2),
                                 groups=self.in_channels)

        self.dw_conv.weight.data = k_dw
        self.dw_conv.bias.data = b_dw

        k_pw, b_pw = self._get_equivalent_kernel_bias(self.pw_conv, 1)
        k_bn, b_bn = self._fuse_bn_tensor(self.bn2, groups=1)
        k_pw, b_pw = k_pw + k_bn, b_pw + b_bn

        self.pw_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=(1, 1))
        self.pw_conv.weight.data = k_pw
        self.pw_conv.bias.data = b_pw

        self.__delattr__('conv1x1')
        if hasattr(self, 'bn1'):
            self.__delattr__('bn1')
        if hasattr(self, 'bn2'):
            self.__delattr__('bn2')

    def _get_equivalent_kernel_bias(self, conv_list, kernel_size):
        kernel_sum = []
        bias_sum = []
        for conv in conv_list:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel = self._pad_tensor(kernel, to_size=kernel_size)
            kernel_sum.append(kernel)
            bias_sum.append(bias)
        return sum(kernel_sum), sum(bias_sum)

    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBnLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_channels // groups  # self.groups
            ks = 1 if groups == 1 else 3
            kernel_value = np.zeros((self.in_channels, input_dim, ks, ks), dtype=np.float32)
            for i in range(self.in_channels):
                if ks == 1:
                    kernel_value[i, i % input_dim, 0, 0] = 1
                else:
                    kernel_value[i, i % input_dim, 1, 1] = 1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def _pad_tensor(tensor, to_size):
        from_size = tensor.shape[-1]
        if from_size == to_size:
            return tensor
        pad = (to_size - from_size) // 2
        return F.pad(tensor, [pad, pad, pad, pad])


mbone_s0 = [
    [64, 2, 1 ,0.75, 4],
    [64, 2, 2 ,0.75, 4],
    [128, 2, 8 ,1.0, 4],
    [256, 2, 5 ,1.0, 4],
    [256, 1, 5 ,1.0, 4],
    [512, 2, 1 ,2.0, 4],
]


class MobileOne(nn.Module):
    def __init__(self, cfg):
        super(MobileOne, self).__init__()
        self.num_stages = len(cfg)

        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 48, (3, 3), (2, 2), 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        in_channels = 48
        for idx in range(1, self.num_stages):
            base_chs, stride, n_blocks, alpha, k = cfg[idx]
            setattr(
                self,
                'stage{}'.format(idx),
                self._build_satge(in_channels, stride, base_chs, n_blocks, alpha, k),
            )
            in_channels = int(base_chs * alpha)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage0(x)
        for idx in range(1, self.num_stages):
            stage = getattr(self, 'stage{}'.format(idx))
            x = stage(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # b, c
        x = self.fc(x)
        return x

    @staticmethod
    def _build_satge(in_chs, stride, base_chs, n_blocks, alpha, k):
        block_list = []
        for idx in range(n_blocks):
            if idx == 0:
                in_ch = in_chs
                s = stride
            else:
                in_ch = int(base_chs * alpha)
                s = 1
            block_list.append(
                MobileOneBlock(in_channels=in_ch, out_channels=int(base_chs * alpha), stride=s, k_blocks=k)
            )
        return nn.Sequential(*block_list)

    def switch_to_deploy(self):
        for idx in range(1, self.num_stages):
            stage = getattr(self, 'stage{}'.format(idx))
            for block in stage:
                if isinstance(block, MobileOneBlock):
                    block.deploy()


if __name__ == '__main__':
    m = MobileOne(mbone_s0)
    m.switch_to_deploy()
    print(m)