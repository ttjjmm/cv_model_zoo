import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBnLayer, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=out_channels,
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
        self.dw_conv = nn.ModuleList(
           ConvBnLayer(in_channels=in_channels,
                       out_channels=in_channels,
                       kernel_size=kernel_size,
                       stride=stride) for _ in range(k_blocks)
        )
        self.conv1x1 = ConvBnLayer(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # point-wise convs
        self.pw_conv = nn.ModuleList(
            ConvBnLayer(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1) for _ in range(k_blocks)
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.is_deploy:
            x = self.act(self.dw_conv(x))
            x = self.act(self.pw_conv(x))
            return x
        else:
            identity = x
            for conv in self.dw_conv:
                x = conv(x)
            x = self.act(x + self.conv1x1(identity) + self.bn1(identity))
            identity = x
            for conv in self.dw_conv:
                x = conv(x)
            x = self.act(x + self.bn2(identity))
            return x

    def deploy(self):
        self.is_deploy = True
        k_dw, b_dw = self._get_equivalent_kernel_bias(self.dw_conv, self.kernel_size)
        k_conv1x1, b_conv1x1 = self._fuse_bn_tensor(self.conv1x1)
        k_bn, b_bn = self._fuse_bn_tensor(self.bn1)
        k_dw, b_dw = k_dw + k_conv1x1 + self._pad_tensor(k_bn, self.kernel_size), b_dw + b_conv1x1 + b_bn
        self.dw_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=(self.kernel_size, self.kernel_size),
                                 stride=(self.stride, self.stride),
                                 padding=(self.kernel_size // 2, self.kernel_size // 2),
                                 groups=self.in_channels)
        self.dw_conv.weight.data = k_dw
        self.dw_conv.bias.data = b_dw

        k_pw, b_pw = self._get_equivalent_kernel_bias(self.pw_conv, 1)
        k_bn, b_bn = self._fuse_bn_tensor(self.bn2)
        k_pw, b_pw = k_pw + k_bn, b_pw + b_bn
        self.pw_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=(1, 1))
        self.pw_conv.weight.data = k_pw
        self.pw_conv.bias.data = b_pw

        self.__delattr__('conv1x1')
        self.__delattr__('bn1')
        self.__delattr__('bn2')

    def _get_equivalent_kernel_bias(self, conv_list, kernel_size):
        kernel_sum = 0
        bias_sum = 0
        for conv in conv_list:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel = self._pad_tensor(kernel, to_size=kernel_size)
            kernel_sum += kernel
            bias_sum += bias
        return kernel_sum, bias_sum

    @staticmethod
    def _fuse_bn_tensor(branch):
        if hasattr(branch, 'conv'):
            kernel = branch.conv.weight
            bn = branch.bn
        else:
            kernel = 1
            bn = branch
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def _pad_tensor(tensor, to_size):
        from_size = tensor.shape[-1]
        if from_size == to_size:
            return tensor
        pad = (to_size - from_size) // 2
        return F.pad(tensor, [pad, pad, pad, pad])



if __name__ == '__main__':
    m = MobileOneBlock(8, 8)
    # m.deploy()
    print(m)