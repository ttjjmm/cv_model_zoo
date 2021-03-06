# Copyright (c) 2022 Jimmy Tao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

def get_activation(act):
    if act =='H-Swish':
        return nn.Hardswish(inplace=True)
    elif act == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif act == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise NotImplementedError



class ConvBnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, stride=1, act=None):
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
        self.act = None if act is None else get_activation(act)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, k_blocks=4, act='H-Swish'):
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
        self.act = get_activation(act)

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
        # if self.bn1 is not None:
        k_bn, b_bn = self._fuse_bn_tensor(self.bn1, groups=self.in_channels)

        k_dw, b_dw = k_dw + k_bn + self._pad_tensor(k_conv1x1, self.kernel_size), b_dw + b_bn + b_conv1x1
        # else:
        #     k_dw, b_dw = k_dw + k_conv1x1, b_dw + b_conv1x1
        self.dw_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=(self.kernel_size, self.kernel_size),
                                 stride=(self.stride, self.stride),
                                 padding=(self.kernel_size // 2, self.kernel_size // 2),
                                 groups=self.in_channels)

        self.dw_conv.weight.data = k_dw
        self.dw_conv.bias.data = b_dw

        k_pw, b_pw = self._get_equivalent_kernel_bias(self.pw_conv, 1)
        # if self.bn2 is not None:
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
            ks = 1 if groups == 1 else self.kernel_size
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




class RepPAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_features=3,
                 act='H-Swish'):
        super(RepPAN, self).__init__()

        self.channel_align = nn.ModuleList()
        for chs in in_channels:
            self.channel_align.append(
                ConvBnLayer(in_channels=chs, out_channels=out_channels, kernel_size=1, stride=1, act='H-Swish')
            )

        in_channels = [out_channels] * len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.spatial_scales = spatial_scales
        self.num_features = num_features

        NET_CONFIG = {
            # k, in_c, out_c, stride, use_se
            "block1": [
                [kernel_size, out_channels * 2, out_channels * 2, 1],
                [kernel_size, out_channels * 2, out_channels, 1],
            ],
            "block2": [
                [kernel_size, out_channels * 2, out_channels * 2, 1],
                [kernel_size, out_channels * 2, out_channels, 1],
            ]
        }

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                nn.Sequential(*[
                    MobileOneBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=k,
                        stride=s,
                        k_blocks=4,
                        act=act)
                    for i, (k, in_c, out_c, s) in enumerate(NET_CONFIG["block1"])
                ]))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvBnLayer(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            self.bottom_up_blocks.append(
                nn.Sequential(*[
                    MobileOneBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=k,
                        stride=s,
                        k_blocks=4,
                        act=act)
                    for i, (k, in_c, out_c, s) in enumerate(NET_CONFIG["block2"])
                ]))


    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = [self.channel_align[idx](input) for idx, input in enumerate(inputs)]

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        # top_features = None
        # if self.num_features == 4:
        #     top_features = self.first_top_conv(inputs[-1])
        #     top_features = top_features + self.second_top_conv(outs[-1])
        #     outs.append(top_features)

        return tuple(outs)


    def switch_to_deploy(self):
        for module in self.modules():
            if isinstance(module, MobileOneBlock):
                module.deploy()



if __name__ == '__main__':
    m = RepPAN(in_channels=[32, 64, 128], out_channels=96, kernel_size=5)
    m.switch_to_deploy()
    print(m)
    # inps = [torch.randn([4, 32, 40, 40]), torch.randn([4, 64, 20, 20]), torch.randn([4, 128, 10, 10])]
    # for o in m(inps):
    #     print(o.shape)



class EffiDecoupleHead(nn.Module):
    def __init__(self, inch, num_classes, num_anchors):
        super(EffiDecoupleHead, self).__init__()
        base_ch = inch
        self.conv = Conv(inch, base_ch, k=1)
        self.conv_cls = Conv(base_ch, base_ch, k=3, p=1)
        self.conv_reg = Conv(base_ch, base_ch, k=3, p=1)
        self.cls_pred = nn.Conv2d(base_ch, num_classes * num_anchors, (1, 1))
        self.reg_pred = nn.Conv2d(base_ch, 4 * num_anchors, (1, 1))
        self.obj_pred = nn.Conv2d(base_ch, 1 * num_anchors, (1, 1))

    def forward(self, x):
        x = self.conv(x)
        cls = self.conv_cls(x)
        reg = self.conv_reg(x)
        cls_pred = self.cls_pred(cls)
        obj_pred = self.obj_pred(cls)
        reg_pred = self.reg_pred(reg)
        return torch.cat([reg_pred, obj_pred, cls_pred], 1)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            # print(mi)
            if isinstance(mi, nn.Conv2d):
                b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
                # print(b[:, 4].shape)
                # print(b)
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            elif isinstance(mi, EffiDecoupleHead):
                b = mi.obj_pred.bias.view(m.na).detach()
                b += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                mi.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
                b = mi.cls_pred.bias.view(m.na, -1).detach()
                b += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
