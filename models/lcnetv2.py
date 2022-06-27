import torch
import torch.nn as nn
import torch.nn.functional as F


NET_CONFIG = {
    # in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut
    "stage1": [64, 3, False, False, False, False],
    "stage2": [128, 3, False, False, False, False],
    "stage3": [256, 5, True, True, True, False],
    "stage4": [512, 5, False, True, False, True],
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True):
        super(ConvBNLayer, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            # weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = identity * x
        return x


class RepDepthwiseSeparable(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size=3,
                 split_pw=False,
                 use_rep=False,
                 use_se=False,
                 use_shortcut=False):
        super(RepDepthwiseSeparable, self).__init__()
        self.is_repped = False
        self.in_channels = in_channels
        self.stride = stride
        self.dw_size = dw_size
        self.split_pw = split_pw
        self.use_rep = use_rep
        self.use_se = use_se
        self.use_shortcut = True if use_shortcut and stride == 1 and in_channels == out_channels else False

        if self.use_rep:
            self.dw_conv_list = nn.ModuleList()
            for kernel_size in range(self.dw_size, 0, -2):
                if kernel_size == 1 and stride != 1:
                    continue
                dw_conv = ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channels,
                    use_act=False)
                self.dw_conv_list.append(dw_conv)
            self.dw_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=(dw_size - 1) // 2,
                groups=in_channels)
        else:
            self.dw_conv = ConvBNLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=dw_size,
                stride=stride,
                groups=in_channels)

        self.act = nn.ReLU()

        if use_se:
            self.se = SEModule(in_channels)

        if self.split_pw:
            pw_ratio = 0.5
            self.pw_conv_1 = ConvBNLayer(
                in_channels=in_channels,
                kernel_size=1,
                out_channels=int(out_channels * pw_ratio),
                stride=1)
            self.pw_conv_2 = ConvBNLayer(
                in_channels=int(out_channels * pw_ratio),
                kernel_size=1,
                out_channels=out_channels,
                stride=1)
        else:
            self.pw_conv = ConvBNLayer(
                in_channels=in_channels,
                kernel_size=1,
                out_channels=out_channels,
                stride=1)

    def forward(self, x):
        if self.use_rep:
            input_x = x
            if self.is_repped:
                x = self.act(self.dw_conv(x))
            else:
                y = self.dw_conv_list[0](x)
                for dw_conv in self.dw_conv_list[1:]:
                    y += dw_conv(x)
                x = self.act(y)
        else:
            x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)
        if self.split_pw:
            x = self.pw_conv_1(x)
            x = self.pw_conv_2(x)
        else:
            x = self.pw_conv(x)
        if self.use_shortcut:
            x = x + input_x
        return x

    def rep(self):
        if self.use_rep:
            self.is_repped = True
            kernel, bias = self._get_equivalent_kernel_bias()

            # self.dw_conv = nn.Conv2d(
            #     in_channels=self.in_channels,
            #     out_channels=self.in_channels,
            #     kernel_size=(3, 3),
            #     stride=(self.stride, self.stride),
            #     padding=(1, 1)
            # )
            self.dw_conv.weight.data = kernel
            self.dw_conv.bias.data = bias

    def _get_equivalent_kernel_bias(self):
        kernel_sum = []
        bias_sum = []
        for dw_conv in self.dw_conv_list:
            kernel, bias = self._fuse_bn_tensor(dw_conv)
            kernel = self._pad_tensor(kernel, to_size=self.dw_size)
            kernel_sum.append(kernel)
            bias_sum.append(bias)
        return sum(kernel_sum), sum(bias_sum)

    @staticmethod
    def _fuse_bn_tensor(branch):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
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


class PPLCNetV2(nn.Module):
    def __init__(self,
                 scale,
                 depths,
                 class_num=1000,
                 dropout_prob=0.,
                 use_last_conv=True,
                 class_expand=1280):
        super(PPLCNetV2, self).__init__()
        self.scale = scale
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand

        self.stem = nn.Sequential(* [
            ConvBNLayer(
                in_channels=3,
                kernel_size=3,
                out_channels=make_divisible(32 * scale),
                stride=2), RepDepthwiseSeparable(
                    in_channels=make_divisible(32 * scale),
                    out_channels=make_divisible(64 * scale),
                    stride=1,
                    dw_size=3)
        ])

        # stages
        self.stages = nn.ModuleList()
        for depth_idx, k in enumerate(NET_CONFIG):
            in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut = NET_CONFIG[
                k]
            self.stages.append(
                nn.Sequential(* [
                    RepDepthwiseSeparable(
                        in_channels=make_divisible((in_channels if i == 0 else
                                                    in_channels * 2) * scale),
                        out_channels=make_divisible(in_channels * 2 * scale),
                        stride=2 if i == 0 else 1,
                        dw_size=kernel_size,
                        split_pw=split_pw,
                        use_rep=use_rep,
                        use_se=use_se,
                        use_shortcut=use_shortcut)
                    for i in range(depths[depth_idx])
                ]))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_last_conv:
            self.last_conv = nn.Conv2d(
                in_channels=make_divisible(NET_CONFIG["stage4"][0] * 2 * scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_prob)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        in_features = self.class_expand if self.use_last_conv else NET_CONFIG["stage4"][0] * 2 * scale
        self.fc = nn.Linear(in_features, class_num)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)
        if self.use_last_conv:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


    def switch_to_deploy(self):
        pass


if __name__ == '__main__':
    # from collections import OrderedDict
    # model_state = OrderedDict()
    # new_state = OrderedDict()
    # model = PPLCNetV2(scale=1.0, depths=[2, 2, 6, 2], dropout_prob=0.2)
    # for k, v in model.state_dict().items():
    #     if k.split('.')[-1] != 'num_batches_tracked':
    #         model_state[k] = v
    # old_dict = torch.load('../utils/PPLCNetV2_base_pretrained.pth')
    #
    # for (k1, v1), (k2, v2) in zip(model_state.items(), old_dict.items()):
    #     if k1.split('.')[-2] == 'fc' and len(v2.shape) == 2:
    #         # print(v2.shape)
    #         v2 = v2.transpose(0, 1)
    #
    #     assert v1.shape == v2.shape, (v1.shape, v2.shape)
    #     new_state[k1] = v2
    #
    # torch.save(new_state, '../pretrains/PPLCNetV2_base_pretrained.pt')
    #
    # model.load_state_dict(new_state, strict=True)

    m = RepDepthwiseSeparable(16, 16, stride=1, use_rep=True)
    print(m)
    m.rep()
    inp = torch.randn((4, 16, 40, 40))
    print(m(inp).shape)


