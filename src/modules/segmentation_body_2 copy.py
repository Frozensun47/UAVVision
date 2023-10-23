# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 10:29
# @Author  : zhoujun
import torch
import torch.nn.functional as F
from torch import nn

from .basic import ConvBnRelu


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False).to('cuda')
        self.bn = nn.BatchNorm2d(c2).to('cuda')
        self.act = nn.SiLU().to('cuda') if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity().to('cuda'))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SPPCSPC(nn.Module):
    def __init__(self, c1=512, c2=512, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 1)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c2, c2, 1, 1)
        self.cv6 = Conv(c2, c2, 3, 1)
        self.cv7 = Conv(2 * c2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.cv3(x1)
        x1 = self.cv4(x1)
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class FPN(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256):
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x
        # c2 64
        # c3 128
        # c4 256
        # c5 512

        # Top-down
        # for c5
        in5 = Conv(512, 512, 1, 1)(c5)

        upsample1 = nn.Upsample(scale_factor=2, mode='nearest')(in5)  # 256
        
        # from c4
        in4 = Conv(256, 256)(c4)  # 256
        #in4 goes to DLAN , in4 goes to TLAN(skip connection)

        concat_4_5_out = torch.cat((upsample1, in4), 1)  # 512
        # DLAN
        in41_branch_1 = Conv(512, 256, 1, 1)(concat_4_5_out)
        in42_branch_1 = Conv(512, 256, 1, 1)(concat_4_5_out)

        in43_branch_1 = Conv(256, 128, 3, 1)(in43_branch_1)
        in43_branch_1_ = Conv(256, 128, 3, 1)(in43_branch_1)

        in44_branch_1 = Conv(128, 128, 3, 1)(in43_branch_1_)
        in44_branch_1_ = Conv(128, 128, 3, 1)(in44_branch_1)

        in45_branch_1 = Conv(128, 128, 3, 1)(in44_branch_1_)
        in45_branch_1_ = Conv(128, 128, 3, 1)(in45_branch_1)

        in46_branch_1 = Conv(128, 128, 3, 1)(in45_branch_1_)
        concat_elan4_branch_1 = torch.cat(
            (in41_branch_1, in42_branch_1, in43_branch_1, in44_branch_1, in45_branch_1, in46_branch_1), 1)
        out_elan4_branch_1 = Conv(1024, 256, 1, 1)(concat_elan4_branch_1)

        in41_branch_2 = Conv(512, 256, 1, 1)(concat_4_5_out)
        in42_branch_2 = Conv(512, 256, 1, 1)(concat_4_5_out)

        in43_branch_2 = Conv(256, 128, 3, 1)(in42_branch_2)
        in43_branch_2_ = Conv(256, 128, 3, 1)(in43_branch_2)

        in44_branch_2 = Conv(128, 128, 3, 1)(in43_branch_2_)
        in44_branch_2_ = Conv(128, 128, 3, 1)(in44_branch_2)

        in45_branch_2 = Conv(128, 128, 3, 1)(in44_branch_2_)
        in45_branch_2_ = Conv(128, 128, 3, 1)(in45_branch_2)

        in46_branch_2 = Conv(128, 128, 3, 1)(in45_branch_2_)
        concat_elan4_branch_2 = torch.cat(
            (in41_branch_2, in42_branch_2, in43_branch_2, in44_branch_2, in45_branch_2, in46_branch_2), 1)
        out_elan4_branch_2 = Conv(1024, 256, 1, 1)(concat_elan4_branch_2)

        out_elan4 = out_elan4_branch_1 + out_elan4_branch_2
        


        # route p3
        in4_3 = Conv(256, 128, 1, 1)(out_elan4)
        in4_3 = nn.Upsample(scale_factor=2, mode='nearest')(in4_3)
        
        in3 = Conv(128, 128, 1, 1)(c3)
        concate_4_3 = torch.cat((in3, in4_3), 1)

        # elan
        in31 = Conv(256, 256, 1, 1)(concate_4_3)
        in32 = Conv(256, 256, 1, 1)(concate_4_3)
        in33 = Conv(256, 128, 3, 1)(in32)
        in34 = Conv(128, 128, 3, 1)(in33)
        in35 = Conv(128, 128, 3, 1)(in34)
        in36 = Conv(128, 128, 3, 1)(in35)
        concat_3 = torch.cat((in31, in32, in33, in34, in35, in36), 1)
        out_elan3 = Conv(1024, 128, 1, 1)(concat_3)  # out3

        in3_4 = MP()(out_elan3)
        in3_4 = Conv(128, 128, 1, 1)(in3_4)
        skip_in3_4 = Conv(128, 128, 1, 1)(out_elan3)
        skip_in3_4 = Conv(128, 128, 3, 2)(skip_in3_4)
        concat4_out = torch.cat((skip_in3_4, in3_4, out_elan4), 1)

        # elan
        in41_out_final = Conv(512, 256, 1, 1)(concat4_out)
        in42_out_final = Conv(512, 256, 1, 1)(concat4_out)
        in43_out_final = Conv(256, 128, 3, 1)(in42_out_final)
        in44_out_final = Conv(128, 128, 3, 1)(in43_out_final)
        in45_out_final = Conv(128, 128, 3, 1)(in44_out_final)
        in46_out_final = Conv(128, 128, 3, 1)(in45_out_final)
        concat_elan4_out_final = torch.cat(
            (in41_out_final, in42_out_final, in43_out_final, in44_out_final, in45_out_final, in46_out_final), 1)
        out_elan4_out_final = Conv(1024, 256, 1, 1)(
            concat_elan4_out_final)  # out 4

        in4_5 = MP()(out_elan4_out_final)
        in4_5 = Conv(256, 128, 1, 1)(in4_5)
        skip_in4_5 = Conv(256, 128, 1, 1)(out_elan4_out_final)
        skip_in4_5 = Conv(128, 128, 3, 2)(skip_in4_5)
        concat5_out = torch.cat((skip_in4_5, in4_5, in5), 1)

        # elan
        in51 = Conv(768, 512, 1, 1)(concat5_out)
        in52 = Conv(768, 256, 1, 1)(concat5_out)
        in53 = Conv(256, 128, 3, 1)(in52)
        in54 = Conv(128, 128, 3, 1)(in53)
        in55 = Conv(128, 128, 3, 1)(in54)
        in56 = Conv(128, 128, 3, 1)(in55)
        concat5 = torch.cat((in51, in52, in53, in54, in55, in56), 1)
        out_elan5 = Conv(1280, 512, 1, 1)(concat5)  # out5

        p5 = Conv(512, 256, 1, 1)(out_elan5)
        p4 = Conv(256, 128, 1, 1)(out_elan4_out_final)
        p3 = Conv(128, 128, 1, 1)(out_elan3)

        x = self._upsample_cat(p3, p4, p5)
        x = Conv(512, 256, 3, 1)(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p3, p4, p5) :
        h, w = 160,160
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([ p3, p4, p5], dim=1)


class FPEM_FFM(nn.Module):
    def __init__(self,
                 backbone_out_channels,
                 inner_channels=128,
                 fpem_repeat=2):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        self.conv_out = inner_channels
        inplace = True
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3],
                                         inner_channels,
                                         kernel_size=1,
                                         inplace=inplace)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(self.conv_out))
        self.out_channels = self.conv_out * 4

    def forward(self, x):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:])
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:])
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:])
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        return Fy


class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # down 阶段
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=stride,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
