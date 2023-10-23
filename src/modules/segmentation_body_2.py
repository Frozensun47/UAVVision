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

class LAN(nn.Module):
    def __init__(self,c1,c2):
        super(LAN,self).__init__()
        self.x1_branch_1 = Conv(c1, c2, 1, 1)
        self.x2_branch_1 = Conv(c1, c2, 1, 1)
        self.x3_branch_1 = Conv(c2, c2, 3, 1)
        self.x3_branch_1_ = Conv(c2, c2, 3, 1)
        self.x4_branch_1 = Conv(c2, c2, 3, 1)
        self.x4_branch_1_ = Conv(c2, c2, 3, 1)
        self.x5_branch_1 = Conv(c2, c2, 3, 1)
        self.x5_branch_1_ = Conv(c2, c2, 3, 1)
        self.x6_branch_1 = Conv(c2, c2, 3, 1)

    def forward(self,x):
        x1=self.x1_branch_1(x)
        x2=self.x2_branch_1(x)
        x3=self.x3_branch_1(x2)
        x3_=self.x3_branch_1_(x3)
        x4=self.x4_branch_1(x3_)
        x4_=self.x4_branch_1_(x4)
        x5=self.x5_branch_1(x4_)
        x5_=self.x5_branch_1_(x5)
        x6=self.x6_branch_1(x5_)

        concat = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        return Conv(768, 128, 1, 1)(concat)


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

        # for c5
        out_5 = Conv(512, 512, 1, 1)(c5)

        #UCCB256
        UCCB5 = Conv(512, 256, 1, 1)(out_5)
        UCCB5 = nn.Upsample(scale_factor=2, mode='nearest')(UCCB5)
        
        # from c4
        out_4 = Conv(256, 256)(c4)  # 256

        #out_4 goes to DLAN , out_4 goes to TLAN(skip connection)
        concat_5_4_out = torch.cat((UCCB5, out_4), 1)  # 512

        # DLAN256
        out_dlan4_branch_1 = LAN(256*2, 256)(concat_5_4_out)
        out_dlan4_branch_2 = LAN(256*2, 256)(concat_5_4_out)
        out_dlan4 = out_dlan4_branch_1 + out_dlan4_branch_2
        out_dlan4 = Conv(256, 256, 1, 1)(out_dlan4)
    

        # route p3
        #UCCB128
        UCCB4 = Conv(256, 128, 1, 1)(out_dlan4)
        UCCB4 = nn.Upsample(scale_factor=2, mode='nearest')(UCCB4)

        
        out_3 = Conv(128, 128, 1, 1)(c3) 

        concat_4_3_out = torch.cat((out_3, UCCB4), 1)

        # DLAN
        # DLAN128
        out_dlan3_branch_1 = LAN(128*2, 128)(concat_4_3_out)
        out_dlan3_branch_2 = LAN(128*2, 128)(concat_4_3_out)
        out_dlan3 = out_dlan3_branch_1 + out_dlan3_branch_2
        out_dlan3 = Conv(128, 128, 1, 1)(out_dlan3)

        #UCCB64
        UCCB3 = Conv(64, 128, 1, 1)(out_dlan4)
        UCCB3 = nn.Upsample(scale_factor=2, mode='nearest')(UCCB3)

        in2 = Conv(64, 64, 1, 1)(c2)
        concat_3_2_out = torch.cat((in2, UCCB3), 1)

        #DLAN64
        out_dlan2_branch_1 = LAN(64*2, 64)(concat_3_2_out)
        out_dlan2_branch_2 = LAN(64*2, 64)(concat_3_2_out)
        out_dlan2 = out_dlan2_branch_1 + out_dlan2_branch_2
        out_dlan2 = Conv(64, 64, 1, 1)(out_dlan2)


        #DCCB64
        in2_3 = MP()(out_dlan2)
        in2_3 = Conv(64, 64, 1, 1)(in2_3)
        skip_in2_3 = Conv(64, 64, 1, 1)(out_dlan2)
        skip_in2_3 = Conv(64, 64, 3, 2)(skip_in2_3)
        out_DCCB2 = torch.cat((skip_in2_3, in2_3), 1)

        concat_2_3_3 = torch.cat((out_3, out_dlan3, out_DCCB2), 1) #128, 128, 128

        #TLAN128
        out_Tlan3_branch_1 = LAN(128*3, 128)(concat_2_3_3)
        out_Tlan3_branch_2 = LAN(128*3, 128)(concat_2_3_3)
        out_Tlan3_branch_3 = LAN(128*3, 128)(concat_2_3_3)
        out_Tlan3 = out_Tlan3_branch_1 + out_Tlan3_branch_2 + out_Tlan3_branch_3
        out_Tlan3 = Conv(128, 128, 1, 1)(out_Tlan3)

        #DCCB128
        out_3_4 = MP()(out_Tlan3)
        out_3_4 = Conv(128, 128, 1, 1)(out_3_4)
        skip_out_3_4 = Conv(128, 128, 1, 1)(out_Tlan3)
        skip_out_3_4 = Conv(128, 128, 3, 2)(skip_out_3_4)
        DCCB3_out = torch.cat((skip_out_3_4, out_3_4), 1)

        concat_3_4_4 = torch.cat(
            (out_4, out_dlan4, DCCB3_out), 1)  # 256, 256, 256

        #TLAN256
        out_Tlan4_branch_1 = LAN(256*3, 256)(concat_3_4_4)
        out_Tlan4_branch_2 = LAN(256*3, 256)(concat_3_4_4)
        out_Tlan4_branch_3 = LAN(256*3, 256)(concat_3_4_4)
        out_Tlan4 = out_Tlan4_branch_1 + out_Tlan4_branch_2 + out_Tlan4_branch_3
        out_Tlan4 = Conv(256, 256, 1, 1)(out_Tlan4)

        #DCCB256
        out_4_5 = MP()(out_Tlan4)
        out_4_5 = Conv(256, 256, 1, 1)(out_4_5)
        skip_out_4_5 = Conv(256, 256, 1, 1)(out_Tlan4)
        skip_out_4_5 = Conv(256, 256, 3, 2)(skip_out_4_5)
        DCCB4_out = torch.cat((skip_out_4_5, out_4_5), 1)

        concat_4_5_5 = torch.cat(
            (out_5, DCCB4_out), 1)  # 512, 512

        #DLAN512
        out_dlan5_branch_1 = LAN(512*2, 512)(concat_4_5_5)
        out_dlan5_branch_2 = LAN(512*2, 512)(concat_4_5_5)
        out_dlan5 = out_dlan5_branch_1 + out_dlan5_branch_2
        out_dlan5 = Conv(512, 512, 1, 1)(out_dlan5)

        p5 = Conv(512, 512, 1, 1)(out_dlan5)
        p4 = Conv(256, 256, 1, 1)(out_Tlan4)
        p3 = Conv(128, 128, 1, 1)(out_Tlan3)
        p2 = Conv(64, 64, 1, 1)(out_dlan2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = Conv(960, 256, 3, 1)(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5) :
        h, w = 160,160
        p2 = F.interpolate(p2, size=(h, w))
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([ p2, p3, p4, p5], dim=1)


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
