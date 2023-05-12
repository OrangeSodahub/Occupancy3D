# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmdet3d.models.backbones.resnet import ConvModule
from mmdet.models import NECKS


@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x