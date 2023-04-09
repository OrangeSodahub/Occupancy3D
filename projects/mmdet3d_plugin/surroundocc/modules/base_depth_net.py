# Copyright (c) Megvii Inc. All rights reserved.
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.builder import BACKBONES
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import depth_loss


__all__ = ['BaseLSSFPN']


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        infer_mode=False,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = Mlp(1, mid_channels, mid_channels)
        self.se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        # self.aspp = ASPP(mid_channels, mid_channels, BatchNorm=nn.InstanceNorm2d)

        self.depth_pred = nn.Conv2d(
            mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
        )
        self.infer_mode = infer_mode

    def forward(
        self,
        x=None,
        intrins=None,
        scale_depth_factor=1000.0,
    ):
        """
            x: img_feat of shape (bs*n, c, h, w)    
            intrins: camera intrinsics of shape (bs, 6, 4, 4)
        """
    
        # TODO: fix scaled_pixel_size in infer mode
        inv_intrins = torch.inverse(intrins)
        # pixel_size of shape (bs, 6)
        pixel_size = torch.norm(
            # stack shape (bs, 6, 2)
            torch.stack(
                [inv_intrins[..., 0, 0], inv_intrins[..., 1, 1]], dim=-1
            ),
            dim=-1,
        ).reshape(-1, 1).float().to(x.device)
        scaled_pixel_size = pixel_size * scale_depth_factor

        x = self.reduce_conv(x)
        x_se = self.mlp(scaled_pixel_size)[..., None, None]
        x = self.se(x, x_se)
        x = self.depth_conv(x)
        depth = self.depth_pred(x)
        return depth


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


@BACKBONES.register_module()
class BaseDepthNet(nn.Module):

    def __init__(self,
                 output_channels,
                 depth_net_conf,
                 use_da=False):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        """
        super(BaseDepthNet, self).__init__()
        self.output_channels = output_channels
        # TODO: fix self.depth_channels
        # max/default is 64
        self.depth_channels = 64
        self.depth_net = DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def forward(self, img_feats, img_metas, is_training=False):
        # img_neck outputs 3 feature layers
        # here only select the first one with
        # shape of (bs, n, c, h, w)=(1, 6, 512, 116, 200)
        img_feat = img_feats[0]
        bs, n, c, h, w = img_feat.shape
        img_feat = img_feat.reshape(bs*n, c, h, w)
        # list[ndarray]: shape (6, 4, 4)
        cam_intrinsic = img_metas[0]['cam_intrinsic']
        intrinsics_mat = torch.from_numpy(np.array(cam_intrinsic))
        intrinsics_mat = intrinsics_mat[None].repeat(bs, 1, 1, 1)
        
        # get depth prediction
        # depth_feature of shape (bs*n, 64, h, w)=(6, 64, 116, 200)
        depth_feature = self.depth_net(img_feat, intrinsics_mat)
        depth_pred = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)
        
        # TODO
        img_feats_with_depth = img_feats
        
        if is_training:
            return img_feats_with_depth, depth_pred
        return img_feats_with_depth

    def loss(self, depth_pred, depth_gt):
        """depth loss
            depth_pred: shape (bs*n, 64, h, w)
            depth_gt: shape (m*3,)
        """
        depth_gt = depth_gt.reshape(-1, 3)
        losses = dict()
        loss = depth_loss(depth_pred, depth_gt)
        
        losses['depth_loss'] = loss

        return losses
