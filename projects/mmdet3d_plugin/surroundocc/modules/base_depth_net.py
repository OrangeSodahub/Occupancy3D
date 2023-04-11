# Copyright (c) Megvii Inc. All rights reserved.
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.builder import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.surroundocc.modules.lmscnet import LMSCNet_SS
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import depth_loss
from projects.mmdet3d_plugin.surroundocc.modules.depth.sampler import Sampler
from projects.mmdet3d_plugin.surroundocc.modules.depth.frustum_grid_generator import FrustumGridGenerator


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


@BACKBONES.register_module()
class BaseDepthNet(nn.Module):

    def __init__(self,
                 occ_size,
                 pc_range,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 output_channels,
                 depth_net_conf,
                 agg_voxel_mode,
                 ssc_net_conf):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        """
        super(BaseDepthNet, self).__init__()
        self.output_channels = output_channels
        # TODO: fix self.depth_channels
        self.d_bound = d_bound
        self.depth_channels = int((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])
        self.depth_net = DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )
        self.ssc_net = LMSCNet_SS(
            ssc_net_conf['class_num'],
            ssc_net_conf['input_dimensions'],
            ssc_net_conf['out_scale'],
        )
        self.agg_voxel_mode = agg_voxel_mode

        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        disc_cfg = {
            "mode": "LID",
            "num_bins": self.depth_channels,
            "depth_min": self.d_bound[0],
            "depth_max": self.d_bound[1],
        }
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(
            grid_size=occ_size, pc_range=pc_range, disc_cfg=disc_cfg)
        self.sampler = Sampler(mode="bilinear", padding_mode="zeros")

        # ssc loss
        self.class_frequencies_level_1 =  np.array([5.41773033e09, 4.03113667e08])
        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level_1 + 0.001)
        )

    def forward_ssc_net(self, voxel_feats, img_metas):
        return self.ssc_net()
        
    def forward_depth_net(self, img_feat, img_metas, with_depth_gt=False):
        # (1, 6, 512, 116, 200)
        bs, n, c, h, w = img_feat.shape
        img_feat = img_feat.reshape(bs*n, c, h, w)
        # intrinsics_mat: shape (bs, n_cam, 4, 4)
        cam_intrinsic = img_metas[0]['cam_intrinsic']
        intrinsics_mat = torch.from_numpy(np.array(cam_intrinsic))
        intrinsics_mat = intrinsics_mat[None].repeat(bs, 1, 1, 1)

        # Generate sampling grid for frustum volume
        image_shape = intrinsics_mat.new_zeros(bs, 2)
        # TODO: fix 116 200
        image_shape[:, 0:2] = torch.as_tensor((116, 200))
        
        # get depth prediction
        # depth_feature of shape (bs*n, depth_channels, h, w)=(6, 112, 116, 200)
        depth_feature = self.depth_net(img_feat, intrinsics_mat)
        depth_pred = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)

        # add cam dim -> depth_pred: (bs, n, 1, depth_channels, h, w)->(1, 6, 1, 112, 116, 200)
        depth_pred = depth_pred.unsqueeze(1)
        depth_pred = depth_pred.reshape(bs, n, depth_pred.shape[1], depth_pred.shape[2],
                                               depth_pred.shape[3], depth_pred.shape[4])

        # generate voxel features based on depth features
        all_voxel_features = []
        for i in range(n):
            # grid: shape (B, X, Y, Z, 3) -> (1, 200, 200, 16, 3)
            grid = self.grid_generator(
                ego_to_lidar = img_metas[0]['ego2lidar'],
                lidar_to_cam=img_metas[0]['lidar2cam'][i],
                cam_to_img=intrinsics_mat[:, i, :3, :],
                # TODO: image shape (feature map)
                image_shape=image_shape,
            ).to(depth_pred.dtype).to(depth_pred.device)
            # sample frustum volume to generate voxel volume
            # voxel_feaetures: shape (B, C, X, Y, Z)->(1, 1, 200, 200, 16)
            # input: depth->(bs, depth_channel, h, w), grid->(bs, X, Y, Z, 3)
            voxel_features = self.sampler(
                input_features=depth_pred[:, i, ...], grid=grid
            )
            if self.agg_voxel_mode == 'mean' and n > 1:
                ones_feat = depth_pred.new_ones(*depth_pred.shape)
                voxel_mask = self.sampler(
                    input_features=ones_feat[:, i, ...], grid=grid
                )
                if i == 0:
                    voxel_masks = [voxel_mask]
                else:
                    voxel_masks.append(voxel_mask)
            all_voxel_features.append(voxel_features)

        if self.agg_voxel_mode == 'sum':
            agg_voxel_features = sum(all_voxel_features)
        elif self.agg_voxel_mode == 'mean':
            agg_voxel_features = sum(all_voxel_features)
            masks = sum(voxel_masks)
            agg_voxel_features[masks > 0] = (
                agg_voxel_features[masks > 0] / masks[masks > 0]
            )

        # TODO: return depth pred when depth loss fixed
        # agg_voxel_feature: (B, C, X, Y ,Z)->(1, 1, 200, 200, 16)
        if with_depth_gt:
            return agg_voxel_features, depth_pred.squeeze(2)
        return agg_voxel_features, None

    def forward(self, img_feats, img_metas, with_depth_gt=False):
        # img_neck outputs 3 feature layers
        # here only select the first one with
        # TODO: optimize
        img_feat = img_feats[0]
        # 3d voxel features: (1, 1, 200, 200, 16)
        voxel_features, depth_pred = self.forward_depth_net(img_feat, img_metas, with_depth_gt)
        voxel_features = voxel_features.squeeze(1)
        occ_pred = self.forward_ssc_net(voxel_features, img_metas)
        print(occ_pred.shape)

        if with_depth_gt:
            return occ_pred, depth_pred
        return occ_pred, None


    def loss(self, ssc_pred, depth_pred, ssc_gt, depth_gt):
        """depth loss
            depth_pred: shape (bs, N, 64, h, w)
            depth_gt: shape (N, H, W)
        """
        # TODO: only support batch_size=1 now
        # depth_pred: shape (n, 64, h, w)
        if depth_pred.shape[0] == 1:
            depth_pred = depth_pred.squeeze(0)
        else:
            raise NotImplementedError

        losses = dict()

        # TODO: depth loss
        loss = depth_loss(depth_pred, depth_gt)
        losses['depth_loss'] = loss

        # ssc loss
        class_weights_level_1 = self.class_weights_level_1.type_as(ssc_gt)
        ones = torch.ones_like(ssc_gt).to(ssc_gt.device)
        ssc_gt = torch.where(torch.logical_or(ssc_gt==255, ssc_gt==0), ssc_gt, ones) # [1, 200, 200, 16]        
        loss_sc = BCE_ssc_loss(ssc_pred, ssc_gt, class_weights_level_1, 0.54)
        losses['loss_sc'] = loss_sc

        return losses
