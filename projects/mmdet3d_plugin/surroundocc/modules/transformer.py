# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from termios import BS0
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 use_shift=True,
                 use_cams_embeds=True,
                 rotate_prev_feat=True,
                 # Here we only use prev_feat upon the first layer
                 # whose feature map shape is (100, 100, 8)
                 # so the center is (50, 50), not (100, 100)
                 rotate_center=[50, 50],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)

        self.encoder = build_transformer_layer_sequence(encoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds
        self.rotate_prev_feat = rotate_prev_feat

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'volume_queries'))
    def forward(
            self,
            mlvl_feats,
            volume_queries,
            volume_h,
            volume_w,
            volume_z,
            grid_length,
            volume_pos=None,
            prev_feat=None,
            **kwargs):

        bs = mlvl_feats[0].size(0)
        # `volume_queries`: (H*W*Z, C) -> (H*W*Z, bs, C)
        # `volume_pos`: (bs, pos_dim, H, W) -> (H*W*Z, bs, pos_dim)
        # TODO: verify, here pos_dim=128, only use volume_pos when C=128
        volume_queries = volume_queries.unsqueeze(1).repeat(1, bs, 1)
        if volume_pos is not None:
            volume_pos = volume_pos.unsqueeze(4).repeat(1, 1, 1, 1, volume_z).flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / volume_w
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / volume_h
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = volume_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_feat is not None:
            # pref_feat: (bs, num_prev_feat, c)
            if prev_feat.shape[1] == volume_h * volume_w * volume_z:
                prev_feat = prev_feat.permute(1, 0, 2)
            elif len(prev_feat.shape) == 4:
                prev_feat = prev_feat.view(bs, -1, volume_h * volume_w * volume_z).permute(2, 0, 1)

            if self.rotate_prev_feat:
                for i in range(bs):
                    # num_prev_feat = prev_feat.size(1) = h*w*z
                    # `rotation_angle` is the difference of angle between
                    # the current frame and the previous frame
                    # Here need to align the angle
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_feat = prev_feat[:, i].reshape(
                        volume_h, volume_w, -1).permute(2, 0, 1)
                    tmp_prev_feat = rotate(tmp_prev_feat, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_feat = tmp_prev_feat.permute(1, 2, 0).reshape(
                        volume_h * volume_w * volume_z, 1, -1)
                    prev_feat[:, i] = tmp_prev_feat[:, 0]

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # `feat`: (bs, num_cam, c, h, w) -> (bs, num_cam, c, hw) -> (num_cam, bs, hw, c)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            if self.use_cams_embeds:
                # `cams_embeds`: (num_cam, 256) -> (num_cam, 1, 1, 256)
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                        None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3) # (num_cam, H*W, bs, embed_dims)

        volume_embed = self.encoder(
                volume_queries,
                feat_flatten,
                feat_flatten,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                volume_pos=volume_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_feat=prev_feat,
                shift=shift,
                **kwargs
            )

        return volume_embed
