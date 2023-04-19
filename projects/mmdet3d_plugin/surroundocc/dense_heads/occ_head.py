# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

@HEADS.register_module()
class OccHead(nn.Module): 
    def __init__(self,
                 *args,
                 cross_transformer_template=None,
                 self_transformer_template=None,
                 positonal_encoding_template=None,
                 num_classes=18,
                 volume_h=[100, 50, 25],
                 volume_w=[100, 50, 25],
                 volume_z=[8, 4, 2],
                 occ_size=[200, 200, 16],
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 use_mask=False,
                 **kwargs):
        super(OccHead, self).__init__()
        self.conv_input = conv_input
        self.conv_output = conv_output
        
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.occ_size = occ_size

        self.img_channels = img_channels

        self.use_mask = use_mask
        self.use_semantic = use_semantic
        self.embed_dims = embed_dims

        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices

        self.cross_transformer_template = cross_transformer_template
        self.self_transformer_template = self_transformer_template
        self.positional_encoding_template = positonal_encoding_template

        self._init_layers()

    def _init_layers(self):
        self.cross_transformer = nn.ModuleList()
        self.self_transformer = nn.ModuleList()
        self.positional_encoding = nn.ModuleList()
        for i in range(self.fpn_level):
            cross_transformer = copy.deepcopy(self.cross_transformer_template)
            self_transformer = copy.deepcopy(self.self_transformer_template)
            positional_encoding = copy.deepcopy(self.positional_encoding_template)

            # `transformer.embed_dims = [128, 256, 512]`
            cross_transformer.embed_dims = self.cross_transformer_template.embed_dims[i]
            # `num_points = _num_points_ = [2, 4, 8]`
            cross_transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.cross_transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]
            # `feedforward_channels = _ffn_dim_ = [256, 512, 1204]`
            cross_transformer.encoder.transformerlayers.feedforward_channels = \
                self.cross_transformer_template.encoder.transformerlayers.feedforward_channels[i]
            # `transformer.embed_dims = [128, 256, 512]`
            cross_transformer.encoder.transformerlayers.embed_dims = \
                self.cross_transformer_template.encoder.transformerlayers.embed_dims[i]
            #`transformer.embed_dims = [128, 256, 512]`
            cross_transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.cross_transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            #`transformer.embed_dims = [128, 256, 512]`
            cross_transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.cross_transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]
            # `num_layers = _num_layers_ = [1, 3, 6]`
            cross_transformer.encoder.num_layers = self.cross_transformer_template.encoder.num_layers[i]

            self_transformer.embed_dims = self.self_transformer_template.embed_dims[i]
            self_transformer.encoder.transformerlayers.feedforward_channels = \
                self.self_transformer_template.encoder.transformerlayers.feedforward_channels[i]
            self_transformer.encoder.transformerlayers.embed_dims = \
                self.self_transformer_template.encoder.transformerlayers.embed_dims[i]
            self_transformer.encoder.num_layers = self.self_transformer_template.encoder.num_layers[i]
            self_transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.self_transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            self_transformer.encoder.transformerlayers.attn_cfgs[0].num_points = \
                self.self_transformer_template.encoder.transformerlayers.attn_cfgs[0].num_points[i]

            

            positional_encoding.num_feats = self.positional_encoding_template.num_feats[i]
            positional_encoding.row_num_embed = self.positional_encoding_template.row_num_embed[i]
            positional_encoding.col_num_embed = self.positional_encoding_template.col_num_embed[i]

            cross_transformer_i = build_transformer(cross_transformer)
            self_transformer_i = build_transformer(self_transformer)
            positional_encoding_i = build_positional_encoding(positional_encoding)

            self.cross_transformer.append(cross_transformer_i)
            self.self_transformer.append(self_transformer_i)
            self.positional_encoding.append(positional_encoding_i)

        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)
        conv_cfg=dict(type='Conv3d', bias=False)

        # in_channles = conv_intput = [_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64] = [512, 256, 256, 128, 128, 64, 64]
        # out_channles = conv_output = [256, _dim_[1], 128, _dim_[0], 64, 64, 32] = [256, 256, 128, 128, 64, 64, 32]
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)


            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))

            self.deblocks.append(deblock)


        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)


        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                    self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))
        self.mask_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.mask_embedding.append(nn.Embedding(1, self.embed_dims[i]))

        self.transfer_conv = nn.ModuleList()
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg=dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.img_channels[i],
                    out_channels=self.embed_dims[i],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                    nn.ReLU(inplace=True))

            self.transfer_conv.append(transfer_block)

        # Original resolution (200, 200, 16)
        g_xx = np.arange(0, self.occ_size[0]) # [0, 1, ..., 199]
        g_yy = np.arange(0, self.occ_size[1]) # [0, 1, ..., 199]
        g_zz = np.arange(0, self.occ_size[2]) # [0, 1, ..., 15]
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
        coords_grid = torch.from_numpy(np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T)
        self.coords_grid = coords_grid

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.cross_transformer[i].init_weights()
            self.self_transformer[i].init_weights()
                
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def get_multiscale_occ_mask(self, original_gt, scale=0):
        ratio = 2 ** (scale + 1)
        B, H, W, Z = np.array(original_gt.shape)
        H , W, Z = H // ratio, W // ratio, Z // ratio
        gt = torch.zeros([B, H, W, Z]).long().to(original_gt.device) + 17
        original_coords = self.coords_grid
        for i in range(gt.shape[0]):
            original_coords = original_coords.to(original_gt.device)
            voxel_semantics_with_coords = torch.vstack([original_coords.T, original_gt[i].reshape(-1)]).T
            voxel_semantics_with_coords = voxel_semantics_with_coords[voxel_semantics_with_coords[:, 3] < 17]
            downsampled_coords = torch.div(voxel_semantics_with_coords[:, :3].long(), ratio, rounding_mode='floor')
            gt[i, downsampled_coords[:, 0], downsampled_coords[:, 1], downsampled_coords[:, 2]] = \
                                                                                voxel_semantics_with_coords[:, 3]
        mask_gt = (gt == 17)
        return mask_gt

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, occ_gt=None, img_metas=None):

        # image feature map shape: (B, N, C, H, W)
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        volume_embed = []
        pos_h = [200, 100, 50]
        pos_w = [400, 100, 25]

        # fpn_level=3 (embeding_dims=[128, 256, 512])
        for i in range(self.fpn_level):
            # `volume_embedding[i]: (in) h[i]*w[i]*z[i] (out) embeding_dim[i]``
            # `                [0]: (in) 80000 (out) 128`
            # `                [1]: (in) 10000 (out) 256`
            # `                [2]: (in) 1250  (out) 512`
            # `volume_queries` of shape (80000, 128)/(10000, 256)/(1250, 512) => (H*W*Z, C)
            volume_queries = self.volume_embedding[i].weight.to(dtype)
            mask_queries = self.mask_embedding[i].weight.to(dtype)
            
            # volume_h_ = [100, 50, 25]
            # volume_w_ = [100, 50, 25]
            # volume_z_ = [8, 4, 2]
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            _, _, C, H, W = mlvl_feats[i].shape
            # `transfer_conv` transform image feature map
            # `transfer_conv[0]: (in) 512 (out) 128`
            # `             [1]: (in) 512 (out) 256`
            # `             [2]: (in) 512 (out) 512`
            view_features = self.transfer_conv[i](mlvl_feats[i].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)
            mask_gt = self.get_multiscale_occ_mask(occ_gt, scale=i) if occ_gt is not None else None

            # cross attention
            volume_embed_i = self.cross_transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                mask_gt=mask_gt,
                img_metas=img_metas
            )

            # Complete volume queries by adding mask tokens
            volume_embed_i_ = torch.empty((volume_h, volume_w, volume_z, self.embed_dims[i]), device=volume_queries.device)
            volume_embed_i_flatten = volume_embed_i_.reshape(-1, self.embed_dims[i])
            volume_embed_i_flatten[mask_gt.reshape(-1), :] = volume_embed_i
            volume_embed_i_flatten[~mask_gt.reshape(-1), :] = mask_queries.view(1, self.embed_dims[i]).expand((~mask_gt).sum(), self.embed_dims[i])

            # self attention
            volume_pos_self_attn = self.positional_encoding[i](torch.zeros((bs, pos_h[i], pos_w[i]), device=volume_queries.device)).to(dtype)
            volume_embed_i = self.self_transformer[i].diffuse_volume_features(
                [view_features],
                volume_embed_i_flatten,
                volume_pos=volume_pos_self_attn,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                pos_h=pos_h[i],
                pos_w=pos_w[i],
                mask_gt=mask_gt,
                img_metas=img_metas
            )

            volume_embed.append(volume_embed_i)

        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            # `volume_embed_reshape: (bs, Z, H, W, C) -> (bs, C, W, H, Z)`
            volume_embed_reshape_i = volume_embed[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)
            
            volume_embed_reshape.append(volume_embed_reshape_i)
        
        outputs = []
        result = volume_embed_reshape.pop()
        # len(self.deblocks) = 7
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)

            # `out_indices = [0, 2, 4, 6]`
            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp
            
        occ_preds = []
        # `self.occ` transform the feature dimension of fused volume feature to `num_classes`
        # `occ_pred: (bs, num_classes, W, H, Z)`
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)
       
        outs = {
            'volume_embed': volume_embed,
            'occ_preds': occ_preds,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             img_metas):
     
        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):

                pred = preds_dicts['occ_preds'][i][:, 0]
                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(voxel_semantics.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i)) #* focal_weight
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
        else:
            pred = preds_dicts['occ_preds']
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):

                pred = preds_dicts['occ_preds'][i]
                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                # `voxel_semantics` has the highest resolution
                # Here we downsample the `voxel_semantics` to generate low level resolution labels
                gt = multiscale_supervision(voxel_semantics.clone(), ratio, preds_dicts['occ_preds'][i].shape, self.coords_grid)
                # align with the pred
                gt = gt.permute(0, 2, 1, 3)
                
                if not self.use_mask:
                    loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
                else:
                    # TODO: Multi-scale mask camera
                    num_total_samples=mask_camera.sum()
                    loss_occ_i = (criterion(pred, gt.long(), mask_camera, avg_factor=num_total_samples) \
                        + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
                
                # focal weight
                loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict
