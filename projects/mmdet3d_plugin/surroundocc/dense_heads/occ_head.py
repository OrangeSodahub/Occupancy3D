# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from projects.mmdet3d_plugin.surroundocc.modules.utils import nusc_class_frequencies


@HEADS.register_module()
class OccHead(nn.Module): 
    def __init__(self,
                 *args,
                 transformer_template=None,
                 num_classes=18,
                 volume_h=[100, 50, 25],
                 volume_w=[100, 50, 25],
                 volume_z=[8, 4, 2],
                 occ_size=[200, 200, 16],
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 single_scale_fusion=False,
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
        self.embed_dims = embed_dims
        self.lambda_xm = 0.1
        self.single_scale_fusion = single_scale_fusion

        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        self.transformer_template = transformer_template

        self._init_layers()

    def _init_layers(self):
        self.transformer = nn.ModuleList()
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            # `transformer.embed_dims = [128, 256, 512]`
            transformer.embed_dims = transformer.embed_dims[i]

            # `num_points = _num_points_ = [2, 4, 8]`
            # Here only one attn_cfgs: `SpatialCrossAttention`
            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            # `feedforward_channels = _ffn_dim_ = [256, 512, 1204]`
            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]
            
            # `transformer.embed_dims = [128, 256, 512]`
            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            #`transformer.embed_dims = [128, 256, 512]`
            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            
            #`transformer.embed_dims = [128, 256, 512]`
            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]
            
            # `num_layers = _num_layers_ = [1, 3, 6]`
            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)

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

        # occ pred for camera branch
        self.occ_camera = nn.ModuleList()
        for i in self.out_indices:
            occ = build_conv_layer(
                conv_cfg,
                in_channels=out_channels[i],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
            self.occ_camera.append(occ)
        # occ pred for fusion branch
        # if only fuse the last scale, use out_channels[-1]
        self.occ_fusion = nn.ModuleList()
        out_indices = self.out_indices if not self.single_scale_fusion else [-1]
        for i in out_indices:
            occ = build_conv_layer(
                conv_cfg,
                in_channels=out_channels[i],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
            self.occ_fusion.append(occ)

        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                    self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))

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
            self.transformer[i].init_weights()
                
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def get_volume_img_feats(self, mlvl_feats, img_metas):
        # image feature map shape: (B, N, C, H, W)
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        volume_embed = []

        # fpn_level=3 (embeding_dims=[128, 256, 512])
        for i in range(self.fpn_level):
            # `volume_embedding[i]: (in) h[i]*w[i]*z[i] (out) embeding_dim[i]``
            # `                [0]: (in) 80000 (out) 128`
            # `                [1]: (in) 10000 (out) 256`
            # `                [2]: (in) 1250  (out) 512`
            # `volume_queries` of shape (80000, 128)/(10000, 256)/(1250, 512) => (H*W*Z, C)
            volume_queries = self.volume_embedding[i].weight.to(dtype)
            
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

            volume_embed_i = self.transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
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
        
        return outputs, volume_embed

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, voxel_pts_feats=None, feature_fuse=None):

        # get volume image features
        volume_img_feats, volume_img_embed = self.get_volume_img_feats(mlvl_feats, img_metas)
        # fuse volume image features and voxel points features
        if feature_fuse is not None:
            # voxel_pts_feats generated by SparseEncoder has: x (dense tensor), pts_feats (sparse tensor)
            volume_feats = feature_fuse['occ_fuser'](volume_img_feats, voxel_pts_feats['x'], self.single_scale_fusion)
            # TODO: add backbone and neck?
            # volume_feats = feature_fuse['occ_encoder_backbone'](volume_feats)
            # volume_feats = feature_fuse['occ_encoder_neck'](volume_feats)

        # camera only branch
        occ_preds_img = []
        # `self.occ` transform the feature dimension of fused volume feature to `num_classes`
        # `occ_pred: (bs, num_classes, W, H, Z)`
        for i in range(len(volume_img_feats)):
            occ_pred = self.occ_camera[i](volume_img_feats[i])
            occ_preds_img.append(occ_pred)

        # fusion (lidar + camera) branch
        occ_preds_fusion = []
        if feature_fuse is not None:
            for i in range(len(volume_feats)):
                occ_pred = self.occ_fusion[i](volume_feats[i])
                occ_preds_fusion.append(occ_pred)
       
        outs = {
            'volume_img_embed': volume_img_embed,
            'occ_preds_img': occ_preds_img,
            'occ_preds_fusion': occ_preds_fusion,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             img_metas):
     
        class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.type_as(preds_dicts['occ_preds_img'][0]), ignore_index=255, reduction="mean"
        )
        
        loss_dict = {}
        for i, pred_camera in enumerate(preds_dicts['occ_preds_img']):

            ratio = 2**(len(preds_dicts['occ_preds_img']) - 1 - i)

            # `voxel_semantics` has the highest resolution
            # Here we downsample the `voxel_semantics` to generate low level resolution labels
            gt = multiscale_supervision(voxel_semantics.clone(), ratio, pred_camera.shape, self.coords_grid)
            gt = gt.permute(0, 2, 1, 3) # align with the pred
            
            # TODO: add loss weights to bce loss
            loss_occ_i_c = (criterion(pred_camera, gt.long()) + sem_scal_loss(pred_camera, gt.long()) + geo_scal_loss(pred_camera, gt.long()))
            loss_occ_i = loss_occ_i_c

            # fusion loss and kl divergence
            if (not self.single_scale_fusion) or (self.single_scale_fusion and i == len(preds_dicts['occ_preds_img'])-1):
                pred_fusion = preds_dicts['occ_preds_fusion'][i] if not self.single_scale_fusion else preds_dicts['occ_preds_fusion'][-1]
                loss_occ_i_f = (criterion(pred_fusion, gt.long()) + sem_scal_loss(pred_fusion, gt.long()) + geo_scal_loss(pred_fusion, gt.long()))
                xm_loss = F.kl_div(
                    F.log_softmax(pred_camera, dim=1),
                    F.softmax(pred_fusion.detach(), dim=1)
                )
                loss_occ_i += loss_occ_i_f + xm_loss * self.lambda_xm
            
            # focal weight
            loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds_img']) - 1 -i))
            loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict
