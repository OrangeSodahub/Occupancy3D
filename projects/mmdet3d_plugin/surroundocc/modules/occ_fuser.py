import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class OccFuser(nn.Module):
    def __init__(self, img_dims, occ_dims, norm_cfg=None) -> None:
        super().__init__()
        self.embed_dims = occ_dims
        self.img_dims = img_dims
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(self.embed_dims[-1], self.embed_dims[-1], 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.embed_dims[-1])[1],
            nn.ReLU(inplace=True),
        )

        self.occ_enc = nn.Sequential(
            nn.Conv3d(self.embed_dims[-1], self.embed_dims[-1], 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.embed_dims[-1])[1],
            nn.ReLU(inplace=True),
        )

        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*self.embed_dims[-1], self.embed_dims[-1], 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.embed_dims[-1])[1],
            nn.ReLU(inplace=True),
            nn.Conv3d(self.embed_dims[-1], 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.init_align_layers()

    def init_align_layers(self):
        # TODO: now only support fuse the last layer (dim=16)
        self.align = ConvModule(
            32,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )

    def forward(self, occ_feats, img_feats):

        # TODO: now only support fuse the last layer (dim=16)
        img_feat = img_feats[0].permute(0, 1, 3, 4, 2)
        for i, occ_feat in enumerate(occ_feats):
            if i != len(occ_feats) - 1:
                continue
            img_feat = self.align(img_feat)
            img_feat = self.img_enc(img_feat)
            occ_feat = self.occ_enc(occ_feat)
            feats = torch.cat([img_feat, occ_feat], dim=1)
            vis_weight = self.vis_enc(feats)
            feats = img_feat * vis_weight + (1 - vis_weight) * occ_feat

            occ_feats[i] = feats
        
        return occ_feats