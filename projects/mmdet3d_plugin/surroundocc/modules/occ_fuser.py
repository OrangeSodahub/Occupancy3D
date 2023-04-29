import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class VisFuser(nn.Module):
    """Fuse the image and point cloud features with visibility weight.
        only used in the branch of fusion (lidar and image)
    """
    def __init__(self, embed_dimgs, norm_cfg=None) -> None:
        super().__init__()
        self.embed_dims = embed_dimgs
        # TODO: now only use the last level feature
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(self.embed_dims[0], self.embed_dims[0], 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.embed_dims[0])[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(self.embed_dims[0], self.embed_dims[0], 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.embed_dims[0])[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*self.embed_dims[0], 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_voxel_feats, pts_voxel_feats):

        # multi-level feature fusion
        voxel_feats_list = []
        
        # TODO: now only fuse the last level feature
        img_voxel_feats = img_voxel_feats[-1]

        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats
        voxel_feats_list.append(voxel_feats)

        return voxel_feats_list
