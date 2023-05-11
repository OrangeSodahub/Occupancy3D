# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from torchvision.transforms.functional import affine
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class SurroundOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_points=False,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 occ_fuser=None,
                 occ_encoder_backbone=None,
                 occ_encoder_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 len_queue=4,
                 voxel_size=[0.4, 0.4, 0.4],
                 is_vis=False,
                 ):

        super(SurroundOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone) if occ_encoder_backbone is not None else None
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck) if occ_encoder_neck is not None else None

        self.use_points = use_points
        self.use_semantic = use_semantic
        self.is_vis = is_vis

        # only use when use_sequential is True
        self.len_queue = len_queue
        self.voxel_size = voxel_size
        self.prev_occ_list = [] # only save `len_queue` previous occ preds
        self.curr_scene_token = None # TODO: for now, only support one gpu with batch size = 1

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, points):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return pts_feats

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, points=None, len_queue=None):
        """Extract features from images and points."""

        # extract features of images
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        voxel_pts_feats = None
        # extract features of points
        if self.use_points and points is not None:
            voxel_pts_feats = self.extract_pts_feat(points)

        return img_feats, voxel_pts_feats

    def forward_pts_train(self,
                          img_feats,
                          voxel_pts_feats,
                          voxel_semantics,
                          mask_camera,
                          img_metas):
        feature_fuse = None
        if self.use_points:
            feature_fuse = dict(
                occ_fuser=self.occ_fuser,
                occ_encoder_backbone=self.occ_encoder_backbone,
                occ_encoder_neck=self.occ_encoder_neck,
            )
        outs = self.pts_bbox_head(
            img_feats, img_metas, voxel_pts_feats, feature_fuse)
        # `voxel_semantics` only used in loss calculation
        # with multi-scale supervision
        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # kwargs: dict_keys(['img_metas', 'img', 'volume_semantics', 'mask_lidar', 'mask_camera])
        # `volume_semantics` of shape (_, 200, 200, 16)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      points=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      ):

        img_feats, voxel_pts_feats = self.extract_feat(img=img, img_metas=img_metas, points=points)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, voxel_pts_feats,
                                            voxel_semantics, mask_camera, img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        # new scene
        # TODO: for now, only support one gpu with batch size = 1
        if self.curr_scene_token is None or img_metas[0][-1]['scene_token'] != self.curr_scene_token:
            self.prev_occ_list = []
            self.curr_scene_token = img_metas[0][-1]['scene_token']

        # get the occ pred of current frame
        output = self.simple_test(
            [img_metas[0][-1]], img, **kwargs)
        
        pred_occ = output['occ_preds_img']

        # `pred_occ` got multi-scale pred results
        # Here we only use the last one with shape of (200, 200, 16)
        # for evalution with ground truth
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]

        # merge the occ pred of multi frames
        pred_occ = self.merge_multi_frame_preds(pred_occ, img_metas, self.prev_occ_list)
        self.prev_occ_list.append(pred_occ)
        while len(self.prev_occ_list) > self.len_queue:
            self.prev_occ_list.pop(0)
        
        if self.is_vis:
            self.generate_output(pred_occ, img_metas)
            return pred_occ.shape[0]

        # `gt_occ/voxel_semantics: (bs, H, W, Z)`
        # `pred_occ: (bs, num_classes, W, H, Z) -> (bs, H, W, Z, num_classes)`
        # `occ_score: (bs, H, W, Z)`
        pred_occ = pred_occ.permute(0, 3, 2, 4, 1)
        occ_score=pred_occ.softmax(-1)
        occ_score=occ_score.argmax(-1)

        return occ_score
        
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, is_train=False)

        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(img=img, img_metas=img_metas)

        output = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

        return output

    def merge_multi_frame_preds(self, target_occ, img_metas_list, prev_occ_list):
        """Only used in offline test pipeline
        :param target_occ: shape (bs, num_classes, W, H, Z)
        """
        # TODO for now, only support one gpu with batch size = 1
        assert len(img_metas_list[0])-1 == len(prev_occ_list)
        bs, num_classes, W, H, Z = target_occ.shape
        assert bs == 1 # only support bs=1
        agg_occ = [target_occ.permute(0, 3, 2, 4, 1)] # (1, bs, H, W, Z, num_classes)
        for i, prev_occ in enumerate(prev_occ_list):
            # affine (rotate + translate), no translation on z-axis
            rotation_angle = np.array([img_metas['can_bus'][-1] for img_metas in img_metas_list[0][:i-self.len_queue-1:-1]]).sum()
            translate_dist = np.array([img_metas['can_bus'][:3] for img_metas in img_metas_list[0][:i-self.len_queue-1:-1]]).sum(0)
            translate_dist = (translate_dist // self.voxel_size).round()
            prev_occ = prev_occ.permute(0, 3, 2, 4, 1).reshape(bs, H, W, -1) # (bs, H, W, Z*num_classes)
            prev_occ[0] = affine(prev_occ[0].permute(2, 0, 1), angle=-rotation_angle, center=(W // 2, H // 2),
                                translate=(translate_dist[1], translate_dist[0]), scale=1., shear=0., fill=(0.,)).permute(1, 2, 0) # NOTE: only support bs=1
            prev_occ = prev_occ.reshape(bs, H, W, Z, num_classes)
            agg_occ.append(prev_occ)
        agg_occ = torch.stack(agg_occ) # (len_queue, bs, H, W, Z, num_classes)
        agg_occ = agg_occ.mean(0) # (bs, H, W, Z, num_classes)
        agg_occ = agg_occ.permute(0, 4, 2, 1, 3) # (bs, num_classes, W, H, Z)
        return agg_occ

    # TODO: modify this
    def generate_output(self, pred_occ, img_metas):
        import open3d as o3d
        
        color_map = np.array(
                [
                    [0, 0, 0, 255],
                    [255, 120, 50, 255],    # barrier              orangey
                    [255, 192, 203, 255],   # bicycle              pink
                    [255, 255, 0, 255],     # bus                  yellow
                    [0, 150, 245, 255],     # car                  blue
                    [0, 255, 255, 255],     # construction_vehicle cyan
                    [200, 180, 0, 255],     # motorcycle           dark orange
                    [255, 0, 0, 255],       # pedestrian           red
                    [255, 240, 150, 255],   # traffic_cone         light yellow
                    [135, 60, 0, 255],      # trailer              brown
                    [160, 32, 240, 255],    # truck                purple
                    [255, 0, 255, 255],     # driveable_surface    dark pink
                    [139, 137, 137, 255],   # other_flat           dark red
                    [75, 0, 75, 255],       # sidewalk             dard purple
                    [150, 240, 80, 255],    # terrain              light green
                    [230, 230, 250, 255],   # manmade              white
                    [0, 175, 0, 255],       # vegetation           green
                ]
            )
        
        if self.use_semantic:
            _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        else:
            voxel = torch.sigmoid(pred_occ[:, 0])
        
        for i in range(voxel.shape[0]):
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2])
            X, Y, Z = torch.meshgrid(x, y, z)
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)
        
            vertices = vv[voxel[i] > 0.5]
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) /  img_metas[i]['occ_size'][0]  + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) /  img_metas[i]['occ_size'][1]  + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) /  img_metas[i]['occ_size'][2]  + img_metas[i]['pc_range'][2]
            
            vertices = vertices.cpu().numpy()
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if self.use_semantic:
                semantics = voxel[i][voxel[i] > 0].cpu().numpy()
                color = color_map[semantics] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
                vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)
    
            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)


            o3d.io.write_point_cloud(os.path.join(save_dir, 'pred.ply'), pcd)
            np.save(os.path.join(save_dir, 'pred.npy'), vertices)
            for cam_name in img_metas[i]['cams']:
                os.system('cp {} {}/{}.jpg'.format(img_metas[i]['cams'][cam_name]['data_path'], save_dir, cam_name))
