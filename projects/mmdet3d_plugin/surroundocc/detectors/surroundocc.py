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
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from torchvision.transforms.functional import rotate


@DETECTORS.register_module()
class SurroundOcc(MVXTwoStageDetector):
    def __init__(self,
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
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 len_queue=4,
                 is_vis=False,
                 version='v1',
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

        self.use_semantic = use_semantic
        self.is_vis = is_vis

        self.len_queue = len_queue
        # only save `len_queue` previous occ preds
        self.prev_occ_list = []
        # TODO: for now, only support one gpu with batch size = 1
        self.curr_scene_token = None

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

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          voxel_semantics,
                          mask_camera,
                          img_metas):

        outs = self.pts_bbox_head(
            pts_feats, img_metas)
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
                      voxel_semantics=None,
                      mask_camera=None,
                      ):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, voxel_semantics, mask_camera,
                                             img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, voxel_semantics=None, **kwargs):
        # new scene
        # TODO: for now, only support one gpu with batch size = 1
        if self.curr_scene_token is None or img_metas[0][-1]['scene_token'] != self.curr_scene_token:
            self.prev_occ_list = []
            self.curr_scene_token = img_metas[0][-1]['scene_token']

        # get the occ pred of current frame
        output = self.simple_test(
            [img_metas[0][-1]], img, **kwargs)
        
        pred_occ = output['occ_preds']
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
        outs = self.pts_bbox_head(x, img_metas)

        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

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
        assert bs == 1
        agg_occ = [target_occ.permute(0, 3, 2, 4, 1)] # (1, bs, H, W, Z, num_classes)
        for img_metas, prev_occ in zip(img_metas_list[:-1], prev_occ_list):
            for i in range(bs):
                # rotate the prev_occ
                rotation_angle = img_metas[i]['can_bux'][-1]
                prev_occ[i, :] = prev_occ[i, :].permute(2, 1, 3, 0).reshape(H, W, -1) # (H, W, Z*num_classes)
                prev_occ[i, :] = rotate(prev_occ, rotation_angle, center=[H // 2, W // 2])
                prev_occ[i, :] = prev_occ[i, :].reshape(H, W, Z, num_classes)
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
