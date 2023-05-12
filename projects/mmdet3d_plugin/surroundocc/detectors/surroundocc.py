# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np

from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import build_neck, build_backbone
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class SurroundOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
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
        self.img_view_transformer = build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)

        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.use_semantic = use_semantic
        self.is_vis = is_vis

        # BEVDet4D
        num_adj = 1
        self.num_frame = num_adj + 1
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames
        # TODO
        self.with_prev = False
        self.align_after_view_transfromation = False
                  
    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        # TODO: bda?
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor                

    def encode_image(self, img, img_metas, len_queue=None):
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

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x = self.encode_image(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        return bev_feat, depth

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation: # False
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation: # False
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats, depth = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats, depth

    def forward_pts_train(self,
                          pts_feats,
                          voxel_semantics,
                          mask_camera,
                          img_metas):
        losses = dict()
        occ_pred = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [voxel_semantics, mask_camera, occ_pred]
        losses_occ = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        losses.update(losses_occ)

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
                      depth_gt=None,
                      ):

        # extract image features
        img_feats, depth = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        # depth branch
        losses_depth = self.img_view_transformer.get_depth_loss(depth_gt, depth)
        losses.update(losses_depth)

        # occ branch
        # TODO: add feature fuser
        losses_occ = self.forward_pts_train(img_feats, voxel_semantics, mask_camera, img_metas)
        losses.update(losses_occ)
        return losses

    def forward_test(self, img_metas, img=None, voxel_semantics=None, **kwargs):
        
        output = self.simple_test(
            img_metas, img, **kwargs)
        
        pred_occ = output['occ_preds']

        # `pred_occ` got multi-scale pred results
        # Here we only use the last one with shape of (200, 200, 16)
        # for evalution with ground truth
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
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

        bbox_list = [dict() for i in range(len(img_metas))]
        output = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

        return output

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
