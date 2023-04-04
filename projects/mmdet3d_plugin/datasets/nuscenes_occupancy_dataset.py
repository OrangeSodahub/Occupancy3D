
import torch
import random
import copy, os
import numpy as np
from tqdm import tqdm
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.parallel import DataContainer as DC
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.datasets.evaluation_metrics import Metric_mIoU, Metric_FScore


@DATASETS.register_module()
class CustomNuScenesOccDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, queue_length=4,use_semantic=False, classes=None, overlap_test=False, eval_fscore=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.queue_length = queue_length
        self.use_semantic = use_semantic
        self.class_names = classes
        self.eval_fscore = eval_fscore
        self._set_group_flag()
        
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_feat_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_feat_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            occ_path = info['occ_gt_path'],
            sample_idx = info['token'],
            frame_idx = info['frame_idx'],
            prev_idx = info['prev'],
            next_idx = info['next'],
            scene_token = info['scene_token'],
            can_bus = info['can_bus'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            timestamp = info['timestamp'] / 1e6,
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix

                if 'lidar2cam' in cam_info.keys():
                    lidar2cam_rt = cam_info['lidar2cam'].T
                else:
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # Transform ego vehicle to global
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        if show_dir is not None:
            if not os.path.exists(show_dir):
                os.mkdir(show_dir)
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin=eval_kwargs.get('begin',None)
            end=eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            # occ_size = [200, 200, 16]
            # pc_range = [-40, -40, -1.0, 40, 40, 5.4]
            occ_size=self.occ_size,
            pc_range=self.pc_range,
            use_lidar_mask=False,
            use_image_mask=True)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                # voxel_size => pc_range // occ_size
                voxel_size=[0.4, 0.4, 0.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        print('\nStarting Evaluation...')
        # `occ_results: (bs, H, W, Z)`
        for index, occ_pred in tqdm(enumerate(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
            if show_dir is not None:
                if begin is not None and end is not None:
                    if index>= begin and index<end:
                        sample_token = info['token']
                        save_path = os.path.join(show_dir,str(index).zfill(4))
                        np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
                else:
                    sample_token=info['token']
                    save_path=os.path.join(show_dir,str(index).zfill(4))
                    np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)

            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        results_dict = self.occ_eval_metrics.count_miou()
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore(results_dict)

        return results_dict