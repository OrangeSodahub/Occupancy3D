#import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os


@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, data_root, use_semantic=True):
        self.use_semantic = use_semantic
        self.data_root = data_root
    
    def __call__(self, results):
        # Occ dataset offered by baseline contains:
        # ['semantics' ,'mask_lidar', 'mask_camera]
        # with shape of [200, 200, 16]
        occ_gt_path = results['occ_path']
        occ_gt_path = os.path.join(self.data_root, occ_gt_path)

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']
        
        semantics = occ_labels['semantics'].astype(np.float32)
        
        # class 0 is 'ignore' class
        if self.use_semantic:
            semantics[..., 3][semantics[..., 3] == 0] = 255
        else:
            semantics = semantics[semantics[..., 3] > 0]
            semantics[..., 3] = 1
        
        # results['gt_occ'] = semantics
        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

