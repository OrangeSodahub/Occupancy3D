#import open3d as o3d
import os
import numpy as np
from mmdet.datasets.builder import PIPELINES


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
        
        # Here class 0 is not the ignore class
        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadDepthGT(object):
    """Load depth groundtruth.

    """

    def __init__(self, data_root):
        self.data_root = data_root
    
    def __call__(self, results):
        depth_gt_path = results['depth_path']
        depth_gt_path = os.path.join(self.data_root, depth_gt_path)
        depth_gt = np.load(depth_gt_path)

        results['depth_gt'] = depth_gt

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str