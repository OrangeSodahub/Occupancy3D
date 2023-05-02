#import open3d as o3d
import os
import numpy as np
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type
from nuscenes.utils.geometry_utils import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile


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


def transform(pc, lidar2ego_trans, lidar2ego_rots):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar2ego_rots).rotation_matrix)
    pc.translate(np.array(lidar2ego_trans))
    return pc.points.T

@PIPELINES.register_module()
class LoadOccPointsFromFile(LoadPointsFromFile):
    def __init__(self, coord_type, load_dim=6, use_dim=[0, 1, 2], shift_height=False, use_color=False, file_client_args=dict(backend='disk')):
        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)

    def __call__(self, results):
        pts_filename = results['lidar_path']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        # transform points to ego vehicle coord
        points[:, :4] = transform(points[:, :4], results['lidar2ego_translation'], results['lidar2ego_rotation'])
        
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results
