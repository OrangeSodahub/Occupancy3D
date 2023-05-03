import os
from tqdm import tqdm
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


data_root = 'data/nuScenes'
info_path = 'data/occ3d-nus/occ_infos_temporal_train.pkl'
save_root = 'data/depth_gt'

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]


# TODO: get downsampled depth gt
def transform_depth(depth, h, w):
    depth_map = np.zeros((h, w))
    depth_coords = depth[:, :2].astype(np.int16)
    valid_mask = ((depth_coords[:, 1] < h)
                  & (depth_coords[:, 0] < w)
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = depth[valid_mask, 2]
    return depth_map


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def generate_depth_map(
    pc,
    im,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    h, w,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.

    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    depth = np.concatenate([points[:2, :].T, coloring[:, None]], axis=1)

    return transform_depth(depth, h, w)


def worker(info):
    timestamp = info['timestamp']
    lidar_path = info['lidar_path']
    points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
    lidar_calibrated_sensor = {
        'translation': info['lidar2ego_translation'],
        'rotation': info['lidar2ego_rotation'],
    }
    lidar_ego_pose = {
        'translation': info['ego2global_translation'],
        'rotation': info['ego2global_rotation']
    }

    filename_list = []
    all_depth_map = []
    for _, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = {
            'translation': info['cams'][cam_key]['sensor2ego_translation'],
            'rotation': info['cams'][cam_key]['sensor2ego_rotation'],
            'camera_intrinsic': info['cams'][cam_key]['cam_intrinsic'],
        }
        cam_ego_pose = {
            'translation': info['cams'][cam_key]['ego2global_translation'],
            'rotation': info['cams'][cam_key]['ego2global_rotation']
        }
        img = mmcv.imread(info['cams'][cam_key]['data_path'])
        H, W, _ = img.shape
        depth_map = generate_depth_map(
            points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose, H, W)
        file_name = timestamp
        save_path = os.path.join(save_root, f'{file_name}.npy')
        all_depth_map.append(depth_map)
        filename_list.append(f'{file_name}.npy')
    np.save(save_path, np.array(all_depth_map).astype(np.float32))
    return filename_list


if __name__ == '__main__':
    mmcv.mkdir_or_exist(save_root)

    # Here we add depth gt information to existing pkl files
    # Need to create symbolic in /occ3d-nus/samples/LIDAR_TOP
    infos = mmcv.load(info_path)
    for info in tqdm(infos['infos']):
        depth_gt_path = worker(info)
        info.update(depth_gt_path=depth_gt_path)

    mmcv.dump(infos, info_path)
