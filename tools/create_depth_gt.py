import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
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

    return points, coloring


version = 'v1.0-trainval'
data_root = 'data/nuScenes'
info_path = 'data/occ3d-nus/occ_infos_temporal_train.pkl'

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]


def worker(info):
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    lidar_token = info['lidar_token']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = info['lidar_path']

    points = np.fromfile(os.path.join(data_root, lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]
    lidar_calibrated_sensor = nusc.get(
                'calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    for _, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = {
            'translation': info['cams'][cam_key]['sensor2ego_translation'],
            'rotation': info['cams'][cam_key]['sensor2ego_rotation'],
            'intrinsic': info['cams'][cam_key]['cam_intrinsic'],
        }
        cam_ego_pose = {
            'translation': info['cams'][cam_key]['ego2global_translation'],
            'rotation': info['cams'][cam_key]['ego2global_rotation']
        }
        img = mmcv.imread(
            os.path.join(data_root, info['cams'][cam_key]['data_path']))
        pts_img, depth = map_pointcloud_to_image(
            points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        save_path = os.path.join(data_root, 'depth_gt', f'{file_name}.bin')
        np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1
                        ).astype(np.float32).flatten().tofile(save_path)
        info['cams'][cam_key]['depth_gt_path'] = f'{file_name}.bin'


if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt'))

    # Here we add depth gt information to existing pkl files
    infos = mmcv.load(info_path)
    # import ipdb; ipdb.set_trace()
    for info in infos:
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()

    mmcv.dump(infos, info_path)
