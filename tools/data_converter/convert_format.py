# ---------------------------------------------
#  Modified by Xiuyu Yang (uncompleted)
# ---------------------------------------------


import os
import pickle
import numpy as np

version = 'nuScenes_Occupancy_v0.1'
data_path = './data/nuscenes/'
train_split = 'nuscenes_occ_infos_train.pkl'
val_split = 'nuscenes_occ_infos_val.pkl'
occ_size = [200, 200, 16]
original_occ_size = [512, 512, 40]

def main():
    train_split_path = os.path.join(data_path, train_split)
    convert(train_split_path)
    val_split_path = os.path.join(data_path, val_split)
    convert(val_split_path)


def convert(split_path):
    original_occ = np.zeros(original_occ_size, dtype=np.float32)
    original_occ += 17
    
    with open(split_path) as f:
        split = pickle.load(f)
        infos = split['infos']
        metadata = split['metadata']

        print(f"Converting {len(infos)} files ...")
        for info in infos:
            lidar_token = info['lidar_token']
            scene_token = info['scene_token']
            occ_gt_path = os.path.join(data_path, version, f'scene_{scene_token}', 'occupancy', lidar_token + '.npy')

            # load grount truth
            occ_gt = np.load(occ_gt_path)
            print(f"scene-{scene_token} lidar-{lidar_token} shape: {occ_gt.shape}")
            original_occ[occ_gt[:, 1], occ_gt[:, 2], occ_gt[:, 0]] = occ_gt[:, 3]
            save_path = os.path.join(data_path, f'{version}_converted', f'scene_{scene_token}', 'occupancy', lidar_token + '.npy')

            # dowansample
            occ_gt = original_occ[56:456, 56:456, 20:]  # 400, 400, 20
            downsample(occ_gt)                          # 200, 200, 10
            np.save(save_path, occ_gt)


def downsample(occ):
    pass