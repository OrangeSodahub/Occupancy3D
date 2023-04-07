import os
import cv2
import pickle
import warnings
import random
import numpy as np
from PIL import Image
import PIL.Image as pil

import mmcv
import torch
import torch.utils.data as data
from torchvision import transforms
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from mmdet.datasets import DATASETS


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


@DATASETS.register_module()
class NuscDepthDataset(data.Dataset):
    def __init__(self,
                 raw_data_root,
                 occ_data_root,
                 ann_file,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_sfm_spatial=False,
                 joint_pose=False,
                 use_fix_mask=False,
                 is_train=False,
                 file_client_args=dict(backend='disk')):

        super(NuscDepthDataset, self).__init__()


        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.use_sfm_spatial = use_sfm_spatial
        self.joint_pose = joint_pose
        self.use_fix_mask = use_fix_mask
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.split = 'train' if self.is_train else 'val'
        self.raw_data_path = raw_data_root
        self.occ_data_path = occ_data_root
        self.ann_file = ann_file
        version = 'v1.0-trainval'
        self.nusc = NuScenes(version=version,
                            dataroot=self.data_path, verbose=False)
        self.file_client = mmcv.FileClient(**file_client_args)

        # Here we do not use these two
        self.depth_path = None
        self.match_path = None

        with open('datasets/nusc/{}.txt'.format(self.split), 'r') as f:
            self.filenames = f.readlines()

        self.camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

        # load occ annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.occ_infos = self.load_annotations(self.ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='pkl')

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n + "_aug", im, -1)] = []

                for i in range(self.num_scales):
                    inputs[(n, im, i)] = []
                    inputs[(n + "_aug", im, i)] = []
                    #print(n, im, i)
                    for index_spatial in range(6):
                        inputs[(n, im, i)].append(self.resize[i](inputs[(n, im, i - 1)][index_spatial]))

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                for index_spatial in range(6):
                    aug = color_aug(f[index_spatial])
                    inputs[(n, im, i)][index_spatial] = self.to_tensor(f[index_spatial])
                    inputs[(n + "_aug", im, i)].append(self.to_tensor(aug))
                
                inputs[(n, im, i)] = torch.stack(inputs[(n, im, i)], dim=0)
                inputs[(n + "_aug", im, i)] = torch.stack(inputs[(n + "_aug", im, i)], dim=0)

    def __len__(self):
        return len(self.filenames)
        #return self.num_frames

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and (not self.use_sfm_spatial) and (not self.joint_pose) and random.random() > 0.5

        frame_index = self.filenames[index].strip().split()[0]
        self.get_info(inputs, frame_index, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        if not self.is_train:
            self.frame_idxs = [0]

        for scale in range(self.num_scales):
            for frame_id in  self.frame_idxs:
                inputs[("K", frame_id, scale)] = []
                inputs[("inv_K", frame_id, scale)] = []
    
        for index_spatial in range(6):
            for scale in range(self.num_scales):
                for frame_id in  self.frame_idxs:
                    K = inputs[('K_ori', frame_id)][index_spatial].copy()
        
                    K[0, :] *= (self.width // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K[1, :] *= (self.height // (2 ** scale)) / inputs['height_ori'][index_spatial]
        
                    inv_K = np.linalg.pinv(K)
        
                    inputs[("K", frame_id, scale)].append(torch.from_numpy(K))
                    inputs[("inv_K", frame_id, scale)].append(torch.from_numpy(inv_K))
    
        for scale in range(self.num_scales):
            for frame_id in  self.frame_idxs:
                inputs[("K",frame_id, scale)] = torch.stack(inputs[("K",frame_id, scale)], dim=0)
                inputs[("inv_K",frame_id, scale)] = torch.stack(inputs[("inv_K", frame_id,scale)], dim=0)

        if do_color_aug:
            #color_aug = transforms.ColorJitter.get_params(
            #    self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        del inputs[("color", 0, -1)]
        if self.is_train:
            for i in self.frame_idxs[1:]:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]
            for i in self.frame_idxs:
                del inputs[('K_ori', i)]
        else:
            del inputs[('K_ori', 0)]
            
        del inputs['width_ori']
        del inputs['height_ori']

        if 'depth' in inputs.keys():
            inputs['depth'] = torch.from_numpy(inputs['depth'])

        if self.is_train:
            inputs["pose_spatial"] = torch.from_numpy(inputs["pose_spatial"])
            for i in self.frame_idxs[1:]:
                inputs[("pose_spatial", i)] = torch.from_numpy(inputs[("pose_spatial", i)])
                
            if self.use_sfm_spatial:
                for j in range(len(inputs['match_spatial'])):
                    inputs['match_spatial'][j] = torch.from_numpy(inputs['match_spatial'][j])
            
            if self.use_fix_mask:
                inputs["mask"] = []
                for i in range(6):
                    temp = cv2.resize(inputs["mask_ori"][i], (self.width, self.height))
                    temp = temp[..., 0]
                    temp = (temp == 0).astype(np.float32)
                    inputs["mask"].append(temp)
                inputs["mask"] = np.stack(inputs["mask"], axis=0)
                inputs["mask"] = np.tile(inputs["mask"][:, None], (1, 2, 1, 1))
                inputs["mask"] = torch.from_numpy(inputs["mask"])
                if do_flip:
                    inputs["mask"] = torch.flip(inputs["mask"], [3])
                del inputs["mask_ori"]         

        return inputs
    
    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        if self.is_train:
            if self.use_sfm_spatial:
                inputs["match_spatial"] = []

            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 
            
            inputs["pose_spatial"] = []
        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []


        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []

        rec = self.nusc.get('sample', index_temporal)

        for index_spatial in range(6):
            cam_sample = self.nusc.get(
                'sample_data', rec['data'][self.camera_names[index_spatial]])
            inputs['id'].append(self.camera_ids[index_spatial])
            color = self.loader(os.path.join(self.data_path, cam_sample['filename']))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])
            
            # TODO: remove this
            if not self.is_train:
                depth = np.load(os.path.join(self.depth_path, cam_sample['filename'][:-4] + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))
            
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)

            ego_spatial = self.nusc.get(
                    'calibrated_sensor', cam_sample['calibrated_sensor_token'])

            if self.is_train:
                pose_0_spatial = Quaternion(ego_spatial['rotation']).transformation_matrix
                pose_0_spatial[:3, 3] = np.array(ego_spatial['translation'])
            
                inputs["pose_spatial"].append(pose_0_spatial.astype(np.float32))
    
            K = np.eye(4).astype(np.float32)
            K[:3, :3] = ego_spatial['camera_intrinsic']
            inputs[('K_ori', 0)].append(K)
            if self.is_train:

                if self.use_sfm_spatial:
                    pkl_path = os.path.join(os.path.join(self.match_path, cam_sample['filename'][:-4] + '.pkl'))
                    with open(pkl_path, 'rb') as f:
                        match_spatial_pkl = pickle.load(f)
                    inputs['match_spatial'].append(match_spatial_pkl['result'].astype(np.float32))

                for idx, i in enumerate(self.frame_idxs[1:]):
                    if i == -1:
                        index_temporal_i = cam_sample['prev']
                    elif i == 1:
                        index_temporal_i = cam_sample['next']
                    cam_sample_i = self.nusc.get(
                        'sample_data', index_temporal_i)
                    ego_spatial_i = self.nusc.get(
                        'calibrated_sensor', cam_sample_i['calibrated_sensor_token'])

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = ego_spatial_i['camera_intrinsic']
                    inputs[('K_ori', i)].append(K)

                    color = self.loader(os.path.join(self.data_path, cam_sample_i['filename']))
                    
                    if do_flip:
                        color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
                    inputs[("color", i, -1)].append(color)

                    pose_i_spatial = Quaternion(ego_spatial_i['rotation']).transformation_matrix
                    pose_i_spatial[:3, 3] = np.array(ego_spatial_i['translation'])
    
        if self.is_train:
            for index_spatial in range(6):
                for idx, i in enumerate(self.frame_idxs[1:]):
                    pose_0_spatial = inputs["pose_spatial"][index_spatial]
                    pose_i_spatial = inputs["pose_spatial"][(index_spatial+i)%6]

                    gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                    inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0) 
                if i != 0:
                    inputs[("pose_spatial", i)] = np.stack(inputs[("pose_spatial", i)], axis=0)

            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)   
        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0) 
            inputs['depth'] = np.stack(inputs['depth'], axis=0)   

        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)

        # load occ info
        occ_gt_path = self.occ_infos[index_temporal]['occ_gt_path']
        occ_gt_path = os.path.join(self.occ_data_path, occ_gt_path)
        
        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']
        
        # Here class 0 is not the ignore class
        inputs['voxel_semantics'] = semantics
        inputs['mask_camera'] = mask_camera