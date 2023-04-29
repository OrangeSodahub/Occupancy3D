import cv2
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def show_projected_point_cloud():
    points = np.loadtxt('/home/zonlin/Openmmlab/Occupancy3D/debug/coloring_CAM_BACK_RIGHT.txt')
    img = cv2.imread('/home/zonlin/Openmmlab/Occupancy3D/debug/n015-2018-07-18-11-07-57+0800__CAM_BACK_RIGHT__1531883530427893.jpg')
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2])) * 255
    print(points[:, 2])

    for point in tqdm(points):
        cv2.circle(img, (int(point[0]), int(point[1])), radius=1, color=(255, 255, int(point[2])), thickness=2)

    cv2.imshow('cam_back', img)
    cv2.waitKey()


bs = 1
num_cam = 6

cam_intrinsic = [[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
                 [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,]]

cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
cam_intrinsic = cam_intrinsic[None, None, ...].repeat(bs, num_cam, 1, 1)
inv_intrins = torch.inverse(cam_intrinsic)
pixel_size = torch.norm(
    torch.stack(
        [inv_intrins[..., 0, 0], inv_intrins[..., 1, 1]], dim=-1
    ),
    dim=-1,
).reshape(-1, 1).float()

print(pixel_size.shape)