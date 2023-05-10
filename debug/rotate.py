import cv2
import torch
from torchvision.transforms.functional import rotate, affine

img = cv2.imread('/home/zonlin/Openmmlab/Occupancy/Occupancy3D/debug/n015-2018-07-18-11-07-57+0800__CAM_FRONT_RIGHT__1531883530420339.jpg')
cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), radius=1, color=(255, 255, 100), thickness=2)
img = affine(torch.from_numpy(img).permute(2, 0, 1), angle=70, center=(img.shape[1]//2, img.shape[0]//2),
             translate=(800, 450), scale=1., shear=0., fill=(255,)).permute(1, 2, 0).numpy()

cv2.imshow('cam_back', img)
cv2.waitKey()