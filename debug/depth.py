import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

points = np.loadtxt('/home/zonlin/Openmmlab/Occupancy3D/debug/coloring_CAM_BACK_RIGHT.txt')
img = cv2.imread('/home/zonlin/Openmmlab/Occupancy3D/debug/n015-2018-07-18-11-07-57+0800__CAM_BACK_RIGHT__1531883530427893.jpg')
points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2])) * 255
print(points[:, 2])

for point in tqdm(points):
    cv2.circle(img, (int(point[0]), int(point[1])), radius=1, color=(255, 255, int(point[2])), thickness=2)

cv2.imshow('cam_back', img)
cv2.waitKey()