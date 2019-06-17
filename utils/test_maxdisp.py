"""
This file tests what the maximum disparity is according to the lidar points
in the KITTI VO dataset.
"""

import numpy as np
import cv2 as cv
import glob

lidarList = glob.glob('/Volumes/EXT_DRIVE/lidarstereo_data/vo/data_odometry_projlidar/dataset/sequences/**/image_2/*.png')

maxDisp = 0


for i, file in enumerate(lidarList):
    print("At {}/{}".format(i+1, len(lidarList)))
    img = cv.imread(file, cv.IMREAD_UNCHANGED)
    img = img / 256
    currMax = np.max(img)
    if currMax > maxDisp: maxDisp = currMax
    print(maxDisp)

