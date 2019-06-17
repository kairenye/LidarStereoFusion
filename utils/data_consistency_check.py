"""
This file makes sure that all the left training images and the corresponding
right training images have the same dimensions, because the network relies on
this assumption. (The check passes for the KITTI VO Dataset).
"""

import glob
import cv2 as cv

left_list = glob.glob('/Volumes/EXT_DRIVE/lidarstereo_data/vo/data_odometry_color/dataset/sequences/**/image_2/*.png')

right_list = glob.glob('/Volumes/EXT_DRIVE/lidarstereo_data/vo/data_odometry_color/dataset/sequences/**/image_3/*.png')

left_list.sort()
right_list.sort()

assert(len(left_list)==len(right_list))

for i in range(0, len(left_list)):
    print("%d/%d" % (i+1, len(left_list)))
    imL = cv.imread(left_list[i])
    imR = cv.imread(right_list[i])
    assert(imL.shape==imR.shape)