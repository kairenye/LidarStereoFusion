"""
This script pulls the KITTI 141 evaluation set's ground_truth files
into a folder called lidarstereonet_test and renames the png file to end
with "_gt".
"""

import glob
import shutil
from scipy import io as sio
import numpy as np


baseDir = "/Users/kairenye/Desktop/lidarstereo_data/"

fileList = glob.glob(baseDir + "data_scene_flow/training/disp_occ_0/*_10.png")

fileList.sort()

namelist = []

print(fileList)

with open("train_mapping.txt", 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    splitLine = line.split(" ")
    if len(splitLine) == 3:
        print("At item %s of %s" % (i+1, len(lines)))
        print(fileList[i])
        gt_src = fileList[i]
        gt_dst = baseDir + "lidarstereonet_test/" + fileList[i].split("/")[-1].replace("_10", "_gt")

        shutil.copyfile(gt_src, gt_dst)
