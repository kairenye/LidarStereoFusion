"""
This script reads the train_mapping.txt file that details the images in the
KITTI 141 evaluation set and pulls the corresponding left and right images as
well as the raw lidar points from the KITTI raw dataset together into a
training data folder.
"""

import glob
import shutil
from scipy import io as sio
import numpy as np

curr_folder = "/Users/kairenye/Desktop/lidarstereo_dataset/"

def getCalibDict(calibDate):
    base_path = "/Users/kairenye/Desktop/lidarstereo_dataset/calib/" + calibDate + "_calib/"
    velo_to_cam_path = base_path + "calib_velo_to_cam.txt"
    cam_to_cam_path = base_path + "calib_cam_to_cam.txt"

    calib_velo_to_cam = {}
    calib_cam_to_cam = {}

    with open(velo_to_cam_path, 'r') as f:
        for line in f.readlines():
            if ':' in line and 'calib_time' not in line:
                key, value = line.split(':', 1)
                calib_velo_to_cam[key] = np.array([float(x) for x in value.split()])

    with open(cam_to_cam_path, 'r') as f:
        for line in f.readlines():
            if ':' in line and 'calib_time' not in line:
                key, value = line.split(':', 1)
                calib_cam_to_cam[key] = np.array([float(x) for x in value.split()])

    return calib_velo_to_cam, calib_cam_to_cam

def writeCalibFile(calib_velo_to_cam, calib_cam_to_cam, sequence):
    R_velo = calib_velo_to_cam['R']
    T_velo = calib_velo_to_cam['T']
    R_velo = R_velo.reshape(3,3)
    T_velo = T_velo.reshape(3,1)
    tmp = np.array([0,0,0,1])
    TR_velo_to_cam = np.vstack((np.hstack((R_velo,T_velo)), tmp))
    R = calib_cam_to_cam['R_rect_00']
    P = calib_cam_to_cam['P_rect_02']
    R = R.reshape(3,3)
    R = np.hstack((R,np.zeros((3,1))))
    R = np.vstack((R,np.zeros((1,4))))
    R[3,3] = 1
    P = P.reshape(3,4)
    outmat = {"TR_velo_to_cam": TR_velo_to_cam, "R": R, "P": P}
    outpath =  curr_folder + "lidarstereonet_test/" + fileList[i].split("/")[-1].replace("_10.png", "_calib.mat")
    sio.savemat(outpath, outmat)



fileList = glob.glob(curr_folder + 'data_scene_flow/training/image_2/*_10.png')

fileList.sort()

print(fileList)
namelist = []

with open("train_mapping.txt", 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    splitLine = line.split(" ")
    if len(splitLine) == 3:
        print("At item %s of %s" % (i+1, len(lines)))
        left_img_src = fileList[i]
        right_img_src = left_img_src.replace("image_2", "image_3")
        left_img_dst = curr_folder + "lidarstereonet_test/" + fileList[i].split("/")[-1].replace("_10", "_l")
        right_img_dst = curr_folder + "lidarstereonet_test/" + fileList[i].split("/")[-1].replace("_10", "_r")
        lidar_src = curr_folder + splitLine[0] + "/" + splitLine[1] + \
                    "/velodyne_points" + "/data/" + splitLine[2].rstrip('\r\n') + ".bin"
        lidar_dst = curr_folder + "lidarstereonet_test/" + fileList[i].split("/")[-1].replace("_10.png", ".bin")

        calib_velo_to_cam, calib_cam_to_cam = getCalibDict(splitLine[0])
        writeCalibFile(calib_velo_to_cam, calib_cam_to_cam, fileList[i].split("/")[-1].replace("_10.png", ""))

        shutil.copyfile(left_img_src, left_img_dst)
        shutil.copyfile(right_img_src, right_img_dst)
        shutil.copyfile(lidar_src, lidar_dst)
