"""
This script downloads all the corresponding KITTI raw data as outlined in
KITTI SceneFlow 2015's train_mapping.txt, so that the corresponding raw lidar
data could be matched with the color images as well as the ground truth
disparities in the KITTI 141 evaluation dataset.
"""

import wget

namelist = []

with open("train_mapping.txt", 'r') as f:
    lines = f.readlines()

count = 0

for line in lines:
    splitLine = line.split(" ")
    if len(splitLine) == 3:
        count += 1
        if splitLine[1] not in namelist:
            namelist.append(splitLine[1])

for i, name in enumerate(namelist):
    if (i < 11): continue
    print("Downloading %s of %s" % (i+1, len(namelist)))
    short_name = name[0:21]
    url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{}/{}.zip".format(short_name, name)
    wget.download(url)

