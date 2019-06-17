'''
This script draws an example output from a trained LidarStereoNet model and
saves it to a png file.
'''

import glob
import numpy as np
import cv2 as cv
import torch
from torch.autograd import Variable
import torch.utils.data
from PIL import Image
from models.lidarstereonet import *
from kitti_loader_eval import *
import warnings
#from tensorboardX import SummaryWriter
import time
from tensorboard import Tensorboard
import sys
warnings.simplefilter("ignore", category=UserWarning)
import matplotlib.pyplot as plt



dataDir = "/home/project/test_data/"
baseDir = "/home/project/"

#imgLList = glob.glob(baseDir + "data_odometry_color/dataset/sequences/**/image_2/*.png")
#imgRList = glob.glob(baseDir + "data_odometry_color/dataset/sequences/**/image_3/*.png")
#lidarLList = glob.glob(baseDir + "data_odometry_projlidar/dataset/sequences/**/image_2/*.png")
#lidarRList = glob.glob(baseDir + "data_odometry_projlidar/dataset/sequences/**/image_3/*.png")
imgLList = glob.glob(dataDir + "*[0-9]_l.png")
imgRList = glob.glob(dataDir + "*[0-9]_r.png")
lidarLList = glob.glob(dataDir + "*projlidar_l.png")
lidarRList = glob.glob(dataDir + "*projlidar_r.png")
gtList = glob.glob(dataDir + "*_gt.png")
checkpointList = glob.glob(baseDir + "final.checkpoint")

imgLList.sort()
imgRList.sort()
lidarLList.sort()
lidarRList.sort()
checkpointList.sort()
print(checkpointList)
#print(len(imgLList))
#print(len(imgRList))
#print(len(lidarLList))
#print(len(lidarRList))
#print(len(checkpointList))
#print(imgLList)
#print(lidarLList)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.cuda.manual_seed_all(999)

    print("Starting LidarStereoNet eval...")
    model = LidarStereoNet(device=device).to(device)

    dataloader = torch.utils.data.DataLoader(
         KittiLoader(imgLList, imgRList, lidarLList, lidarRList, ground_truth=gtList, training=False), batch_size = 1, shuffle=False, num_workers= 8, drop_last=False)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     left_img, right_img, lidar_left_img, lidar_right_img = sample_batched
    #     a = left_img.squeeze(dim=0).numpy()
    #     b = right_img.squeeze(dim=0).numpy()
    #     c = lidar_left_img.squeeze(dim=0).numpy()
    #     d = lidar_right_img.squeeze(dim=0).numpy()

    #     # print(a.shape)
    #     # cv.imshow('abc', np.transpose(a, (1,2,0)).astype(np.uint8))
    #     # cv.waitKey(0)

    tau0 = [10.0, 15.0, 20.0]
    tau1 = 0.05

    for i_path, path in enumerate(checkpointList):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optim_dict'])
        epoch = checkpoint['epoch']
        model.eval()

        print('Checkpoint', str(i_path), ' : ', str(path), 'starting...')

        for i_batch, batch in enumerate(dataloader):
            # prevTime = time.time()
            imgL, imgR, lidarL, lidarR, ground_truth, orig_left_img  = batch
            disp_updateL, disp_updateR, _, _, _, _ = \
                model(imgL.to(device), imgR.to(device), lidarL.to(device), lidarR.to(device))

            dist = lidarL.detach().squeeze(dim=0)
            dist = dist.squeeze(dim=0).cpu().numpy()
            img = orig_left_img.squeeze(dim=0).cpu().numpy()
#            img = np.transpose(img, (1,2,0))
            print(dist.shape)
            print(img.shape)

            #beginning of drawing code
            c = (np.max(dist) - dist) / (np.max(dist)-np.min(dist)) * 255
            print(c.shape)

            imgplot = plt.imshow(img,cmap='gray',alpha=0.8)
            plt.axis([0, img.shape[1], 0, img.shape[0]])
            plt.grid(False)                         # set the grid
            cm = plt.get_cmap("hot")
            c_ = np.zeros((img.shape[0]*img.shape[1]))
            pts = np.zeros((img.shape[0]*img.shape[1], 2))
            print(c[0,0])
            print(c_.shape)
            print(img.shape[0])
            print(img.shape[1])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if c[i,j] > 1:
                   #     plt.scatter(i, j, s=0.1, marker='o', c=c[i,j], cmap=cm)
                        c_[i*img.shape[1]+j] = c[i,j]
                        pts[i*img.shape[1]+j,0] = j
                        pts[i*img.shape[1]+j,1] = i
            plt.scatter(pts[:,0],pts[:,1],s=0.1,marker='o',alpha=0.18, c=c_, cmap=cm)
            ax=plt.gca()                            # get the axis
            ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
            ax.xaxis.tick_top()                     # and move the X-Axis
            #ax.yaxis.set_ticks(np.arange(0, 375, 1)) # set y-ticks
            ax.yaxis.tick_left()
            plt.savefig(baseDir + 'foo7.png')
            plt.show()
            #end of drawing code

            input("stop")
            sys.stdout.flush()

if __name__ == '__main__':
        main()
