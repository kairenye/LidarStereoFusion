'''
This script evaluates a trained model's performance on KITTI's SceneFlow
performance metrics. Note that since during evaluation we have access to
ground truth data, the dataloader is different from that used during training.
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
checkpointList = glob.glob(baseDir + "0_****.checkpoint")

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

    #this is a dataloader written just for eval part
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

    #Refer to KITTI SceneFlow 2015 devkit's MATLAB code for details
    #Basically, tau0 is your x-px thresholds, and tau1 is percentage threshold.
    tau0 = [10.0, 15.0, 20.0]
    tau1 = 0.05

    for i_path, path in enumerate(checkpointList):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optim_dict'])
        epoch = checkpoint['epoch']
        model.eval()

        print('Checkpoint', str(i_path), ' : ', str(path), 'starting...')
        cumErr = np.array([0.0, 0.0, 0.0])
        cumRel_err = 0.0
        cumDelta = 0.0
        cumDensity = 0.0
        iter_count = 0

        for i_batch, batch in enumerate(dataloader):
            # prevTime = time.time()
            imgL, imgR, lidarL, lidarR, ground_truth, _ = batch
            disp_updateL, disp_updateR, _, _, _, _ = \
                model(imgL.to(device), imgR.to(device), lidarL.to(device), lidarR.to(device))

            #neg_ones = (torch.ones_like(predL_disp) * -1).to(device)
            #zeros = torch.zeros_like(predL_disp).to(device)
            #predL_disp = torch.where(disp_updateL > 0, disp_updateL, neg_ones).to(device)
            #gt_disp = torch.where(ground_truth > 0, ground_truth, neg_ones).to(device)
            #n_valid = torch.nonzero(torch.where(gt_disp > 0, gt_disp, zeros)).size()

            estDisp = disp_updateL.detach().squeeze(dim=0).cpu().numpy()
            gtDisp = ground_truth.squeeze(dim=0).numpy()

            estDisp[estDisp == 0.0] = -1
            gtDisp[gtDisp == 0.0] = -1

            n_total = np.count_nonzero(gtDisp > 0)

            iter_count += 1

            E = np.abs(gtDisp - estDisp)
            d_err = np.array([0.0,0.0,0.0])
            for t in range(3):
                d_err[t] = np.count_nonzero((gtDisp > 0) * (E>tau0[t]) * ((E/gtDisp) > tau1))/n_total * 100.0
            rel_err = np.sum((gtDisp > 0) * (E/gtDisp))/n_total * 100.0
            delta1 = np.count_nonzero(((gtDisp > 0) * (estDisp > 0) * (np.maximum(estDisp/gtDisp, gtDisp/estDisp) < 1.25)))/np.count_nonzero( (gtDisp > 0) * (estDisp > 0)) * 100.0
            density = np.count_nonzero(estDisp > 0)/(estDisp.shape[0] * estDisp.shape[1]) * 100.0
            cumErr += d_err
            cumRel_err += rel_err
            cumDelta += delta1
            cumDensity += density
            print("10px: {}, 15px: {}, 20px: {}, abs rel: {}%, delta: {}, density: {}, cum2px: {}, cum3px: {}, cum5px: {}, cumRel_Err: {}, cumDelta: {}, cumDensity: {}".format(d_err[0], d_err[1], d_err[2], rel_err, delta1, density, cumErr[0], cumErr[1], cumErr[2], cumRel_err, cumDelta, cumDensity))


    #        print("Epoch: %d, Iter: %d, Batch: %d, Loss: %d" % (epoch+1, iter_count, i_batch, loss.item()))
            # print(time.time()-prevTime)
            sys.stdout.flush()

        print('Checkpoint: ', i_path)
        print('Num iters: ', iter_count)
        print('Final disp error: ', cumErr/float(iter_count))
        print('Final relative absolute error: ', cumRel_err/float(iter_count))
        print('Final delta: ', cumDelta/float(iter_count))
        print('Final density: ', cumDensity/float(iter_count))

if __name__ == '__main__':
        main()
