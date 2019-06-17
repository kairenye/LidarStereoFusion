'''
This file is the main script for training LidarStereoNet on a GPU-enabled
instance.
'''

import glob
import numpy as np
import cv2 as cv
import torch
from torch.autograd import Variable
import torch.utils.data
from PIL import Image
from models.lidarstereonet import *
from kitti_loader import *
import warnings
#from tensorboardX import SummaryWriter
import time
from tensorboard import Tensorboard
import sys
warnings.simplefilter("ignore", category=UserWarning)


baseDir = "/home/project/"

imgLList = glob.glob(baseDir + "data_odometry_color/dataset/sequences/**/image_2/*.png")
imgRList = glob.glob(baseDir + "data_odometry_color/dataset/sequences/**/image_3/*.png")
lidarLList = glob.glob(baseDir + "data_odometry_projlidar/dataset/sequences/**/image_2/*.png")
lidarRList = glob.glob(baseDir + "data_odometry_projlidar/dataset/sequences/**/image_3/*.png")

imgLList.sort()
imgRList.sort()
lidarLList.sort()
lidarRList.sort()

learning_rate = 0.001
num_epochs = 5


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        m.bias.data.zero_()

#truncated L2 loss function. refer to LidarStereoNet paper for details
def loss_trunc_L2(input, target, mask, eps=3, size_average=True):
        num_batches = input.size(0)
        diff = (input - target) * mask
        clamped_input = torch.clamp(diff, min=-eps, max=eps)
        loss = 0.5 * torch.sum(clamped_input ** 2)
        if size_average:
           return loss/num_batches
        return loss

def main():
     # for epoch in range(1, args.epochs + 1):
     #         total_train_loss = 0
     #         print("This is the %d-th epoch" %(epoch))
     #    for batch_idx, (imgL, imgR, lidar_left, lidar_right) in enumerate(TrainImgLoader):
     #            loss = train(imgL, imgR, lidar_left, lidar_right)
     #            total_train_loss += loss
     #    total_train_loss /= len(TrainImgLoader)
     #    print('epoch %d total training loss is %.3f' &(epoch, total_train_loss))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.cuda.manual_seed_all(999)


    print("Starting LidarStereoNet training...")
    model = LidarStereoNet(device=device).to(device)
    model.apply(weight_init)

    dataloader = torch.utils.data.DataLoader(
         KittiLoader(imgLList, imgRList, lidarLList, lidarRList, True), batch_size = 1, shuffle=True, num_workers= 8, drop_last=False)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     left_img, right_img, lidar_left_img, lidar_right_img = sample_batched
    #     a = left_img.squeeze(dim=0).numpy()
    #     b = right_img.squeeze(dim=0).numpy()
    #     c = lidar_left_img.squeeze(dim=0).numpy()
    #     d = lidar_right_img.squeeze(dim=0).numpy()

    #     # print(a.shape)
    #     # print(b.shape)
    #     # print(c.shape)
    #     # print(d.shape)
    #     # print(np.max(c))
    #     # print(np.max(d))
    #     # cv.imshow('abc', np.transpose(a, (1,2,0)).astype(np.uint8))
    #     # cv.waitKey(0)
    #     # cv.imshow('abc', np.transpose(b, (1,2,0)).astype(np.uint8))
    #     # cv.waitKey(0)
    #     # cv.imshow('abc', np.transpose(c, (1,2,0)).astype(np.uint8))
    #     # cv.waitKey(0)
    #     # cv.imshow('abc', np.transpose(d, (1,2,0)).astype(np.uint8))
    #     # cv.waitKey(0)

    #     input("")
    # data1 = torch.randn((2,3,256,512)).to(device)
    # data2 = torch.randn((2,3,256,512)).to(device)
    # data3 = torch.randn((2,1,256,512)).to(device)
    # data4 = torch.randn((2,1,256,512)).to(device)
    # disp_updateL, disp_updateR, lidar_cleanL, lidar_cleanR, maskL, maskR = model(data1, data2, data3, data4)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter_count = 0

    #writer = SummaryWriter(logdir='log/')
    tensorboard = Tensorboard('logs')

    #main training loop
    for epoch in range(num_epochs):

        print("Epoch %d/%d" % (epoch+1, num_epochs))

        for i_batch, batch in enumerate(dataloader):
            #prevTime = time.time()
            imgL, imgR, lidarL, lidarR = batch
            disp_updateL, disp_updateR, lidar_cleanL, lidar_cleanR, maskL, maskR = \
                model(imgL.to(device), imgR.to(device), lidarL.to(device), lidarR.to(device))
            #referred to PSMNet code for the loss weightings
            loss = 0.5*loss_trunc_L2(disp_updateL[0], lidar_cleanL, maskL) \
                    + 0.7*loss_trunc_L2(disp_updateL[1], lidar_cleanL, maskL) \
                    + 1*loss_trunc_L2(disp_updateL[2], lidar_cleanL, maskL) \
                    + 0.5*loss_trunc_L2(disp_updateR[0], lidar_cleanR, maskR) \
                    + 0.7*loss_trunc_L2(disp_updateR[1], lidar_cleanR, maskR) \
                    + 1*loss_trunc_L2(disp_updateR[2], lidar_cleanR, maskR) \

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_count % 450 == 0:
                fn = 'checkpoint_models/' + str(epoch) + '_' + str(i_batch)+ '.checkpoint'
                state = {'epoch': epoch,'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()}
                torch.save(state, fn)

            print("Epoch: %d, Iter: %d, Batch: %d, Loss: %d" % (epoch+1, iter_count, i_batch, loss.item()))
            tensorboard.log_scalar('Loss', loss.item(), iter_count)
            iter_count = iter_count + 1
            #print(time.time()-prevTime)
            #writer.close()
            sys.stdout.flush()


    fn = 'checkpoint_models/final.checkpoint'
    state = {'epoch': num_epochs-1,'state_dict': model.state_dict(),
                'optim_dict' : optimizer.state_dict()}
    torch.save(state, fn)
    sys.stdout.flush()

if __name__ == '__main__':
        main()
