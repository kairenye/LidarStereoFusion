"""
This file implements the classes that implement the entire LidarStereoNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from importlib.machinery import SourceFileLoader
from models.feature_extractor import *
import numpy as np
import math
import time

"""
This class defines the disparity regression forward module at the end of the
network, where the dimensionality of the 3D volumes are reduced to produce
the final disparity outputs. See PSMNet paper for details. Credit: PSMNet code
"""
class DisparityRegression(nn.Module):
    def __init__(self, maxdisp, device):
        super(DisparityRegression, self).__init__()
        #support cuda here
        self.disp = Variable(torch.reshape(torch.arange(maxdisp), (1,maxdisp,1,1)), requires_grad=False).type(torch.cuda.FloatTensor)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

"""
This class defines the Hourglass component of the network, which serves as
feature matching according to the LidarStereoNet paper. Credit: PSMNet code
"""
class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x, presqu, postsqu):

        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True)

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post


"""
This class defines the core LidarStereoNet architecture without taking into
account the verify-update flow and the feature extraction parts of the network.
This module is used by the subsequent class LidarStereoNet to achieve
feature matching and disparity regression.
"""
class LidarStereoCoreNet(nn.Module):
    def __init__(self, device, maxdisp=192):
        super(LidarStereoCoreNet, self).__init__()
        self.maxdisp = maxdisp
        self.device = device
        self.dres0 = nn.Sequential(convbn_3d(96, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1= nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, featL, featR, H, W, verify):

        cost = Variable(torch.cuda.FloatTensor(featL.size()[0], featL.size()[1]*2, int(self.maxdisp/4),  featL.size()[2],  featL.size()[3]).zero_(), volatile= not self.training)

        for i in range(int(self.maxdisp/4)):
            if i > 0 :
             cost[:, :featL.size()[1], i, :,i:]   = featL[:,:,:,i:]
             cost[:, featL.size()[1]:, i, :,i:] = featR[:,:,:,:-i]
            else:
             cost[:, :featL.size()[1], i, :,:]   = featL
             cost[:, featL.size()[1]:, i, :,:]   = featR

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training and (not verify):
            cost1 = F.upsample(cost1, [self.maxdisp, H, W], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, H, W], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = DisparityRegression(self.maxdisp, self.device)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = DisparityRegression(self.maxdisp, self.device)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp, H, W], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
    #For your information: This formulation 'softmax(c)' learned "similarity"
    #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
    #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = DisparityRegression(self.maxdisp, self.device)(pred3)

        if self.training and (not verify):
            return pred1, pred2, pred3
        else:
            return pred3

"""
This class defines the final LidarStereoNet architecture by taking into
account the verify-update flow and the feature extraction parts of the network.
This module calls the LidarStereoCoreNet to achieve feature matching
and disparity regression.
"""
class LidarStereoNet(nn.Module):
    def __init__(self, device):
        super(LidarStereoNet, self).__init__()
        self.corenet = LidarStereoCoreNet(device=device)
        self.stereo_extractor = StereoFeatureExtractor()
        self.lidar_extractor = LidarFeatureExtractor()
        self.device = device

    def forward(self, imgL, imgR, lidarL, lidarR):
        H = imgL.size()[2]
        W = imgL.size()[3]
        imgL_f = self.stereo_extractor(imgL)
        imgR_f = self.stereo_extractor(imgR)
        #with torch.no_grad():
        #    lidarL_f = self.lidar_extractor(lidarL)
        #    lidarR_f = self.lidar_extractor(lidarR)

        #    featL = torch.cat((imgL_f, lidarL_f), dim=1)
        #    featR = torch.cat((imgR_f, lidarR_f), dim=1)

        #    disp_verifyL = self.corenet(featL, featR, H, W, verify=True)
        #    disp_verifyL = torch.unsqueeze(disp_verifyL, dim=1)

        #    disp_verifyR = self.corenet(featR, featL, H, W, verify=True)
        #    disp_verifyR = torch.unsqueeze(disp_verifyR, dim=1)


        #Use this mask if you have implemented all the loss functions
        #maskL = torch.gt((disp_verifyL - lidarL), 3).type(torch.cuda.FloatTensor)
        #maskR = torch.gt((disp_verifyR - lidarR), 3).type(torch.cuda.FloatTensor)


        #Use this mask if you're only using the lidar loss. Confirmed with
        # author of the paper that this is the right behavior for the mask.
        maskL = torch.gt(lidarL, 0).type(torch.cuda.FloatTensor)
        maskR = torch.gt(lidarR, 0).type(torch.cuda.FloatTensor)

        lidar_cleanL = lidarL * maskL
        lidar_cleanR = lidarR * maskR

        # print(lidar_cleanL.requires_grad)
        # print(lidar_cleanR.requires_grad)

        # print(lidar_cleanL.size())
        # print(lidar_cleanR.size())
        # print(maskL.size())
        # print(maskR.size())
        lidar_clean_L_f = self.lidar_extractor(lidar_cleanL)
        lidar_clean_R_f = self.lidar_extractor(lidar_cleanR)

        # print(lidar_clean_L_f.requires_grad)
        # print(lidar_clean_R_f.requires_grad)

        feat_cleanL = torch.cat((imgL_f, lidar_clean_L_f), dim=1)
        feat_cleanR = torch.cat((imgR_f, lidar_clean_R_f), dim=1)

        disp_updateL = self.corenet(feat_cleanL, feat_cleanR, H, W, verify=False)
        disp_updateR = self.corenet(feat_cleanR, feat_cleanL, H, W, verify=False)

        return disp_updateL, disp_updateR, lidar_cleanL, lidar_cleanR, maskL, maskR




