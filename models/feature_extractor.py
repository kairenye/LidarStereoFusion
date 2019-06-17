"""
This file implements the classes that implement the feature extractors for both
stereo and lidar in the LidarStereoNet.
"""

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F

#Below are some nn.Sequential definitions that are used in the stereo feature
# extractor network. Credit: PSMNet source code
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

"""
The BasicBlock comprises the stereo feature extractor by stacking 2 convolution
layers and a downsample layer, if applicable.
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

"""
This class implements the entire stereo feature extractor network.
LidarStereoNet paper's network description is a bit vague and different from
the implementation here, but the paper mentions they refer to PSMNet. Therefore,
the PSMNet's implementation is shown here.
"""
class StereoFeatureExtractor(nn.Module):
    def __init__(self):
        super(StereoFeatureExtractor, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        import torch.nn.functional as F
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

"""
This class implements the sparsity-invariant convolution layer.
Credit: https://github.com/yxgeee/DepthComplete
"""
class SparseConv(nn.Module):
  # Convolution layer for sparse data
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    super(SparseConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
    self.if_bias = bias
    if self.if_bias:
      self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
    self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    #nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    self.pool.require_grad = False

  def forward(self, input):
    x, m = input
    mc = m.expand_as(x)
    x = x * mc
    #print(x.size())
    x = self.conv(x)

    weights = torch.ones_like(self.conv.weight)
    mc = F.conv2d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
    mc = torch.clamp(mc, min=1e-5)
    mc = 1. / mc
    x = x * mc
    if self.if_bias:
      x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
    m = self.pool(m)

    return x, m

"""
This class defines a block (with activation) of sparsity-invariant CNN block.
Credit: https://github.com/yxgeee/DepthComplete
"""
class SparseConvBlock(nn.Module):

  def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    super(SparseConvBlock, self).__init__()
    self.sparse_conv = SparseConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, input):
    x, m = input
    x, m = self.sparse_conv((x, m))
    assert (m.size(1)==1)
    x = self.relu(x)
    return x, m

"""
This class defines the entire lidar feature extractor network by utilizing the
sparsity-invariant convolution block defined above.
"""
class LidarFeatureExtractor(nn.Module):

  def __init__(self, in_channel=1, out_channel=16, kernels=[11,7,5,3,3], strides=[1,2,1,2,1,1],mid_channel=16):
    super(LidarFeatureExtractor, self).__init__()
    channel = in_channel
    convs = []
    for i in range(len(kernels)):
      assert (kernels[i]%2==1)
      convs += [SparseConvBlock(channel, mid_channel, kernels[i], stride=strides[i], padding=(kernels[i]-1)//2)]
      channel = mid_channel
    self.sparse_convs = nn.Sequential(*convs)
    self.mask_conv = nn.Conv2d(mid_channel+1, out_channel, 1)

  def forward(self, x):
    m = (x>0).detach().float()
    x, m = self.sparse_convs((x,m))
    x = torch.cat((x,m), dim=1)
    x = self.mask_conv(x)
    return x

