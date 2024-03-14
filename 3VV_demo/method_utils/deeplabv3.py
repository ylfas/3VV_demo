# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from method_utils.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from method_utils.aspp import ASPP, ASPP_Bottleneck
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        weight = torch.cat([avg_out, max_out], dim=1)
        weight = self.conv1(weight)  # 对池化完的数据cat 然后进行卷积
        return x*self.sigmoid(weight)


class DACblock_gai(nn.Module):
    def __init__(self, channel):
        super(DACblock_gai, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        self.conv3x3_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv3x3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv3x3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)

        self.conv_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)  # (1280 = 5*256)
        self.bn_conv_1 = nn.BatchNorm2d(256)

        self.bn_conv = nn.BatchNorm2d(512)
        self.conv_2 = nn.Conv2d(256, 4, kernel_size=1)
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.SA1(nonlinearity(self.dilate1(x)))
        dilate2_out = self.SA2(nonlinearity(self.conv3x3_1(nonlinearity(self.conv1x1(self.dilate2(x))) + dilate1_out)))
        dilate3_out = self.SA3(nonlinearity(self.conv3x3_2(nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x)))) + dilate2_out)))
        dilate4_out = self.SA4(nonlinearity(self.conv3x3_3(nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x))))) + dilate3_out)))
        dilate_out = torch.cat((dilate1_out, dilate2_out, dilate3_out, dilate4_out), 1)
        out = x + self.conv1x1_4(dilate_out)
        out = F.relu(self.bn_conv(out))
        out = F.relu(self.bn_conv_1(self.conv_1(out)))
        out = self.conv_2(out)
        # out = dilate1_out + dilate2_out +  dilate3_out + dilate4_out + x

        return out


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        # self.SA5 = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        dilate_out = torch.cat((dilate1_out, dilate2_out, dilate3_out, dilate4_out), 1)
        out = self.conv1x1_4(dilate_out)

        # out = self.conv_2(out)

        # out = dilate1_out + dilate2_out +  dilate3_out + dilate4_out + x
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3, self).__init__()

        self.num_classes = 4

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        # self.dblock = DACblock_gai(512)
        # self.spp = SPPblock(512)
        # self.SA1 = SpatialAttention()

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        # e4 = self.dblock(feature_map)
        # e4 = self.spp(feature_map)

        output = self.aspp(feature_map)# (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

class DeepLabV3_2cls(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3_2cls, self).__init__()

        self.num_classes = 1

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        # self.dblock = DACblock_gai(512)
        # self.spp = SPPblock(512)

        # self.SA1 = SpatialAttention()

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        # e4 = self.dblock(feature_map)
        # e4 = self.spp(feature_map)

        output = self.aspp(feature_map)# (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

class DeepLabV3_gai(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3_gai, self).__init__()

        self.num_classes = 4

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here

        self.dblock = DACblock_gai(512)
        self.SA1 = SpatialAttention()

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        e4 = self.dblock(feature_map)

        output = F.upsample(e4, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)