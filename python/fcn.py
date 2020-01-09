# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = models.vgg16()
        #if pretrained_net:
        vgg.load_state_dict(torch.load('/home/sun/桌面/ai_rbote/github_course'
                                       '/AI_rebot/FCN-pytorch-master/models/vgg16-397923af.pth'))
        # features list 31  内容：Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace).....
        # classifier list   内容 :
        # Linear(in_features=25088, out_features=4096, bias=True), ReLU(inplace), Dropout(p=0.5),
        # Linear(in_features=4096,.....)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        '''
        100 padding for 2 reasons:
            1) support very small input size
            # 允许裁剪以匹配不同图层的要素图的大小
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        # padding填充0
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x):
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()



#class FCN8s(nn.Module):

#    def __init__(self, pretrained_net, n_class):
#        super().__init__()
#        self.n_class = n_class
#        self.vgg=models.vgg16()
#        if(pretrained_net):
#            self.vgg.load_state_dict(torch.load('C:\\Users\\wen\\Desktop\\FCN-pytorch-master\\vgg16-397923af.pth'))
#        self.pretrained_net = pretrained_net
#        self.relu    = nn.ReLU(inplace=True)
#        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#        self.bn1     = nn.BatchNorm2d(512)
#        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#        self.bn2     = nn.BatchNorm2d(256)
#        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#        self.bn3     = nn.BatchNorm2d(128)
#        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#        self.bn4     = nn.BatchNorm2d(64)
#        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#        self.bn5     = nn.BatchNorm2d(32)
#        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

#    def forward(self, x):
#        output = self.pretrained_net(x)
        
#        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
#        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
#        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

#        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
#        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
#        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
#        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
#        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
#        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
#        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
#        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

#        return score  # size=(N, n_class, x.H/1, x.W/1)



#class VGGNet(VGG):
#    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
#        super().__init__(make_layers(cfg[model]))
#        self.ranges = ranges[model]

#        if pretrained:
#            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

#        if not requires_grad:
#            for param in super().parameters():
#                param.requires_grad = False

#        if remove_fc:  # delete redundant fully-connected layer params, can save memory
#            del self.classifier

#        if show_params:
#            for name, param in self.named_parameters():
#                print(name, param.size())

#    def forward(self, x):
#        output = {}

#        # get the output of each maxpooling layer (5 maxpool in VGG net)
#        for idx in range(len(self.ranges)):
#            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
#                x = self.features[layer](x)
#            output["x%d"%(idx+1)] = x

#        return output


#ranges = {
#    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
#    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
#    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
#    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
#}

## cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
#cfg = {
#    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

#def make_layers(cfg, batch_norm=False):
#    layers = []
#    in_channels = 3
#    for v in cfg:
#        if v == 'M':
#            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#        else:
#            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#            if batch_norm:
#                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#            else:
#                layers += [conv2d, nn.ReLU(inplace=True)]
#            in_channels = v
#    return nn.Sequential(*layers)

