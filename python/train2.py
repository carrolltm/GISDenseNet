# -*- coding: utf-8 -*-

from __future__ import print_function
import  cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import warnings
from collections  import Counter

import collections
from fcn import  FCN8s
# from Cityscapes_loader import CityscapesDataset
from CamVid_loader import CamVidDataset

import numpy as np
import time
import sys
import os
n_class = 32


import os
print("Start")
print(os.getcwd())

# 一次多少张图片   一共630张图片
batch_size = 1    # 8都不行 GPU超出内存了
# 多少批      批量梯度下降计算全部训练集样本梯度的平均，然后更新梯度
epochs = 1
lr = 1e-4
# 防止训练过程中陷入局部最小值 ref:https://blog.csdn.net/u013989576/article/details/70241121
# 下面几个都是跟学习率有关
momentum = 0
w_decay = 1e-5
step_size = 50
# 批标准化
gamma = 0.5     # 方差？？？
configs = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(
    batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

if sys.argv[1] == 'CamVid':
    root_dir = "CamVid/"
else:
    root_dir = "CityScapes/"
train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
# GPU的数量
num_gpu = list(range(torch.cuda.device_count()))
train_data = None
if sys.argv[1] == 'CamVid':
    train_data = CamVidDataset(csv_file=train_file, phase='train')
# else:
#     train_data = CityscapesDataset(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
val_data = None
if sys.argv[1] == 'CamVid':
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
# else:
#     val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

# vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCN8s(num_classes=n_class)

if use_gpu:
    ts = time.time()
    # vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
# 学习率自动下降
# 根据时期数调整学习率的方法
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,  # 学习速率衰减的周期。 50
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs  学习速率衰减的乘法因子 0.1


# create dir for score   分数
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
# 交并分数
IU_scores = np.zeros((epochs, n_class))
# 像素分数
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()  # 这里他放在For第一行
            if use_gpu:
                # batch_size = 6
                # batch['X'] :[6, 3, 480, 640]
                inputs = Variable(batch['X'].cuda())
                # print(inputs.shape)
                # [6, 32, 480, 640]
                labels = Variable(batch['Y'].cuda())

            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])
            # outputs:[6, 32, 480, 640]
            #
            outputs = fcn_model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if iter % 5 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # 保存
        torch.save(fcn_model.state_dict(),'params.pkl')
        # 验证集调整超参数
        # val(epoch)



# 验证集
def val(epoch):
    # 测试模块固定
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):

        if use_gpu:
            # 1 ,3,720,960
            inputs = Variable(batch['X'].cuda())

        else:
            inputs = Variable(batch['X'])

        # 输出模型
        output = fcn_model(inputs)

        output = output.data.cpu().numpy()
        # print("评估大小：", output.shape)
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        # print(pred.shape)
        for p, t in zip(pred, target):
            # 整个的区域的误差
            total_ious.append(iou(p, t))
            # if epoch==1:
            #     print(p)
            # 这里是一个float的数，像素级别的误差
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    # 开始思是val_len  *n_class
    total_ious = np.array(total_ious).T  # n_class * val_len   32 *70
    # 一行的算术平均值
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    # nan标识空缺数据
    # print("epoch{}, pix_acc: {}, meanIoU: {}'\n'IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    print("epoch{}, pix_acc: {}, meanIoU: {}".format(epoch, pixel_accs, np.nanmean(ious)))
    # 两种分数判断
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    # print("meanIU",ious)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
    print("pixel_accs:",pixel_accs)


# 验证集
def val_socre():
    # 测试模块固定
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    # for iter, batch in enumerate(val_loader):
    #
    #     if use_gpu:
    #         inputs = Variable(batch['X'].cuda())
    #     else:
    #         inputs = Variable(batch['X'])
    img = cv2.imread(
        "/home/sun/桌面/ai_rbote/github_course/AI_rebot/FCN-pytorch-master/python/CamVid/701_StillsRaw_full/0001TP_006690.png")
    img = cv2.resize(img, (720,960), interpolation=cv2.INTER_CUBIC)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[:, :, ::-1]  # switch to BGR
    img = np.transpose(img, (2, 0, 1)) / 255.

    # img[0] -= self.means[0]
    # img[1] -= self.means[1]
    # img[2] -= self.means[2]
    img = torch.from_numpy(img.copy()).float()
    inputs= Variable(img.unsqueeze(0))

    # 输出模型
    output = fcn_model(inputs)

    output = output.data.cpu().numpy()
    # print("评估大小：", output.shape)
    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    print(pred.shape)
    print(pred)







# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
# 交集/并集    判断效果
def iou(pred, target):
    # 预测图片和目标图片
    # 720*960
    ious = []
    for cls in range(n_class):
        # 判断
        pred_inds = pred == cls
        target_inds = target == cls
        # 交集和并集
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
         # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))

    return ious


def pixel_acc(pred, target):
    # 越大越好
    # == 就是对应位置是否相等，sum就是每个位置的bool相加
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


def vai_ima():
    # ref:  https://blog.csdn.net/Hungryof/article/details/81364487
    # https://blog.csdn.net/liangguohuan/article/details/7088304
    num_class = 32
    # weight_path = "C:\\Users\\wen\\Desktop\\w2.pth"
    print(model_path)
    # 网络框架
    net = FCN8s(num_class)
    work=torch.load('params.pkl')
    new_state_dict = collections.OrderedDict()
    # 每个卷积里面的权重
    # for  key, v in work.items():
    #     print (key, v)
    ##解决strict=false的麻烦
    for k,v in work.items():
        name = k[7:]
        new_state_dict[name] = v
    # 填充参数
    net.load_state_dict( new_state_dict)
    # net.load_state_dict(torch.load('params.pkl'), strict=False)

    net.eval()

    print(net)


    img = cv2.imread(
        "/home/sun/桌面/ai_rbote/github_course/AI_rebot/FCN-pytorch-master/python/CamVid/701_StillsRaw_full/0001TP_006690.png")
    img = cv2.resize(img, (720, 960), interpolation=cv2.INTER_CUBIC)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[:, :, ::-1]  # switch to BGR
    img = np.transpose(img, (2, 0, 1)) / 255.

    # img[0] -= self.means[0]
    # img[1] -= self.means[1]
    # img[2] -= self.means[2]
    img = torch.from_numpy(img.copy()).float()
    img = Variable(img.unsqueeze(0))

    output = net(img)  # forward pass 1 3 720 960
    output = output.data;
    output = output.data.cpu().numpy()

    # added by wen
    # print("评估大小：", output.shape)
    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    print(pred.shape)
    print(pred)
    # added by wen
    # print(output)
    cnter = Counter(output[0].flat)
    radio = np.zeros(num_class)
    height = 720
    width = 960

    for i in range(num_class):
        try:
            radio = ((int)(cnter[i])) / (height * width * 1.0)
            print(radio)

        except:
            pass

if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    #train()

    #val_socre()
    vai_ima()
