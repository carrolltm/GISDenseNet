# -*- coding:utf-8 -*-

# 处理数据
from transform import  transform_lat,transform_lon,wgs2gcj
import  cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import  pandas as pd
import collections
import numpy as np
import time
import sys
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import pandas as pd




# 一次多少张图片   一共630张图片
from DenseSet import FaceLandmarksDataset, Rescale,RandomCrop,ToTensor,Dataset2

batch_size = 16    # 8都不行 GPU超出内存了
# 多少批      批量梯度下降计算全部训练集样本梯度的平均，然后更新梯度
epochs = 10
lr = 0.001
# 防止训练过程中陷入局部最小值 ref:https://blog.csdn.net/u013989576/article/details/70241121
# 下面几个都是跟学习率有关
momentum = 0
w_decay = 1e-5


# 训练文件位置
# train_file = os.path.join(root_dir, "auto_save_scores_wealthy.csv")  # 图片值和分数

use_gpu = torch.cuda.is_available()
# GPU的数量
num_gpu = list(range(torch.cuda.device_count()))
#
# # 对数据预处理的调用
# normMean = [0.4948052, 0.48568845, 0.44682974]
# normStd = [0.24580306, 0.24236229, 0.2603115]
# normTransform = transforms.Normalize(normMean, normStd)
root_dir = "scores"
# 后面如果加入标准差和方差的话可以参考
# https://blog.csdn.net/weixin_40766438/article/details/100750633
# 需要在 FaceLandmarksDataset中改改，直接替换成transfrom.Compose
def data_deal(cvs_file,root_dir):
    all_data = FaceLandmarksDataset(csv_file=cvs_file,
                                    root_dir=root_dir,
                                    transform=transforms.Compose([
                                        Rescale(256),
                                        RandomCrop(224),
                                        ToTensor()
                                    ]))
    # all_data = FaceLandmarksDataset(csv_file='scores/auto_save_scores_beautiful.csv',
    #                                 root_dir='scores/beautiful/',
    #                                 )

    # trian_size = int(0.8 * len(all_data))
    # # 因为得到的是一个数组字典，所以要划分数据集的话需要一个个往数组里里面添加
    # train_dataset = []
    # test_dataset = []
    # for i in range(trian_size):
    #     train_dataset.append(all_data[i])
    # for i in range(trian_size, len(all_data)):
    #     test_dataset.append(all_data[i])
    # train_transform = transforms.Compose([
    #     transforms.RandomSizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # val_trainsform = transforms.Compose([
    #     transforms.Scale(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # # 因为要将train和test分别进行trainsform，所以只能重新写一个类进行transform，实在想不到好办法了
    # train_dataset2 = Dataset2(train_dataset, transform=train_transform)
    # test_dataset2 = Dataset2(test_dataset, transform=val_trainsform)

    train_data = all_data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader




# 加载模型
from Desnet import  DenseNet
desecnn = DenseNet()
# print(desecnn)
# 使用GPU
if use_gpu:
    ts = time.time()
    # vgg_model = vgg_model.cuda()
    desecnn = desecnn.cuda()
    desecnn = nn.DataParallel(desecnn, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))



# 损失函数hg
# criterion = nn.BCEWithLogitsLoss()
criterion =  torch.nn.MSELoss()

step_size = 10
# 批标准化
gamma = 0.5
# 优化器
optimizer = torch.optim.Adam(desecnn.parameters(), lr=1e-4)

# 学习率自动下降
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,  # 学习速率衰减的周期。 50
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs  学习速率衰减的乘法因子 0.1


# 训练
def train(store_pkl,train_loader):
    number_step=0
    for epoch in range(epochs):
        ts = time.time()
        for iter, batch in enumerate(train_loader):

            if use_gpu:
                # batch_size = 6
                inputs = Variable(batch['image'].cuda())
                # print(inputs.shape)
                labels = Variable(batch['landmarks'].cuda())
                # print(labels.shape)
            else:
                inputs, labels = Variable(batch['image']), Variable(batch['landmarks'])
            outputs = desecnn(inputs)
            # print(outputs.shape)

            # print(outputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()  # 这里他放在For第一行
            loss.backward()
            optimizer.step()
            scheduler.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save( desecnn.state_dict(),store_pkl )# 'Denseparams.pkl'

def val_score():
    desecnn.eval()
    img = cv2.imread(
        "/home/sun/桌面/ai_rbote/github_course/AI_rebot/FCN-pytorch-master/python/scores/wealthy/113.812425_30.746098_0_0.png")
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[:, :, ::-1]  # switch to BGR
    img = np.transpose(img, (2, 0, 1)) / 255.
    # print(img)

    # img[0] -= self.means[0]
    # img[1] -= self.means[1]
    # img[2] -= self.means[2]
    img = torch.from_numpy(img.copy()).float()
    inputs = Variable(img.unsqueeze(0))

    # 输出模型
    output =desecnn(inputs)

    output = output.data.cpu().numpy()
    # print("评估大小：", output.shape)

    pred = output

    print('the val prcc is :',pred)


def vai_ima(file_name,store_file,pkl_par):
    # ref:  https://blog.csdn.net/Hungryof/article/details/81364487
    # https://blog.csdn.net/liangguohuan/article/details/7088304


    # 网络框架
    net = DenseNet()
    work=torch.load(pkl_par)
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


    data_list = os.listdir(file_name)
    data_len = len(data_list)  # 长度
    # print(data_list)
    print("the length of viedo_data_len:",data_len)
    # 保存分数
    # lists = []
    i =0
    store_file = store_file
    t = open(store_file, "w")
    t.write("lat,long,score\n")
    for idx, name in enumerate(data_list):
        if 'jpg' not in name:
            continue
        lat = name.split('_')[0]
        long = name.split('_')[1][:-4]
        img_name = os.path.join(file_name, name)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img.copy()).float()
        img = Variable(img.unsqueeze(0))/255.

        output = net(img)
        output = output.data
        output = output.data.cpu().numpy()

        # added by wen
        # print("评估大小：", output.shape)
        print(output)
        # 保存地址
        # i = i + 1
        # if i % 10 == 0:
        #     print(i)
        # 列名 0-31

    # 保存到文件里面
        t.write("{},{},{}\n".format(lat,long,output[0][0]*100))
    # head = columns = ["filename"] + ['score']
    # my_df = pd.DataFrame(lists, columns=head)
    # my_df.to_csv(store_file , index=False)


# 数据的坐标系转换，时间排序
def deal_data(in_file):
    reference = 'tran_ref_csv/before.csv'
    # reference = 'tran_ref_csv/after.csv'

    reference = pd.read_csv(reference)
    df = pd.read_csv(in_file)
    df = pd.merge(reference, df, 'inner', on=['lat', 'long'])
    df = df.drop(['Time'], axis=1)
    # 这地方是否可以删除
    # df.to_csv(out_file, index=False)
    # df = pd.read_csv(out_file)
    df_out = pd.DataFrame(columns=['score', 'lat', 'long'])
    for _, row in df.iterrows():
        lat, long = row['lat'], row['long']
        lat, long = wgs2gcj(lat, long)
        df_out = df_out.append({'score': row['score'], 'lat': lat, 'long': long}, ignore_index=True)
    df_out.to_csv(in_file, index=False)

    pass

if __name__=='__main__':
    # 六个模型训练
    # 储存模型参数的地址
    pkl_list  = ['MoudlePar/beautiful2params.pkl','MoudlePar/boring2params.pkl','MoudlePar/depressing2params.pkl',
              'MoudlePar/lively2params.pkl','MoudlePar/safety2params.pkl','MoudlePar/wealthy2params.pkl']
    csv_files = ['scores/auto_save_scores_beautiful.csv','scores/auto_save_scores_boring.csv','scores/auto_save_scores_Depressing.csv',
                 'scores/auto_save_scores_lively.csv','scores/auto_save_scores_safety.csv','scores/auto_save_scores_wealthy.csv']
    root_dirs = ['scores/beautiful/','scores/boring/','scores/depressing/',
                 'scores/lively/','scores/safety/','scores/wealthy/']

    # choice train
    for i in range(len(pkl_list)):
        store_pkl= pkl_list[i]
        print(i)
        train_datset=data_deal(cvs_file=csv_files[i],root_dir=root_dirs[i])

        train(store_pkl,train_loader=train_datset)
        # val_score()
        print(root_dirs[i][7:],"The train is end!")

    # 输入的视频数据测试
    # 输入视频图片所在文件
    # viedo_dir = "scores/"
    data_dir = os.path.join(root_dir, "pics")  # 该文件夹下的视频数据
    # # 保存输出csv的文件地址 /默认当前目录
    viedo_result = ["show_result/beautiful.csv","show_result/boring.csv","show_result/depressing.csv",
                    "show_result/lively.csv","show_result/safety.csv","show_result/wealthy.csv"]
    for i in range(len(viedo_result)):
        store_file = viedo_result[i]
        # 保存到文件里面
        vai_ima(data_dir,store_file,pkl_list[i])
        deal_data(store_file)










