
from __future__ import print_function
import torch
from torch.autograd import Variable
import time
import cv2
from imutils.video import FPS, WebcamVideoStream
import argparse
from fcn import *
from collections  import Counter
import numpy as np
import scipy.misc
import collections
import os
import os.path
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
import csv






num_class = 32





parser = argparse.ArgumentParser(description='gis')
parser.add_argument('--weights', default='C:\\Users\\wen\\Desktop\\w2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# 图片的文件夹
# root='scores/pics'
root='scores/wealthy'
csvfile="wealthy2.csv"
#stream = cv2.VideoCapture(path)

height = 720
width = 960
weight_path = "params.pkl"


    
num_class = 32
# 网络框架
net = FCN8s(num_class)
work=torch.load(weight_path)
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



def segment(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
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
    pred = output.transpose(0, 2, 3, 1).reshape(-1, num_class).argmax(axis=1).reshape(N, h, w)
    #print(pred.shape)
    #print(pred)
    # added by wen
    # print(output)
    cnter = Counter(pred[0].flat)
    radios = np.zeros(num_class)
   
    for i in range(num_class):
        try:
            radios[i] = ((int)(cnter[i])) / (height * width * 1.0)
            #print(radio[i])

        except:
            pass
    return radios.tolist()

if __name__ == '__main__':
    lists=[]
    i=0
    for f in listdir(root):
        lists.append([f]+segment(join(root,f)))
        i=i+1
        if i%10==0:
            print(i)

    # 列名 0-31
    head=columns=["filename"]+list(range(num_class))+['score']
    my_df = pd.DataFrame(lists, columns=head)
    my_df.to_csv(csvfile, index=False)
