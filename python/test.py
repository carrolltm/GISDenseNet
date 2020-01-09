# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
# a= Variable(torch.tensor([[1,2,3],[4,5,6]]))
# b=torch.tensor([[1,2,3],[4,5,6]])
# # print((a==b).sum())
# print(b.numpy()[0][0]*100)
import  pandas as pd

sds = '30.461282_114.61528090000002.jpg'
sds=sds.split('_')[1][:-4]
print(sds)
