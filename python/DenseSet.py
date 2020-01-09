# -*- coding:utf-8 -*-
import imageio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import scipy.misc
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
'''
数据处理流程：
1：首先根据图片路径读取图片
2：对图片都调用预处理的方法
3：预处理方法也可以实现数据增强
'''


'''
预处理方法:
Rescale ：调整图片大小
RandomCrop：随机裁剪图片，这是一种数据增强的方法
ToTensor：将 numpy 格式的图片转换为 pytorch 的数据格式 tensors ，这里需要交换坐标。
'''
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): 带有标注信息的 csv 文件路径
            root_dir (string): 图片所在文件夹
            transform (callable, optional): 可选的用于预处理图片的方法
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    # idx 为
    def __getitem__(self, idx):
        # 读取图片
        # print(idx)
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        img_name = img_name+'.png'
        # 这里需要添加Pilmode
        # Array
        image =imageio.imread(img_name, pilmode='RGB')
        # 读取分数并转换为 numpy 数组
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])  # 1*1  这是二维的
        landmarks = landmarks.astype('double')
        sample = {'image': image, 'landmarks': landmarks}    # 图片和分数


        # 进行一系列变化
        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """将图片调整为给定的大小.

    Args:
        output_size (tuple or int): 期望输出的图片大小. 如果是 tuple 类型，输出图片大小就是给定的 output_size；
                                    如果是 int 类型，则图片最短边将匹配给的大小，然后调整最大边以保持相同的比例。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        # 判断给定大小的形式，tuple 还是 int 类型
        if isinstance(self.output_size, int):
            # int 类型，给定大小作为最短边，最大边长根据原来尺寸比例进行调整
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # # 根据调整前后的尺寸比例，调整关键点的坐标位置，并且 x 对应 w，y 对应 h
        # landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
    # 给定图片，随机裁剪其任意一个和给定大小一样大的区域
class RandomCrop(object):
    """给定图片，随机裁剪其任意一个和给定大小一样大的区域.

    Args:
        output_size (tuple or int): 期望裁剪的图片大小。如果是 int，将得到一个正方形大小的图片.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        # # 调整关键点坐标，平移选择的裁剪起点
        # landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
class ToTensor(object):
    """将 ndarrays 转换为 tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 调整坐标尺寸，numpy 的维度是 H x W x C,而 torch 的图片维度是 C X H X W
        image = image.transpose((2, 0, 1)).astype(np.double)
        # print(image)


        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        landmarks=torch.Tensor([sample['landmarks'][0][0]]).type(torch.FloatTensor)/100.

        return {'image': image,
                'landmarks': landmarks}
class Dataset2(Dataset):
    def __init__(self,dataset,transform):
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img,label=self.dataset
        return img,label

#
# if  __name__=="__main__":
#     transformed_dataset = FaceLandmarksDataset(csv_file='scores/auto_save_scores_wealthy.csv',
#                                         root_dir='scores/wealthy/',
#                                         transform=transforms.Compose([
#                                             Rescale(256),
#                                             RandomCrop(224),
#                                             ToTensor()
#                                         ]))

    # example 例子
    # # 读取前 4 张图片并展示
    # sample = face_dataset[0]
    # # print(0, sample['image'].shape, sample['landmarks'])
    # scale = Rescale(256)
    # crop = RandomCrop(128)
    # composed = transforms.Compose([Rescale(256),
    #                                RandomCrop(224)])
    #
    # transformed_sample = composed(sample)
    # print(transformed_sample['image'].shape)
    # for i in range(len(transformed_dataset)):
    #     sample = transformed_dataset[i]
    #
    #     print(i, sample['image'].size(), sample['landmarks'].shape)
    #
    #     if i == 3:
    #         break














