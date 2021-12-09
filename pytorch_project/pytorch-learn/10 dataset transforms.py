# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:52:36 2021

@author: dell
"""
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

'''
本节介绍如何自定义数据的转化方式，通过定义一个transform的类
从而实现对数据的初步处理，以及同时使用多个transform的方法
'''


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./pytorchTutorial-master/data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]  # n_samples, 1

        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform():  # 一个乘法的变换
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample  # unpack our sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
# print(type(features), type(labels))

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward ,update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{n_iterations}, inputs {inputs.shape}')


# 同时使用多个transform的方法
composed = torchvision.transforms.Compose([ToTensor, MulTransform(2)])
dataset = WineDataset(transform=composed)