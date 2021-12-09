# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:33:54 2021

@author: dell
"""
import torch
import torch.nn as nn
import numpy as np

'''
realize in numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss  # /float(predicted.shape[0])


# Y must be one hot encoded
Y = np.array([1, 0, 0])

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(l1, l2)
'''

# realize in pytorch
loss = nn.CrossEntropyLoss()
# 3samples
# nsamples * nclasses = 3*3
Y = torch.tensor([2, 0, 1])
# raw value for pred, no need forsoftmax
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.01, 1.0, 2.1], [0.1, 3.0, 0.1]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

# item()方法可以去掉torch，直接输出数字
print(l1.item(), l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1, predictions2)

'''
多分类问题的激活函数用softmax，二分类问题的激活函数用SIGMOD
损失函数多分类用nn.CrossEntropyLoss(),二分类用nn.BCELoss()
'''
