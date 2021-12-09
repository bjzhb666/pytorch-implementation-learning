# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:25:36 2021

@author: dell
"""
import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
# 使用0值表示沿着每一列或行标签\索引值向下执行方法
# 使用1值表示沿着每一行或者列标签模向执行对应的方法


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)