# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:57:14 2021

@author: dell
"""
'''
Most popular activation function
1. step function (not use)
2. sigmod   1/(1+exp(-x))二分类问题 值域0-1
3.TanH    scaled SIGMOD function 2/(1+exp(-2x))-1值域-1~+1 隐藏层好选择
4.ReLU   max(0,x) most typical choice (RElU 用在隐藏层也可)
5. Leaky ReLU ：x, if x>0; else a*x(a<<1，非常小) 可以解决梯度消失问题 vanishing gradient
6. softmax多分类
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# option 1 (craete nn module)
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # nn.Sigmoid() nn.Softmax() nn.TanH() nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# option 2 (use activation function directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1)
        out = torch.sigmoid(self.linear2)
        return out
    '''
    torch.softmax()
    torch.tanh()
    F.relu the same as torch.relu()
    F.leaky_relu()只在这个API中有，torch中没有
    '''
