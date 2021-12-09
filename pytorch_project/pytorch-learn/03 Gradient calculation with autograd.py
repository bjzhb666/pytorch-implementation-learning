# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 09:54:55 2021

@author: dell
"""
# PyTorch Tutorial 03 - Gradient Calculation With Autograd
''' 
当不想记录梯度时，有如下三种操作
x.requires_grad_(False)
x.detach()
with torch.no_grad():
'''
import torch
x = torch.randn(3, requires_grad=True)
print(x)


x.requires_grad_(False)
# when there is an underscore, it means the operation will modify the variable itself
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    z = x + 2
    print(z)

weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD([weights], lr=0.01)
optimizer.step()  # 优化步
optimizer.zero_grad()  # 与梯度清零类似

for epoch in range(3):
    model_output = (weights*3).sum()
    
    print(model_output)

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

'''
输出如下：如果没有grad.zero这一步
这反映了计算梯度时候，梯度会累加起来
这是错误的，因此每次必须empty the gradient
tensor([3., 3., 3., 3.])
tensor([6., 6., 6., 6.])
tensor([9., 9., 9., 9.])
'''
