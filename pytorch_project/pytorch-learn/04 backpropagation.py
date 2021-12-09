# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:45:29 2021

@author: dell
"""
# backpropagation04
'''
forward pass
backward pass
update
'''
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#  forward pass and  compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#  backward pass
loss.backward()
print(w.grad)

# next forward
