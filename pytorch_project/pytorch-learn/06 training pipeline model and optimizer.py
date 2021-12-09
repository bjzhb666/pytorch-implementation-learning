# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:06:50 2021

@author: dell
"""
import torch
import torch.nn as nn
#  f = w *x
#  f = 2 *x
#  要使用Linear函数，需要将tensor这里变成二维的
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

#  model = nn.Linear(input_size, output_size)  # 一层


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #  define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)
#  gradient descent
#  MSE = 1/N * (w*x - y)**2
#  dJ/dw = 1/N 2x (w*x-y)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
#%%
#  Training
lr = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(n_iters):
    #  prediction=forward pass
    y_pred = model(X)
    #  loss
    l = loss(Y, y_pred)
    #  gradient
    l.backward()  # dl/dw
    #  update weights
    optimizer.step()
    #  zero gradient
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch{epoch+1}: w={w[0][0].item():.3f},loss={l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
