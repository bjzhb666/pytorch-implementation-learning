# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:55:55 2021

@author: dell
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:05:39 2021

@author: dell
"""
#  video 05 Gradient Descent with Autograd
#  (second part : gradient descent using lib)
import torch

#  f = w *x
#  f = 2 *x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


#  model prediction
def forward(x):
    return w * x


#  loss function
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


#  gradient descent
#  MSE = 1/N * (w*x - y)**2
#  dJ/dw = 1/N 2x (w*x-y)
'''
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
'''

print(f'Prediction before training: f(5) = {forward(5):.3f}')
#%%
#  Training
lr = 0.01
n_iters = 10

for epoch in range(n_iters):
    #  prediction=forward pass
    y_pred = forward(X)
    #  loss
    l = loss(Y, y_pred)
    #  gradient
    l.backward()  # dl/dw
    #  dw = gradient(X, Y, y_pred)

    #  update weights
    with torch.no_grad():  # 更新学习率不应该计算到梯度中的计算中，所以这一步要扔出去
        w -= lr * w.grad
    #  zero gradient
    w.grad.zero_()
    if epoch % 1 == 0:
        print(f'epoch{epoch+1}: w={w:.3f},loss={l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
