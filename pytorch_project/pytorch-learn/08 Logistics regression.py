# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:35:14 2021

@author: dell
"""
'''
the procedure of building a neural network
0. Prepare the data
1. Design a model(input, output_size, forward pass)
2. Construct loss and optimizer
3. Training loop
    -forward pass :compute prediction and loss
    -backward pass: gradients
    -update weigths
'''
# logistics regression
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# data
bc = datasets.load_breast_cancer()  # 二分类问题
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale,logistic regression 中常把数据变成标准正态分布
sc = StandardScaler()  # transform the data into 标准正态分布
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# 把y_train和y_test转为列向量
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# Model
# f=wx+b, sigmoid at the end
class LosgisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LosgisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LosgisticRegression(n_features)
# loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#%%
# training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model.forward(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
#%%
# test and check the acc (Note that no grad here)
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
  
    print(f'accuracy = {acc:.4f}')
