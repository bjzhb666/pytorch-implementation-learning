"""
Created on 2022.04.01
Paper: Attention is all you need.
Implementation of additive attention and the NMT using transformer
Author: ZHB
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义可视化注意力机制的函数
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(4, 4),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                # sharex=True, sharey=True,
                                 squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

def test_show_heatmaps():
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

# 加性注意力层,不管是自定义层、自定义块、自定义模型，都是通过继承Module类完成的
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, dropout=False, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # https://docs.pythontab.com/interpy/args_kwargs/Usage_kwargs/ **kwargs的用法，其实就是让程序可扩展性更好，后面并没有用到
        self.W_k = nn.Linear(key_size, num_hidden, bias=False)
        self.W_q = nn.Linear(query_size, num_hidden, bias=False)
        self.W_v = nn.Linear(num_hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, key, value, valid_length): # forward函数需要传入真正的实参
        keys, queries = self.W_k(key), self.W_q(queries) # 类调用的时候需要写self
        
