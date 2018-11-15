#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:43:07 2018
"""

'''选择函数：三次函数（有鞍点）
@param: x_start: x的起始点
@param: df: 目标函数的一阶导函数
@param: epochs: 迭代周期
@param: lr: 学习率
@param: momentum: 冲量
@param: decay: 学习率衰减因子
'''

import numpy as np
import matplotlib.pyplot as plt


# 目标函数:y=x^2
def func(x):
#    return x*x
    return 0.1 * x*x*x + 1*x*x - 1


# 目标函数一阶导数:dy/dx=2*x
def dfunc(x):
#    return 2*x
    return 0.3 * x * x + 2* x

# Gradient Descent
def GD(x_start, df, epochs, lr):
    """
    梯度下降法。给定起始点与目标函数的一阶导函数，求在epochs次迭代中x的更新值
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr
        x += v
        xs[i+1] = x
    return xs #x在每次迭代后的位置（包括起始点），长度为epochs+1

def GD_momentum(x_start, df, epochs, lr, momentum):
    """
    带有冲量的梯度下降法。
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr + momentum * v
        x += v
        xs[i+1] = x
    return xs

def GD_decay(x_start, df, epochs, lr, decay):
    """
    带有学习率衰减因子的梯度下降法。
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # 学习率衰减
        lr_i = lr * 1.0 / (1.0 + decay * i)
        # v表示x要改变的幅度
        v = - dx * lr_i
        x += v
        xs[i+1] = x
    return xs

def demo1_GD_lr():
    # 函数图像
    line_x = np.linspace(-5, 1, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate',figsize = (10,10))

    x_start = -0.5
    epochs = 100

    lr = [0.1, 0.3, 0.9]

    color = ['r', 'g', 'y']
    size = np.ones(epochs+1) * 10
    size[-1] = 70
    for i in range(len(lr)):
        x = GD(x_start, dfunc, epochs, lr=lr[i])
        plt.subplot(1, 3, i+1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}'.format(lr[i]))
        plt.scatter(x, func(x), c=color[i])
        plt.legend()
    plt.show()

def demo2_GD_momentum():
    line_x = np.linspace(-5, 1, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate, Momentum',figsize = (10,30))
#    plt.title('Gradient Desent: Learning Rate, Momentum')

    x_start = -0.5
    epochs = 100

    lr = [0.01, 0.1, 0.6, 0.9]
    momentum = [0.0, 0.1, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    row = len(lr)
    col = len(momentum)
    size = np.ones(epochs+1) * 10
    size[-1] = 70
    for i in range(row):
        for j in range(col):
            x = GD_momentum(x_start, dfunc, epochs, lr=lr[i], momentum=momentum[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, func(x), c=color[i], label='lr={}, mo={}'.format(lr[i], momentum[j]))
            plt.scatter(x, func(x), c=color[i], s=size)
            plt.legend(loc=0)
    plt.show()
    
def demo3_GD_decay():
    line_x = np.linspace(-5, 1, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Decay',figsize = (10,30))
#    plt.title('Gradient Desent: Decay')


    x_start = -0.5
    epochs = 100

    lr = [0.1, 0.3, 0.9, 0.99]
    decay = [0.0, 0.01, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    row = len(lr)
    col = len(decay)
    size = np.ones(epochs + 1) * 10
    size[-1] = 70
    for i in range(row):
        for j in range(col):
            x = GD_decay(x_start, dfunc, epochs, lr=lr[i], decay=decay[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(lr[i], decay[j]))
            plt.scatter(x, func(x), c=color[i], s=size)
            plt.legend(loc=0)
    plt.show()
    
if __name__ == '__main__':
    demo1_GD_lr()
    demo2_GD_momentum()
    demo3_GD_decay()