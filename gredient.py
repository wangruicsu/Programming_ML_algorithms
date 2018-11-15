#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:43:07 2018
"""

'''
【实现】
--1.梯度下降法  
--2️.带momentum的梯度下降法  
--3️.带衰减因子的梯度下降法
| --------------------------------------------------------|
| 函数(有鞍点)     y = 0.1 * x*x*x + x*x - 1                |
| ---------|----------------------|-----------------------|
|  param   |  取值                 |  说明                 |  
| x_start  |  -5                  |  x的起始点             |    
| df       |                      |  目标函数的一阶导函数    | 
| epochs   |   5                  |  迭代周期              |    
| lr       |[0.01, 0.1, 0.6, 0.9] |  学习率                |       
| momentum |[0.0, 0.1, 0.5, 0.9]  |  冲量                 |       
| decay    |                      |  学习率衰减因子         |      
| ---------|----------------------|-----------------------|
'''

import numpy as np
import matplotlib.pyplot as plt

def func(x):
    '''目标函数'''
    return 0.1 * x*x*x + x*x - 1

def dfunc(x):
    '''目标函数一阶导数:dy/dx'''
    return 0.3 * x * x + 2* x

def GD(x_start, df, epochs, lr):
    """
    梯度下降法
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
#        print(dx)
        v = - dx * lr
        x += v
        xs[i+1] = x
#    print(xs)
    return xs #x在每次迭代后的位置（包括起始点），长度为epochs+1

def GD_momentum(x_start, df, epochs, lr, momentum):
    """
    带有冲量的梯度下降法
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        v = - dx * lr + momentum * v
        x += v
        xs[i+1] = x
    return xs

def GD_decay(x_start, df, epochs, lr, decay):
    """
    带有学习率衰减因子的梯度下降法
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # 学习率衰减
        lr_d = lr * 1.0 / (1.0 + decay * i)
        v = - dx * lr_d
        x += v
        xs[i+1] = x
    return xs

def demo1_GD_lr():
    '''画图：梯度下降法'''
    line_x = np.linspace(-9, 4, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate',figsize = (10,10))

    x_start = -5
    epochs = 5

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
    line_x = np.linspace(-9, 4, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate, Momentum',figsize = (10,30))
#    plt.title('Gradient Desent: Learning Rate, Momentum')

    x_start = -5
    epochs = 5

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
    line_x = np.linspace(-9, 4, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Decay',figsize = (10,30))

    x_start = -5
    epochs = 5

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

def Adagrad(x_start, df, epochs, lr):
    """
    Adagrad
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    grad_squared = 0
    for i in range(epochs):
        dx = df(x)
#        print(dx)
        grad_squared += dx * dx
        v = - dx * lr/(np.sqrt(grad_squared) + 1e-7)
        x += v
        xs[i+1] = x
#    print(xs)
    return xs #x在每次迭代后的位置（包括起始点），长度为epochs+1

def demo4_Adagrad():
    '''画图：梯度下降法'''
    line_x = np.linspace(-9, 4, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate',figsize = (10,10))

    x_start = -5
    epochs = 5

    lr = [0.1, 0.3, 0.9]

    color = ['r', 'g', 'y']
    size = np.ones(epochs+1) * 10
    size[-1] = 70
    for i in range(len(lr)):
        x = Adagrad(x_start, dfunc, epochs, lr=lr[i])
        plt.subplot(1, 3, i+1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}'.format(lr[i]))
        plt.scatter(x, func(x), c=color[i])
        plt.legend()
    plt.show()
    
def RMSProp(x_start, df, epochs, lr, decay):
    """
    RMSProp
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    grad_squared = 0
    for i in range(epochs):
        dx = df(x)
#        print(dx)
        grad_squared += decay * grad_squared + (1 - decay) * dx * dx
        v = - dx * lr/(np.sqrt(grad_squared) + 1e-7)
        x += v
        xs[i+1] = x
#    print(xs)
    return xs #x在每次迭代后的位置（包括起始点），长度为epochs+1

def demo5_RMSProp():
    line_x = np.linspace(-9, 4, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Decay',figsize = (10,10))

    x_start = -5
    epochs = 5

    lr = [0.1, 0.3, 0.9, 0.99]
    decay = [0.0, 0.01, 0.9]

    color = ['k', 'r', 'g', 'y']

    row = len(lr)
    col = len(decay)
    size = np.ones(epochs + 1) * 10
    size[-1] = 70
    for i in range(len(decay)):
        x = RMSProp(x_start, dfunc, epochs, lr=lr[1],decay = decay[i])
        plt.subplot(1, 3, i+1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(lr[1], decay[i]))
        plt.scatter(x, func(x), c=color[i])
        plt.legend()
    plt.show()

   
if __name__ == '__main__':
    demo2_GD_momentum()
    demo3_GD_decay()
    
    demo1_GD_lr()  
    demo4_Adagrad()
    demo5_RMSProp()
    