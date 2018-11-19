#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习算法：adaboost
数据集数据类型：:list
李航机器学习：例子8.1

| --------------------------------------------------------|
| Adaboost                                                |
| ---------|----------------------|-----------------------|
|  param   |  取值                 |  说明                 |  
|  数据集   |                      |  x的起始点             |    
| epochs   |                      |  迭代周期              |    
| param    |                      |                 |       
| ---------|----------------------|-----------------------|

Created on Fri Nov 19 10:23:27 2018
"""

"""
little thoughts：
本来想采取二维数据的，不过这样每一次都要到一个线性分类器，就可以直接掉包用 SVM ，LR等，然后再集成起来。
突然就领悟到 AdaBoost 是一种思想一种框架一个壳，它只是组合弱分类器，它并不产生基分类器，
实际上基分类器是什么，是需要自己选择的。
集成算法干的是优化的事儿。
怎么有点像投资组合啊，你有投基金啊活期定期啊股票啊，才有优化组合这个事。
"""
import math
import pandas as pd
import numpy as np
from numpy import *
# 载入数据，数据类型：一维数据
def loadData(): 
    x = [0,1,2, 3, 4, 5,6,7,8, 9]
    y = [1,1,1,-1,-1,-1,1,1,1,-1]
    return x, y #返回数据特征和数据类别

# 基学习器
def get_Gmx(x,y,D,fx):
#    print(D)
    Gmx_all = []
    Gmxi_not_yi = []
    em = []
    pred = []
#    item = 0
    pred = []

    for i in range(len(x) - 1):
        item = 0
        Gmx_all.append((x[i] + x[i+1])/2.0)
        if fx:
#            pred = [1]*int(math.ceil(Gmx_all[i])) + [-1]*int(len(x) - math.ceil(Gmx_all[i]))
            for j in range(len(fx)):
                am = fx[j]
                item = [1]*int(math.ceil(am)) + [-1]*int(len(x) - math.ceil(am))
                pred_am = [am*k for k in item]
                pred = list(map(lambda x: x[0]+x[1], zip(pred, pred_am)))
        else:
            pred = [1]*int(math.ceil(Gmx_all[i])) + [-1]*int(len(x) - math.ceil(Gmx_all[i]))
        Gmxi_not_yi = list(map(lambda x: x[0]-x[1], zip(y, pred)))
#        print(Gmxi_not_yi)
        item1 = 0
        for n in range(len(D)):
            if Gmxi_not_yi[n] != 0: 
                item1 += D[n]
        em.append(item1)
    # 基分类器
    print(em)
    Gmx = Gmx_all[em.index(min(em))]
    # 计算基学习器系数 am
    am = 1/2 * math.log((1-min(em))/min(em))
    # 计算该基学习器的预测结果，用于更新数据权值
    y_Gmx = [1]*int(math.ceil(Gmx)) + [-1]*int(len(x) - math.ceil(Gmx))

    return am, Gmx,y_Gmx

# 更新新的 数据权值 D
def calW(D_old,am,y,y_Gmx):
    Zm_item = []  #计算 规范化因子 参考李航《统计学习方法》公式(8.5)
    for i in range(len(y)):
        Zm_item.append(D_old[i]*math.exp(-am*y[i]*y_Gmx[i]))
    Zm = sum(Zm_item)
    D_new = [i/Zm for i in Zm_item]
#    D_new = list(map(lambda x: x[0]*x[1]/Zm, zip(D_old, Zm_item)))
    return D_new
    
# 参数：x_list,y_list,maxIterNum:基学习器数量,errorThreshold:分类器误差阈值
def AdaBoost(x,y,maxIterNum,errorThreshold):
    D1 = [1.0/len(x)]*len(x)
    fx_err = 0
    D = D1
    amGmx = [] #am * Gmx 后的结果存储
    Gmx_all = []  #
    am = 0
    am_all = []
    Gmx = 0
    Gx = [0]*len(x)
    fx = []
    for i in range(maxIterNum):
#        print(D)
        am, Gmx,y_Gmx = get_Gmx(x,y,D,fx)
        fx.append(am)
        
        Gmx_all.append(['am:',am,'Gmx: ',Gmx])
        
        amGmx = [am*j for j in y_Gmx]
        #更新的分类器 的 分类误差
        Gx = list(map(lambda x: x[0]+x[1], zip(Gx, amGmx)))
        Gx_sign = np.sign(Gx)
        fx_err = 1 - list(map(lambda x: x[0]-x[1], zip(y, Gx_sign))).count(0)/len(x)
        print(fx_err)
        if fx_err < errorThreshold:break
    
        D = calW(D,am,y,y_Gmx)
        
    return Gmx_all
    

if __name__ == '__main__':
    x, y = loadData()
    Gx = AdaBoost(x,y,maxIterNum = 100,errorThreshold = 0.01)
    print(Gx)
    

    