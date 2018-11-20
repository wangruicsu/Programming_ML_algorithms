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
| param    |                      |                       |       
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
import numpy as np
import pandas as pd
from numpy import *
from decimal import Decimal

# 载入数据，数据类型：一维数据
def loadData(): 
    x = [0,1,2, 3, 4, 5,6,7,8, 9]
    y = [1,1,1,-1,-1,-1,1,1,1,-1]
    return x, y #返回数据特征和数据类别

def get_Gmx(x,y,D,fx):
    """
    函数功能：学习新的分类器

    @param:x
    @param:y
    @param:D
    @param:fx 已有分类器，格式[[a1,G1],...,[am,Gm]]
    """
    # 学习新的分类器
    gmx = []  #所有可能的基学习器
    pred_new = [0]*len(y) # 新分类器的预测结果
    pred_new_base = [] #新基学习器的预测结果
    err_base = []
    for j in range(len(x) - 1):
        gmx.append(0.5*(x[j] + x[j+1]))
        pred_new_base = [1]*int(math.ceil(gmx[j])) + [-1]*int(len(x) - math.ceil(gmx[j]))
        # 预测误差em为预测错误项的权重值之和。预测结果和 y 相乘，为-1的元素（异号）即代表着预测错误
        compare_pred_y = list(map(lambda x: x[0]*x[1], zip(pred_new_base,y)))
        # 根据err_list 为 -1的元素的索引，取得对应权重 D 的值并求和
        err = 0
        for k in range(len(compare_pred_y)):
            if compare_pred_y[k]==-1:err += D[k]
        if not err: break  # 如果直接找到了分类错误为0的基分类器就不再往下找了
        err_base.append(err) # 更新基学习器的误差 list
    # 选择基学习器，坑在这里，摔了好几跤哇
    # 错误思路❌：选择最小误差对应的分割，废话，2.5和8.5分的最好，肯定每次都选到这两个，于是就陷入了
    mini_err = min(err_base)
    mini_err_index = []
    for n in range(len(err_base)):
        if err_base[n] == mini_err:
            mini_err_index.append(n)
    print(mini_err_index)
    Gmx =  gmx[err_base.index(min(err_base))]  #新的基学习器 Gmx，误差err_base中最小元素的索引就对应着 gmx 中的最优分类器
    for i in range(len(fx)):
        if Gmx == fx[i][0]:
            Gmx = gmx[mini_err_index[1]]
    em = min(err_base) # 新的基分类器的误差
#    if em==0:
#        return am, Gmx,y_Gmx
    am_new = 1/2 * math.log((1-em)/em)
    am_new = int(am_new*10000)/10000 #保留四位小数
    fx.append([am_new,Gmx]) # 新分类器
    # 计算新分类的预测结果
    pred_fx_new = [0]*len(y)
    pred_Gm = []#每一个基学习器的预测结果
    # 计算每一个已有的基学习器预测结果pred_am 并和 已有的分类器 fx 预测结果pred_fx 累加
    for i in range(len(fx)):
        pred_Gm = [fx[i][0]]*int(math.ceil(fx[i][1])) + [-fx[i][0]]*int(len(x) - math.ceil(fx[i][1]))
        pred_fx_new = list(map(lambda x: x[0]+x[1], zip(pred_fx_new, pred_Gm)))    
    pred_fx_new = np.sign(pred_fx_new)
    return fx,pred_fx_new

# 更新新的 数据权值 D
def calW(D_old,am,y,y_Gmx):
    Zm_item = []  #计算 规范化因子 参考李航《统计学习方法》公式(8.5)
    for i in range(len(y)):
        Zm_item.append(D_old[i]*math.exp(-am*y[i]*y_Gmx[i]))
    Zm = sum(Zm_item)
    # 控制精度为五位数，round 不好用，round(13.949999999999999,2) = 13.949999999999999
    D_new = [int(i/Zm*100000)/100000 for i in Zm_item]
#    D_new = [Decimal(i/Zm).quantize(Decimal('0.0000')) for i in Zm_item]
    
#    print(D_new)
    return D_new
    
# 参数：x_list,y_list,maxIterNum:基学习器数量,errorThreshold:分类器误差阈值
def AdaBoost(x,y,maxIterNum,errorThreshold):
    D1 = [1.0/len(x)]*len(x)
    fx_err = 0
    D = D1
    fx = []
    for i in range(maxIterNum):
#        print(D)
        fx,y_fx = get_Gmx(x,y,D,fx)
#        print(fx,y_fx)
        #计算分类器的误差
        compare_fx_y = list(map(lambda x: x[0]*x[1], zip(y, y_fx)))
        err_fx = 0
        for k in range(len(y)):
            if compare_fx_y[k]==-1:
                err_fx += D[k]
#        print(err_fx)
        if err_fx < errorThreshold:break
        am = fx[-1][0]
#        print(D,am,y,y_fx)
        D = calW(D,am,y,y_fx)
        
    return fx
    

if __name__ == '__main__':
    x, y = loadData()
    Gx = AdaBoost(x,y,maxIterNum = 10,errorThreshold = 0.01)
#    print(Gx)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    