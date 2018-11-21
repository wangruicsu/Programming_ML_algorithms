#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习算法：adaboost
李航机器学习：例子8.1

Created on Fri Nov 21 2018
"""

import math
import numpy as np
import pandas as pd
from numpy import *
from decimal import Decimal

# 载入数据，type: 1-D list
def loadData(): 
    x = [0,1,2, 3, 4, 5,6,7,8, 9]
    y = [-1,-1,-1,-1,-1,1,-1,1,1,1]
#    y = [1,1,1,-1,-1,-1,1,1,1,-1]
    return x, y #返回数据特征和数据类别

def get_Gmx(x,y,D,fx):
    """
    函数功能：学习新的分类器

    @param:x  数据 type: 1-D list
    @param:y  标签 type: 1-D list
    @param:D  权值分布 type: 1-D list
    @param:fx 已有分类器，格式[[a1,±G1],...,[am,Gm]]
              注意：有两种分类器[1,-1],[-1,1]
                +Gm：if x < Gm: y = 1  
                -Gm：if x < Gm: y = -1
    """
    
    '学习新的分类器'
    gmx = []  #所有可能的基学习器
    err_base = []  #所有可能的基学习器 对应的误差

    # 1，-1基分类器  
    pred_new = [0]*len(y) # 新分类器的预测结果
    pred_new_base = [] #新基学习器的预测结果
    for j in range(len(x) - 1):
        gmx.append(0.5*(x[j] + x[j+1])) #不考虑全为1或全为-1的情况
        pred_new_base = [1]*int(math.ceil(gmx[-1])) + [-1]*int(len(x) - math.ceil(gmx[-1]))
        # 预测误差em为预测错误项的权重值之和。预测结果和 y 相乘，为-1的元素（异号）即代表着预测错误
        compare_pred_y = list(map(lambda x: x[0]*x[1], zip(pred_new_base,y)))
        # 根据err_list 为 -1的元素的索引，取得对应权重 D 的值并求和
        err = 0
        for k in range(len(compare_pred_y)):
            if compare_pred_y[k]==-1:err += D[k]
        if not err: break  # 如果直接找到了分类错误为0的基分类器就不再往下找了
        err_base.append(err) # 更新基学习器的误差 list
    # -1，1基分类器
    pred_new = [0]*len(y) # 新分类器的预测结果
    pred_new_base = [] #新基学习器的预测结果
    for j in range(len(x) - 1):
        gmx.append(-0.5*(x[j] + x[j+1]))
        pred_new_base = [-1]*int(math.ceil(-gmx[-1])) + [1]*int(len(x) - math.ceil(-gmx[-1]))
        # 预测误差em为预测错误项的权重值之和。预测结果和 y 相乘，为-1的元素（异号）即代表着预测错误
        compare_pred_y = list(map(lambda x: x[0]*x[1], zip(pred_new_base,y)))
        # 根据err_list 为 -1的元素的索引，取得对应权重 D 的值并求和
        err = 0
        for k in range(len(compare_pred_y)):
            if compare_pred_y[k]==-1:err += D[k]
        if not err: break  # 如果直接找到了分类错误为0的基分类器就不再往下找了
        err_base.append(err) # 更新基学习器的误差 list

    # 错误思路❌：每次选择最小误差对应的分类器，没有排除掉已经被选择过的基学习器，废话，2.5和8.5分的最好，肯定每次都选到这两个，于是就陷入了这两个基学习器狂刷存在感的僵局
    '忽略已经被选择过的基学习器'
    sorted_err = np.argsort(err_base)
    print(sorted_err)
    p = sorted_err[0] # p 误差最小的基分类器的在 gmx 中的索引。从第一个开始，当是已经被选择的分类器时，就跳过
#    print(gmx,err_base,sorted_err)

    if fx:
        for n in range(len(fx)):
            if fx[n][1] != gmx[sorted_err[n]]:
                p = sorted_err[n]
                break  #坑 ⭕️ 没有 break
    
    '得到 新分类器fx'
    Gmx = gmx[p]  # 新的基学习器  #坑 ⭕️  sorted_err[p]
    em = err_base[p]  # 新的基学习器的误差
    am_new = 1/2 * math.log((1-em)/em)
    am_new = int(am_new*10000)/10000 #保留四位小数
    fx.append([am_new,Gmx]) # 新分类器
    '计算新分类的预测结果'
    pred_fx_new = [0]*len(y)
    pred_Gm = []#每一个基学习器的预测结果
    # 计算每一个已有的基学习器预测结果pred_am 并和 已有的分类器 fx 预测结果pred_fx 累加
    for i in range(len(fx)):
        if fx[i][1] > 0:  # [1,-1]分类器
            pred_Gm = [fx[i][0]]*int(math.ceil(fx[i][1])) + [-fx[i][0]]*int(len(x) - math.ceil(fx[i][1]))
        else : # [-1,1]分类器
            pred_Gm = [-fx[i][0]]*int(math.ceil(-fx[i][1])) + [fx[i][0]]*int(len(x) - math.ceil(-fx[i][1]))

        pred_fx_new = list(map(lambda x: x[0]+x[1], zip(pred_fx_new, pred_Gm)))    
    pred_fx_new = np.sign(pred_fx_new)
    return fx,pred_fx_new

def calW(D_old,am,y,y_Gmx):
    """
    函数功能：计算 权值分布

    @param:D_old  当前权值分布 type: 1-D list
    @param:am     最新基分类器的系数 type:  float
    @param:y_Gmx  分类器的预测标签 type: 1-D list
    @param: y     标签 type:1-D list
    """
    Zm_item = []  #计算 规范化因子 参考李航《统计学习方法》公式(8.5)
    for i in range(len(y)):
        Zm_item.append(D_old[i]*math.exp(-am*y[i]*y_Gmx[i]))
    Zm = sum(Zm_item)
    D_new = [int(i/Zm*100000)/100000 for i in Zm_item] # 控制精度为五位数，round 不好用，round(13.949999999999999,2) = 13.949999999999999
    return D_new
    
def AdaBoost(x,y,maxIterNum,errorThreshold):
    """
    函数功能：得到 分类器

    @param:x  数据 type: 1-D list
    @param:y  标签 type: 1-D list
    @param:maxIterNum     基学习器数量 type: float
    @param:errorThreshold 分类器误差阈值 type:float
    """
    D1 = [1.0/len(x)]*len(x)
    fx_err = 0
    D = D1
    fx = []
    for i in range(maxIterNum):
        fx,y_fx = get_Gmx(x,y,D,fx)
        '计算分类器的误差'
        compare_fx_y = list(map(lambda x: x[0]*x[1], zip(y, y_fx)))
        err_fx = 0
        for k in range(len(y)):
            if compare_fx_y[k]==-1:
                err_fx += D[k]
        if err_fx < errorThreshold:break
    
        am = fx[-1][0]
        D = calW(D,am,y,y_fx)
    return fx
    
if __name__ == '__main__':
    x, y = loadData()
    Gx = AdaBoost(x,y,maxIterNum = 8,errorThreshold = 0.01)
    print(Gx)   
    
"""
little thoughts：
本来想采取二维数据的，不过这样每一次都要到一个线性分类器，就可以直接掉包用 SVM ，LR等，然后再集成起来。
突然就领悟到 AdaBoost 是一种思想一种框架一个壳，它只是组合弱分类器，它并不产生基分类器，
实际上基分类器是什么，是需要自己选择的。
集成算法干的是优化的事儿。
怎么有点像投资组合，你有投基金啊活期定期啊股票啊，才有优化组合这个事。
"""

"""注意⭕️：AdaBoost 是不是稳定的算法：
当第一次同时出现两个误差并列最小即2.5和8.5时，
李航《统计学习方法》P140 例8.1选择的是2.5，于是三次就得到了误差为0的分类器
本算法选择的是8.5，但第二次就选择了2.5（所以说对于同一时间出现的两个误差相同的基分类器，会依次选中它们，不会错过。）

实验：改变标签使得当第一次有三个同样误差最小的基学习器，看最后是怎样
y = [-1,-1,-1,-1,-1,1,-1,1,1,1]
本算法选择了第一个，但是可以看到第二个基学习器就不是-7.5了，权值分布已经使得-8.5在第二次选择基学习器时不那么优秀了
fx = [[1.0986, -6.5], [1.4166, -4.5], [0.5893, 5.5]]

如果第一次选的是 -7.5 呢
对于确定程序产生的分类器是确定的，可是不同程序实现的分类器得到的可能就不是一样的。
目前认为是不稳定的。
⭕️：问老大：Google 不到和 AdaBoost 相关的内容
"""
"""
注意⭕️
所有基学习器一共18种，每种基学习器的预测标签是确定的，所以预测正确率一定，而根据 P139 公式(8.8)可知，它们的分类误差和权值矩阵有关系⭕️
"""

"""可以做的改进：
1. 不考虑全为1或全为-1的情况
此时可以把 gmx 首尾分别追加两个元素即
0，此时的预测标签为[1]*len(x)
-9，此时的预测标签为[-1]*len(x)
此时可以在 gmx 第一个元素

2. 每调用一次get_Gmx()都会重新计算一次 gmx 和 err_base
实际上只需要计算一次，按照误差由小到大排好序即可。单独写一个函数获取所有可能基学习器及其误差 list，作为get_Gmx()参数即可

"""
