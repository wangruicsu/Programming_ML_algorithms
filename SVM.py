#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM
datatype:mat

Created on Mon Oct 29 08:59:06 2018

@author: raine
"""

from numpy import *

def loadData(filename): #读取数据
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
#        print(lineArr[0],lineArr[1])
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return mat(dataMat),mat(labelMat) #返回数据特征和数据类别

# 将所有用到的数据都打包在一个对象中，该对象相当于一个数据结构。
# 输入包括：数据集dataMat，标签classLabels，常数 C，容错率toler
# 常数 C：C>= alpha>=0。用于控制最大间隔和保证大部分的点的函数间隔小于1.0。
class dataStruct:
    def __init__(self,dataMat,classLabel,C,toler):
        self.X = dataMat
        self.labelMat = classLabel
        self.C = C
        self.toler = toler
        self.m = shape(dataMat)[0] #取行数
        self.alpha = mat(zeros((self.m,1)))
        self.b = 0
        self.errCache = mat(zeros((self.m,2))) #第一位：是否有效的标志位，是否更新过

def calLH(C,ai,aj,yi,yj):
    if yi != yj:
        L = max(0,aj-ai)
        H = min(C,C+aj-ai)
    else:
        L = max(0,aj+ai-C)
        H = min(C,aj+ai)
    return L,H

def calErr(DA,k):
#    print(DA.alpha,DA.labelMat,multiply(DA.alpha,DA.labelMat))
    fxk = float(multiply(DA.alpha,DA.labelMat).T*DA.X*DA.X[k,:].T) + DA.b
    errk = fxk - float(DA.labelMat[k])   
    return errk

def updateEk(DA,k):
    Ek = calErr(DA,k)
    DA.errCache[k] = [1,Ek]
    
  
def calAlphaJ(ai,aj,yi,yj,Ei,Ej,xi,xj):
    Kii = 1
    Kjj = 1
    Kij = (yi*yj*xi*xj.T)[0,0]
    n = Kii+Kjj-2*Kij
    print(n)
    aj_new_unc = aj+yj*(Ei-Ej)/n
    return aj_new_unc
    
# 调整α
def adjustAlphaJ(aj,L,H):
    if aj < L : aj = L
    if aj > H : aj = H
    return aj

def calAlphaI(ai,aj,yi,yj,aj_new):
    ai_new = ai+yi*yj*(aj-aj_new)
    return ai_new

def selectJ(DA,i,Ei):
    # 启发式：选择使|E1-E2|最大的 aj。因为 ai 已定，所以 Ei 已定。
    # 当 Ei 为正，则选择最小的 Ex 作为 Ej，如果 Ei 为负，则选择最大的 Ex 作为 Ej
    DA.errCache[i] = [1,Ei]
    j = i
    Ej = 0
    vaildErrCache = sum(DA.errCache[:,0])
    # 至少更新过两次 Ex，选择使|E1-E2|最大的 aj
    if vaildErrCache > 1:
        c = abs(DA.errCache[:,1] - Ei )
        sortE = argsort(-abs(DA.errCache[:,1] - Ei ),axis = 0)
        j = sortE[0,0]
        Ej = DA.errCache[j,1]
        if i==j:
            j = sortE[1,0]
            Ej = DA.errCache[j,1]     
    # 在第一次挑选 aj 时，除了 Ei，其他的 Ex 还都是0，所以随机挑选
    else:
        while(j==i):
            j =  int(random.uniform(0,DA.m))
        Ej = calErr(DA,j)
    return Ej,j

'''选择第二个 alpha 即 j
第二个变量的选择标准是希望使 aj 有足够大的变化。'''
def innerLoop(DA,i):
    '''return 0：ai、aj 有更新
       return 0：ai、aj 无更新'''
    Ei = calErr(DA,i)
    # 相当于copy一份原数据，仅用于直观的计算不影响原数据。值为标量。
    ai = DA.alpha[i,0]
    yi = DA.labelMat[i,0]

    #  根据 KKT 条件（《统计学习方法》P128 式（7.111-7.113））
    # 第一个变量应该违背 KKT 条件
    # 由于 Ei=g(xi)-yi →→→ g(xi)=Ei+yi
    # 违背 KKT 条件的四种可能
    case1 = ((ai==0) and (yi*Ei < -DA.toler))
    case2 = ((0<ai) and (ai<DA.C) and (yi*Ei < -DA.toler))
    case3 = ((0<ai) and (ai<DA.C) and (yi*Ei > DA.toler))
    case4 = ((ai==DA.C) and (yi*Ei > DA.toler))
    
    case5 = ((ai<DA.C) and (yi*Ei < -DA.toler))
    case6 = ((0<ai) and (yi*Ei > DA.toler))
#    print(case1,case2,case3,case4)
    if case1 or case2 and case3 and case4:
        Ej,j = selectJ(DA,i,Ei)
        aj = DA.alpha[j,0]
        yj = DA.labelMat[j,0]
        xi = DA.X[i,:]
        xj =  DA.X[j,:]
        #更新 aj，Ej
        aj_new_unc = calAlphaJ(ai,aj,yi,yj,Ei,Ej,xi,xj)
        L,H = calLH(DA.C,ai,aj,yi,yj)
        DA.alpha[j,0] = adjustAlphaJ(aj_new_unc,L,H)
#        print("************",L,aj_new_unc,H,adjustAlphaJ(aj,L,H),DA.alpha[j,0])
        # 内环退出条件1①,aj 无法取到合理的值。计算aj_new（参考《统计学习方法》p127公式7.108）
        if (L==H) and (DA.alpha[j,0]!=L):
            return 0
        # 内环退出条件②，aj 更新量过少
        if (DA.alpha[j,0] -aj)< 0.00001:
            return 0
        #更新 ai，Ei
        DA.alpha[i,0] = calAlphaI(ai,aj,yi,yj,DA.alpha[j,0])
        #更新 b。（参考《统计学习方法》p130公式7.115-117）
        K11 = (xi*xi.T)[0,0]
        K12 = (xi*xj.T)[0,0]
        K21 = (xj*xi.T)[0,0]
        K22 = (xj*xj.T)[0,0]
        bi_new = DA.b - Ei-yi*K11*(DA.alpha[i,0]-ai) - yj*K21*(DA.alpha[j,0]-aj) 
        bj_new = DA.b - Ej-yi*K12*(DA.alpha[i,0]-ai) - yj*K22*(DA.alpha[j,0]-aj)
        if 0<DA.alpha[i,0] and DA.alpha[i,0]<DA.C:
            DA.b = bi_new
        elif 0<DA.alpha[j,0] and DA.alpha[j,0]<DA.C:
            DA.b = bj_new
        else:
            DA.b = (bi_new + bj_new)/2.0
        #更新 Ex.
        updateEk(DA,i)
        updateEk(DA,j)
        return 1
    return 0

def SMO(dataMat,classLabel,C,toler,maxIters):
    DA = dataStruct(dataMat,classLabel.T,C,toler)
    iters = 0
    isAlphaChanged = 0
    wholeSet = 1
    while((iters < maxIters) and (isAlphaChanged or wholeSet)):
        wholeSet = 0
        isAlphaChanged = 0
        for i in range(DA.m):
#            print("挑选第 %d 个参数" %i)
            isAlphaChanged += innerLoop(DA,i)
            iters += 1
#            print("fullset, iters: %d, isAlphaChanged: %d" %(iters, isAlphaChanged))
    return DA.alpha,DA.b    


if __name__ == '__main__':
    filename_train = "/Users/raine/Desktop/train_data.txt"  
    dataMat,classLabel = loadData(filename_train)
    alphas,b = SMO(dataMat, classLabel, 200, 0.0001, 10000) #通过SMO算法得到b和alpha
    print(b,'\t')
    print(alphas)
#    #外循环
#    C = 0.6
#    toler = 0.001
#    maxIters = 40
#    alpha,b = SMO(dataMat,classLabel,C,toler,maxIters)
#    print(alpha,b)
    

    
    
##测试selectJ(DA,i,Ei) 的 if 分支
#DA = dataStruct(dataMat,classLabel,C,toler)
#DA.errCache[0] = [1,5]
#DA.errCache[1] = [0,0]
#DA.errCache[2] = [0,1]
#DA.errCache[3] = [0,2]
#DA.errCache[4] = [0,3]
##测试calErr()
#xx = dataMat*dataMat[1,:].T
#ay = multiply(DA.alpha,DA.labelMat).T
#fxk = ay * xx + DA.b
#ayxx = multiply(DA.alpha,DA.labelMat).T*dataMat*dataMat[1,:].T
#ayxxx = ay * xx
#aa = float(multiply(DA.alpha,DA.labelMat).T*DA.X*DA.X[1,:].T) + DS.b
#errk_hand = fxk - float(DS.labelMat[1])
#errk = calErr(DA,1)
#i = 3
#Ei = DA.errCache[i,1]
#Ej,j = selectJ(DA,i,Ei)
#
#iters = 0
#maxIters = 40
#'''退出循环的条件：达到最大循环次数 | alpha 没有更新 | '''
## 测试 calAlphaJ(aj,yj,Ei,Ej,xi,xj)
#ai = DA.alpha[i,0]
#aj = DA.alpha[j,0]
#yi = DA.labelMat[i,0]
#yj = DA.labelMat[j,0]
#xi = DA.X[i,:]
#xj =  DA.X[j,:]
#aj_new_unc = calAlphaJ(ai,aj,yi,yj,Ei,Ej,xi,xj)
## 测试calLH(C,ai,aj,yi,yj) 两个分支
#L,H = calLH(C,ai,aj,yi,yj)
#aj_new = adjustAlphaJ(aj,L,H)
## 测试calAlphaI(ai,aj,yi,yj,aj_new)
#ai_new = calAlphaI(ai,aj,yi,yj,aj_new)
## 测试 b
#K11 = (xi*xi.T)[0,0]
#K12 = (xi*xj.T)[0,0]
#K21 = (xj*xi.T)[0,0]
#K22 = (xj*xj.T)[0,0]
#test =DA.alpha[i,0]
#bi_new = DA.b - Ei-yi*K11*(DA.alpha[i,0]-ai) - yj*K21*(DA.alpha[j,0]-aj) 
#bj_new = DA.b - Ej-yi*K12*(DA.alpha[i,0]-ai) - yj*K22*(DA.alpha[j,0]-aj)
#print(ai_new,aj_new)
#if 0<ai_new and ai_new<DA.C:
#    b = bi_new
#elif 0<aj_new and aj_new<DA.C:
#    b = bj_new
#else:
#    b = (bi_new + bj_new)/2.0
#    
##测试
#test = DA.alpha[j,0]
#print(test)
#LL = 1
#HH = 1
#if (LL==HH) and (test!=LL):
#    print("return")
#
#case1 = ((ai==0) and (yi*Ei < -DA.toler))
#case2 = ((0<ai) and (ai<DA.C) and (yi*Ei < -DA.toler))
#case3 = ((0<ai) and (ai<DA.C) and (yi*Ei > DA.toler))
#case4 = ((ai==DA.C) and (yi*Ei > DA.toler))
#    