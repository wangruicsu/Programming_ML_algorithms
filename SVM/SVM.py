#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习算法：SVM
数据集数据类型：:mat

done：
完成：自写线性核函数的 SVM（也即是没有核函数）

对比：sklearn 的 linear 核函数、rbf 核函数
 ------------------------------------
| 核函数                  |      正确率|
 ------------------------------------
|自编 SVM 线性核函数       |       66% |
|sklearn 的 linear 核函数 |       31% |
|sklearn 的 rbf 核函数    |       90% |
 ------------------------------------

"""

import pandas as pd
import numpy as np
from numpy import *
from collections import Counter
from sklearn.svm import SVC

# 载入数据，数据类型：二维矩阵 mat
def loadData(filename): 
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return mat(dataMat),mat(labelMat) #返回数据特征和数据类别

# 将所有用到的数据都打包在一个对象中，该对象相当于一个数据结构。
class dataStruct:
    # 输入包括：数据集dataMat，标签classLabels，常数 C，容错率toler
    def __init__(self,dataMat,classLabel,C,toler):
        self.X = dataMat
        self.labelMat = classLabel
        self.C = C   # 常数 C：C>= alpha>=0。用于控制最大间隔和保证大部分的点的函数间隔小于1.0。
        self.toler = toler
        self.m = shape(dataMat)[0] #取行数
        self.alpha = mat(zeros((self.m,1)))
        self.b = 0
        self.errCache = mat(zeros((self.m,2))) #第一位：是否有效的标志位，是否更新过

# 计算第 k 个样本的预测值和标签之间的误差
def calErr(DA,k):
#    print(DA.alpha,DA.labelMat,multiply(DA.alpha,DA.labelMat))
    fxk = float(multiply(DA.alpha,DA.labelMat).T*DA.X*DA.X[k,:].T) + DA.b
    errk = fxk - float(DA.labelMat[k])   
    return errk

# 更新dataStruct中的 Ek
def updateEk(DA,k):
    Ek = calErr(DA,k)
    DA.errCache[k] = [1,Ek]
    
#  计算aj_new_unc
def calAlphaJ(ai,aj,yi,yj,Ei,Ej,xi,xj):
    Kii = 1
    Kjj = 1
    Kij = (yi*yj*xi*xj.T)[0,0]
    n = Kii+Kjj-2*Kij
    aj_new_unc = aj+yj*(Ei-Ej)/n
    return aj_new_unc

# 计算 aj 的上下边界L，H
def calLH(C,ai,aj,yi,yj):
    if yi != yj:
        L = max(0,aj-ai)
        H = min(C,C+aj-ai)
    else:
        L = max(0,aj+ai-C)
        H = min(C,aj+ai)
    return L,H
    
# 考虑上下边界 L，H，调整aj
def adjustAlphaJ(aj,L,H):
    if aj < L : aj = L
    if aj > H : aj = H
    return aj

#  根据 aj，计算ai_new
def calAlphaI(ai,aj,yi,yj,aj_new):
    ai_new = ai+yi*yj*(aj-aj_new)
    return ai_new

# 选择 aj
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
        print("vaildErrCache > 1",j)
    # 在第一次挑选 aj 时，除了 Ei，其他的 Ex 还都是0，所以随机挑选
    else:
        while(j==i):
            print(DA.m)
            j =  int(random.randint(0,DA.m) )
        Ej = calErr(DA,j)
        print("随机挑选 j",j)
    print("select j",j)
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
    if case1 or case2 and case3 and case4:
        # 选取 aj
        Ej,j = selectJ(DA,i,Ei)
        print(" i：",i,"j：",j)
        
        aj = DA.alpha[j,0] # 相当于copy一份原数据，仅用于直观的计算不影响原数据。值为标量。
        yj = DA.labelMat[j,0]
        xi = DA.X[i,:]
        xj =  DA.X[j,:]
       
        #更新 aj，Ej
        aj_new_unc = calAlphaJ(ai,aj,yi,yj,Ei,Ej,xi,xj)
        L,H = calLH(DA.C,ai,aj,yi,yj)
        DA.alpha[j,0] = adjustAlphaJ(aj_new_unc,L,H)
        
        # 内环退出条件1①,aj 无法取到合理的值。计算aj_new（参考《统计学习方法》p127公式7.108）
        if (L==H) and (DA.alpha[j,0]!=L):
            return 0
        # 内环退出条件②，aj 更新量过少
        print(" yi",yi,"\n","yj",yj,"\n","L:",L,"\n","H:",H,"\n",'aj_new_unc',aj_new_unc,"\n","aj_new:",DA.alpha[j,0],"\n","ai",ai,"\n","aj:",aj,"\n","ai_new-aj",DA.alpha[j,0] -aj,"\n")
        if abs(DA.alpha[j,0] -aj)< 0.00001:
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
    # 创建数据对象，一切数据上的更新迭代都在该对象中。
    DA = dataStruct(dataMat,classLabel.T,C,toler)
    iters = 0
    isAlphaChanged = 0  # 当 参数a 没有更新时就退出循环
    wholeSet = True
    while((iters < maxIters) and (isAlphaChanged or wholeSet)):
        isAlphaChanged = 0
        if wholeSet:
            for i in range(DA.m):
                isAlphaChanged += innerLoop(DA,i)
                print("fullset, iters: %d, isAlphaChanged: %d" %(iters, isAlphaChanged))
            iters += 1
        else:
            notBound = nonzero((DA.alpha.A > 0)*(DA.alpha.A < C))[0] # 取 0<a<C 的索引 
            for i in notBound:
                isAlphaChanged += innerLoop(DA,i)
                print("not bound, iters: %d, isAlphaChanged: %d" %(iters, isAlphaChanged))
            iters += 1
        if wholeSet: wholeSet = False
        elif(isAlphaChanged == 0): wholeSet = True
        print("iteration number: %d" %iters)
    return DA.alpha,DA.b    

def calW(alpha,dataMat,labelMat):
    X = dataMat
    y = labelMat.T
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alpha[i]*y[i],X[i,:].T)
    return mat(w)
    
    
if __name__ == '__main__':
    filename_train = "train_data.txt"  
    filename_test = "test_data.txt"  
    
    trainMat,trainLabel = loadData(filename_train)
    testMat,testLabel = loadData(filename_test)
    
    alphas,b = SMO(trainMat, trainLabel, 0.8, 0.001, 4000) #通过SMO算法得到b和alpha
    w = calW(alphas,trainMat,trainLabel)
    
    # 预测
    pre_y = [] #预测得到的误差
    pre_Err = []
    for i in range(shape(testMat)[0]):
        pre_yi = (testMat[i]*w+b)[0,0]
        if pre_yi > 0: pre_yi = 1
        else: pre_yi = -1
        pre_y.append(pre_yi)
    for i in range(len(pre_y)):
        pre_Err.append((pre_y[i] - (testLabel.T)[i])[0,0])
    result = pd.value_counts(pre_Err) # 分类误差情况
    print("\n b \n",b)
    print("\n alphas \n",alphas.T)
    print("\n 我的预测结果误差为 \n", result)
    # TODO：
    # 为什么每次挑选到的 aj 基本一样？当然了，大部分的 a 对应的样本不是支持向量，极少数对应于支持向量 x的 a 才有值，才会被选择更新
    # 为什么 L==H 总是成立
    '''调用sklearn 库作对比'''
    svm_linear = SVC(C = 1.0,kernel='linear',tol = 0.001,max_iter=4000)
    svm_rbf = SVC(C = 1.0,kernel='rbf',tol = 0.001,max_iter=4000)

    # 处理trainLabel的 shape
    z = trainLabel.T
    trainLabel_array = []
    for i in range(len(z)):
        trainLabel_array.append(z[i,0])
    trainLabel_array = np.array(trainLabel_array)
    # 处理testLabel的 shape
    z = testLabel.T
    testLabel_array = []
    for i in range(len(z)):
        testLabel_array.append(z[i,0])
    testLabel_array = np.array(testLabel_array)
    # SVM 进行预测
    svm_linear.fit(trainMat, trainLabel_array)
    svm_linear_pred = svm_linear.predict(testMat)
    pre_Err_svm_linear = svm_linear_pred - testLabel_array
    result_svm_linear = pd.value_counts(pre_Err_svm_linear)
    print("\n sklraen linear 的预测结果误差为 \n", result_svm_linear)
    svm_rbf.fit(trainMat, trainLabel_array)
    svm_rbf_pred = svm_rbf.predict(testMat)
    pre_Err_svm_rbf = svm_rbf_pred - testLabel_array
    result_svm_rbf = pd.value_counts(pre_Err_svm_rbf)
    print("\n sklraen rbf 的预测结果误差为 \n", result_svm_rbf)