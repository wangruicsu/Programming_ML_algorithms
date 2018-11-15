#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:39:19 2018
@逻辑回归分类 代码实现
@dataset：马的疝气病症数据集Horse Colic database
          任务需求：预测病马的死亡率
          http://archive.ics.uci.edu/ml/datasets/Horse+Colic
          data files: 
            -- horse-colic.data: 300 training instances 
            -- horse-colic.test: 68 test instances 
          Number of Attributes:27个
          Missing Values? yes
          abdominalDistension:腹胀   An IMPORTANT parameter.
          
          surgery：1（手术过）2（没有手术治疗过）
          age：1（成年马）2(小马 6个月以下)
          HospitalID：马的编号（如果马被治疗> 1次，可能不是唯一的）
          rectalTemperature:直肠温度 正常37.8
          pulse：脉搏，成年马30-40，运动马匹较低20-25，具有疼痛病变或患有循环休克的动物可能具有升高的心率
          respiratoryRate：呼吸频率，正常8-10，波动很大，存疑
          extremitiesTemperature：四肢温度，1 = Normal，2 = Warm，3 = Cool，4 = Cold，低表明可能有休克，高则与直肠温度身升高有关
          peripheralPulse：外围脉冲，1 = normal，2 = increased，3 = reduced，4 = absent 
          mucousMembranes：粘膜（越大越严重），1 = normal pink，2 = bright pink，3 = pale pink，4 = pale cyanotic，5 = bright red / injected，6 = dark cyanotic 
          capillaryReFillTime：毛细管再充填时间，1 = < 3 seconds，2 = >= 3 seconds 
          pain：对马的疼痛程度的主观判断，疼痛越多，需要手术的可能性就越大，疼痛的事先治疗可能会在一定程度上掩盖疼痛程度。1 = alert, no pain，2 = depressed，3 = intermittent mild pain，4 = intermittent severe pain，5 = continuous severe pain 
          peristalsis：肠蠕动，当马中毒时，肠蠕动减少，1 = hypermotile，2 = normal，3 = hypomotile，4 = absent 
          abdominalDistension：腹胀，腹胀的动物可能会疼痛并且肠道蠕动减少，有严重腹胀的马可能需要手术才能减轻压力，1 = none，2 = slight，3 = moderate，4 = severe 
          nasogastricTube：鼻胃管，胃中的气体可能会让马感到不适，1 = none，2 = slight，3 = significant 
          nasogastricReflux：鼻胃反流，回流量越大，从肠道其余部分流体通道出现严重阻塞的可能性越大，1 = none，2 = > 1 liter，3 = < 1 liter 
          nasogastricRefluxPH：鼻胃反流PH，比例从0到14，其中7为中立，正常值在3到4范围内。
          rectalExaminationFeces：直肠检查 - 粪便，没有粪便可能表示阻塞，1 = normal，2 = increased，3 = decreased，4 = absent 
          abdomen：腹部，3可能是由机械撞击引起的阻塞，通常在医学上进行治疗。4和5表示手术病变，1 = normal，2 = other，3 = firm feces in the large intestine，4 = distended small intestine，5 = distended large intestine 
          packedCellVolume：填充细胞体积。血液中按体积计的红细胞数，正常范围是30到50.随着循环受损或动物变得脱水，水平上升。
          totalProtein：总蛋白质，正常值位于6-7.5（gms / dL）范围内。值越高，脱水越大
          abdominocentesisAppearance：腹腔穿刺术的外观，正常的液体是清澈的，而混浊或血清阳性表明肠道受损，1 = clear，2 = cloudy，3 = serosanguinous 
          abdomcentesisTotalProtein：总蛋白，蛋白质水平越高，肠道受损的可能性越大。
          outcome：马的最终情况，1 = lived，2 = died，3 = was euthanized 
          surgicalLesion：是否手术病变， 当这些病理数据不为这些情况采集时时这个变量没有意义，1 = Yes，2 = No 
          lesionType1：
          lesionType2：
          lesionType3：
          cp_data：1 = Yes，2 = No 

@author: raine
"""

import math
import numpy as np
import pandas as pd

# 加载数据
def loadData(filename):
    colname = ["surgery", "age", "HospitalID", "rectalTemperature", "pulse" \
               , "respiratoryRate", "extremitiesTemperature", "peripheralPulse", "mucousMembranes"\
               , "capillaryReFillTime", "pain", "peristalsis", "abdominalDistension", "nasogastricTube"\
               , "nasogastricReflux", "nasogastricRefluxPH", "rectalExaminationFeces", "abdomen"\
               , "packedCellVolume", "totalProtein", "abdominocentesisAppearance", "abdomcentesisTotalProtein"\
               , "outcome", "surgicalLesion", "lesionType1", "lesionType2", "lesionType3", "cp_data"]
    
    Data = pd.read_csv(filename, names = colname,sep = ' ')
    # 文件中的缺失值都是以？表示的
    Data.replace("?", np.nan, inplace = True)
    # 删除标签缺失的样本
    Data = Data[Data['outcome'].notnull()]
    #处理缺失值，全部填0
    Data.fillna(0, inplace = True)
    #处理标签[存活，死亡，安乐死] → [存活，死亡]
    Data.outcome.replace("3","2",inplace = True)
    #统一数据类型
    for name in colname:
        Data[name] = Data[name].astype('float')
    #删除 ID 列
    #Data.drop("HospitalID",axis = 1,inplace=True)
    
    labels = pd.DataFrame(Data.outcome).as_matrix()
    Data.drop("outcome", axis = 1, inplace = True)
    Data = Data.as_matrix()
    return Data, labels

def sigmoid(x):
    for i in range(len(x)):
        x[i] = 1.0/(1+math.exp(-x[i]))
    return x

def softmax(x):
    return 1 if x > 0.5 else 2
    
#梯度上升算法，数据结构：二维数组 array
def gradAscent(dataArr, classLabels, numIter = 500):
    #初始化权重 weights
    rows,columns = np.shape(dataArr)
    weights = np.ones((columns,1))

    alpha = 0.001    
    for i in range(numIter):
        #print(np.dot(dataArr, weights))
        h = sigmoid(np.dot(dataArr, weights))
        grad = np.dot(dataArr.transpose(),(classLabels - h))
        weights = weights + alpha*grad
    return weights

def stoGradAscent(dataArr, classLabels, numIter = 50):
    #初始化权重 weights
    rows,columns = np.shape(dataArr)
    weights = np.ones(columns)

    alpha = 0.001    
    for i in range(numIter):
        for j in range(len(classLabels)):
            h = 1.0/(1+math.exp(-np.dot(dataArr[j],weights))) #维度：标量
            grad = dataArr[j].transpose()*(classLabels[i]-h) #维度：27*1
            weights = weights + alpha*grad
    return weights

def test(data, labels, weights):
    errCount = 0
    for i in range(len(data)):
        if softmax(1.0/(1+math.exp(-np.dot(data[i],weights)))) != labels[i]:
            errCount += 1
    errRate = errCount/len(labels)
    return errRate
    
if __name__ == '__main__':
    filename_train = "horse-colic.data.txt"
    filename_test = "horse-colic.test.txt"
    trainData, trainLabels = loadData(filename_train)
    testData, testLabels = loadData(filename_test)

    #采用全批量的梯度上升法。所有的预测结果均为1，真正类率和假正例率均为1.
    weights = gradAscent(trainData, trainLabels)
    errRate = test(testData, testLabels, weights)
    print("the error rate of  all data test is: %f"%errRate)
    #采用随机梯度上升法
    weights = stoGradAscent(trainData, trainLabels)
    errRate = test(testData, testLabels, weights)
    print("the error rate of SGD test is: %f"%errRate)