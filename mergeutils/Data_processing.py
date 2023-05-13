from typing import Literal
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
def Minmaxscaler(data:Literal):
    scaler = MinMaxScaler() #实例化
    scaler = scaler.fit(data) #fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(data) #通过接口导出结果
    return result
def StandScaler(data:Literal):
    scaler = StandardScaler() #实例化
    scaler = scaler.fit(data) #fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(data) #通过接口导出结果
    return result
def getpearson(train_sample_X:np.array,files:Literal):
    data = pd.DataFrame()
    i=0
    for file in files:
        data.loc[:, file] = train_sample_X[:,i]
        i=i+1
    pearson=data.corr(method='pearson')
    return pearson
def result_classify(result:np.ndarray):
    result=np.where((result<=0.1) & (result>0),1,result)
    result=np.where((result<=0.15) & (result>0.1),2,result)
    result=np.where((result<0.15)&(result>0.25),3,result)
    result=np.where((result<=0.25)&( result>0.4),4,result)
    result=np.where((result<=1)&(result>0.4),5,result)
    return result