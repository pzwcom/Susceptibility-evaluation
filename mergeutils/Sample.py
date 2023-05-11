import numpy as np
from typing import Literal
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
class Sample_methods:
    def smote(hua:Literal,feihua:Literal):
        '''
        Description:传入已经叠加好的滑坡矩阵和非滑坡矩阵进行重采样。
        hua:滑坡点样本集
        feihua:非滑坡点样本集
        return:采样好的两个集分开X,y
        '''
        sm = SMOTE(random_state=666)
        allsample_features=np.vstack((feihua,hua))
        y=allsample_features[:,len(allsample_features[0])-1]
        X=allsample_features[:,:len(allsample_features[0])-1]
        del allsample_features
        X_res, y_res = sm.fit_resample(X, y)          
        result=[X_res,y_res]                    
        del X_res,y_res
        return result
    def equal(hua:Literal,feihua:Literal):
        '''
        Description:传入已经叠加好的滑坡矩阵和非滑坡矩阵进行重采样。
        hua:滑坡点样本集
        feihua:非滑坡点样本集
        return:采样好的两个集分开X,y
        '''
        print('start Sample')
        rand_arr = np.arange(feihua.shape[0])
        np.random.seed(100)#复现结果给定随机种子
        np.random.shuffle(rand_arr)
        hua_count=len(hua[:,0])
        feihua=feihua[rand_arr[0:hua_count*10]]
        allsample_features=np.vstack((feihua,hua))
        rand_arr = np.arange(allsample_features.shape[0])
        np.random.seed(100)#复现结果给定随机种子
        np.random.shuffle(rand_arr)
        allsample_features=allsample_features[rand_arr]
        y=allsample_features[:,len(allsample_features[0])-1]
        X=allsample_features[:,:len(allsample_features[0])-1]      
        result=[X,y]                   
        del X,y,allsample_features
        return result
    def randomUnderSampler(hua:Literal,feihua:Literal):
        '''
        Description:传入已经叠加好的滑坡矩阵和非滑坡矩阵进行重采样。
        hua:滑坡点样本集
        feihua:非滑坡点样本集
        return:采样好的两个集分开X,y
        '''
        rdu=RandomUnderSampler(random_state=0)
        allsample_features=np.vstack((feihua,hua))
        y=allsample_features[:,len(allsample_features[0])-1]
        X=allsample_features[:,:len(allsample_features[0])-1]
        del allsample_features
        X_res, y_res= rdu.fit_resample(X, y)
        print(sorted(Counter(y_res).items()))
        result=[X_res,y_res]                    
        del X_res,y_res
        return result
    def allSampler(hua:Literal,feihua:Literal):
        '''
        Description:传入已经叠加好的滑坡矩阵和非滑坡矩阵进行重采样。
        hua:滑坡点样本集
        feihua:非滑坡点样本集
        return:采样好的两个集分开X,y
        '''
        allsample_features=np.vstack((feihua,hua))
        y=allsample_features[:,len(allsample_features[0])-1]
        X=allsample_features[:,:len(allsample_features[0])-1]
        result=[X,y]                    
        return result

    def EasyEnsembleSampler(hua:Literal,feihua:Literal):
        '''
        Description:传入已经叠加好的滑坡矩阵和非滑坡矩阵进行重采样。
        hua:滑坡点样本集
        feihua:非滑坡点样本集
        return:采样好的两个集分开X,y
        '''
        Eeb=RandomUnderSampler(random_state=42)
        allsample_features=np.vstack((feihua,hua))
        y=allsample_features[:,len(allsample_features[0])-1]
        X=allsample_features[:,:len(allsample_features[0])-1]
        X_res, y_res= Eeb.fit_resample(X, y)
        print(sorted(Counter(y_res).items()))
        result=[X_res,y_res]     
        result=[X,y]                    
        return result