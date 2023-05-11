

import joblib 
import numpy as np
from ast import Str
from typing import Literal
from mergeutils.Init import set_seeds
from mergeutils.Data_processing import Minmaxscaler,StandScaler
#机器学习模块
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#显示模块
from mergeutils.Figure import display_roc
#s深度学习模块
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.models import *
from numpy import array
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
class Machine_Learning:
    def RandomForest(train_sample:Literal,trainsize:float,modeldir:Str,param_grid=None):
        '''
        train_sample:抽取的滑坡点和非滑坡点合集
        trainsize:训练集比例
        modeldir:模型保存位置
        param_grid:参数字典
        '''
        X=train_sample[0][:,2:]
        y=train_sample[1]
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=trainsize,random_state=42)
        random_model=RandomForestClassifier(random_state=100)
        #格网法确定超参数
        if(param_grid!=None):
            grid = GridSearchCV(random_model,param_grid,cv = 4,scoring = 'roc_auc',n_jobs =1)
            #在训练集上训练
            grid.fit(X_train,y_train)
            #作图
            # means = grid.cv_results_['mean_test_score']
            # plt.plot(param_grid['n_estimators'],means,label = "RDF")
            # plt.legend()
            # plt.show()
            # 返回最优的训练器
            random_model = grid.best_estimator_
            print(random_model)
            print(grid.best_score_)
            #输出最优训练器的精度
        # scores=grid.cv_results_
        # print(scores['mean_test_score'])
        # drawparafigure(scores['mean_test_score'],param_grid['max_depth'])
        clt=random_model.fit(X_train,y_train)
        pre_test=clt.predict_proba(X_test)
        # 计算因子重要性
        # importances = clt.feature_importances_
        # for f in range(X_train.shape[1]):
        #     print(importances[f])
        # 展示roc
        display_roc(y_test,pre_test,'randforest_roc')
        
        del pre_test
        clt=random_model.fit(X,y)
        joblib.dump(clt,modeldir,compress=3) 
        return clt

    def Logistic(train_sample:Literal,trainsize:float,modeldir:Str,param_grid=None):
            '''
            train_sample:抽取的滑坡点和非滑坡点合集
            trainsize:训练集比例
            modeldir:模型保存位置
            param_grid:参数字典
            '''
            X=Minmaxscaler(train_sample[0][:,2:])
            y=train_sample[1]
            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=trainsize)
            random_model=LogisticRegression()
            #格网法确定超参数
            if(param_grid!=None):
                grid = GridSearchCV(random_model,param_grid,cv = 4,scoring = 'roc_auc',n_jobs = -1)
                #在训练集上训练
                grid.fit(X_train,y_train)
                #返回最优的训练器
                random_model = grid.best_estimator_
                print(random_model)
                print(grid.best_score_)
                #输出最优训练器的精度
            # scores=grid.cv_results_
            # print(scores['mean_test_score'])
            # drawparafigure(scores['mean_test_score'],param_grid['max_depth'])
            clt=random_model.fit(X_train,y_train)
            pre_test=clt.predict_proba(X_test)
            # 计算因子重要性
            # importances = clt.feature_importances_
            # for f in range(X_train.shape[1]):
            #     print(importances[f])
            # 展示roc
            display_roc(y_test,pre_test,'Logistic_roc')
            del pre_test
            clt=random_model.fit(X,y)
            joblib.dump(clt,modeldir,compress=3) 
            return clt


    def SVM(train_sample:Literal,trainsize:float,modeldir:Str,param_grid=None):
            '''
            train_sample:抽取的滑坡点和非滑坡点合集
            trainsize:训练集比例
            modeldir:模型保存位置
            param_grid:参数字典
            '''
            X=Minmaxscaler(train_sample[0][:,2:])
            y=train_sample[1]
            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=trainsize)
            random_model=SVC(probability=True)
            #格网法确定超参数
            if(param_grid!=None):
                grid = GridSearchCV(random_model,param_grid,cv = 4,scoring = 'roc_auc',n_jobs = -1)
                #在训练集上训练
                grid.fit(X_train,y_train)
                #返回最优的训练器
                random_model = grid.best_estimator_
                print(random_model)
                print(grid.best_score_)
                #输出最优训练器的精度
            # scores=grid.cv_results_
            # print(scores['mean_test_score'])
            # drawparafigure(scores['mean_test_score'],param_grid['max_depth'])
            clt=random_model.fit(X_train,y_train)
            pre_test=clt.predict_proba(X_test)
            # 计算因子重要性
            # importances = clt.feature_importances_
            # for f in range(X_train.shape[1]):
            #     print(importances[f])
            # 展示roc
            display_roc(y_test,pre_test,'svm_roc')
            del pre_test
            clt=random_model.fit(X,y)
            joblib.dump(clt,modeldir,compress=3) 
            return clt
    
    
        
    
        
class Deep_Learning:
   
    def CNN(train_sample:array,trainsize:int,outdir:Str):
        '''
        train_sample:采样好的样本集
        trainsize:训练集比例
        outdir:输出文件夹
        return:CNN模型
        '''
        set_seeds(1)
        X=np.expand_dims(Minmaxscaler(train_sample[0][:,2:]),-1)
        Y=np.expand_dims(train_sample[1],-1)
        X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=trainsize)
        #构建神经网络
        model=Sequential()
        model.add(Conv1D(64,3,1,padding="same",activation='relu',input_shape=(len(X[0,:]),1)))
        model.add(Conv1D(64,3,1,padding="same",activation='relu',))
        model.add(MaxPool1D(2))
        model.add(Dropout(0.5))
        model.add(Conv1D(64,3,1,padding="same",activation='relu',))
        model.add(Conv1D(64,3,1,padding="same",activation='relu',))
        model.add(MaxPool1D(2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=150,activation='relu'))
        model.add(Dense(units=2,activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.0005),metrics = ['accuracy'])
        model.summary()
        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=50)
        pre_test=model.predict(X_test)
        display_roc(y_test,pre_test,'cnn_roc')
        model.save(outdir)
        del pre_test
        return model
