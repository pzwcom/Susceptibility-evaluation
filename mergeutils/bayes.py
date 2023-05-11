from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from mergeutils.Figure import display_roc
import pandas as pd
import joblib 
from ast import Str
from typing import Literal




def bayes_RandomForest(train_sample:Literal,trainsize:float,modeldir:Str,param_grid=None):
        '''
        train_sample:抽取的滑坡点和非滑坡点合集
        trainsize:训练集比例
        modeldir:模型保存位置
        param_grid:参数字典
        '''
        X=train_sample[0][:,2:]
        y=train_sample[1]
        def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
            val = cross_val_score(
                RandomForestClassifier(n_estimators=int(n_estimators),
                                    min_samples_split=int(min_samples_split),
                                    max_features=min(max_features, 0.999),  # float
                                    max_depth=int(max_depth),
                                    random_state=2),
                X, y, scoring='roc_auc', cv=5
            ).mean()
            return val
        # 步骤二：确定取值空间
        pbounds = {'n_estimators': (10, 250),  # 表示取值范围为10至250
                'min_samples_split': (2, 25),
                'max_features': (0.1, 0.999),
                'max_depth': (5, 15)}

        # 步骤三：构造贝叶斯优化器
        optimizer = BayesianOptimization(
            f=rf_cv,  # 黑盒目标函数
            pbounds=pbounds,  # 取值空间
            verbose=1,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
            random_state=1,
        )
        optimizer.maximize(  # 运行
            init_points=5,  # 随机搜索的步数
            n_iter=25,  # 执行贝叶斯优化迭代次数
        )
        print(optimizer.res)  # 打印所有优化的结果
        print(optimizer.max)  # 最好的结果与对应的参数
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