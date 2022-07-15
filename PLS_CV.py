# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:48:50 2022
参数说明，Xcal,ycal,最大成分个数（默认值10），交叉验证，交叉验证重复几次，是否图片显示最小MSE对应的comp个数
@author: ZSP
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
def PLS_CV(Xcal,ycal,max_comp=10,cv_split=5,cv_repeat=2,show_fig=1):
    cv = RepeatedKFold(n_splits=cv_split, n_repeats=cv_repeat, random_state=1)
    mse=[0]
    '''
    n= len(Xcal)
    score = -1*model_selection.cross_val_score(PLSRegression(n_components=1),np.ones((n,1)), 
               ycal.loc[:,0], cv=cv, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    '''
    for i in np.arange(1, max_comp+1):
        pls = PLSRegression(n_components=i)
        score = -1*model_selection.cross_val_score(pls, Xcal, 
                     ycal, cv=cv,scoring='neg_mean_squared_error').mean()
        mse.append(score)
        #返回最小mse与对应的component
        min_mse=min(mse for mse in mse if mse>0)
        min_comp=mse.index(min(mse for mse in mse if mse>0))
        #绘图
        if show_fig==1:
            plt.plot(mse)
            plt.plot(mse)
            plt.xlabel('Number of PLS Components')
            plt.ylabel('MSE')
            plt.title('Model-Compoents')
    return min_mse, min_comp