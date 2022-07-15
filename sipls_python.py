# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:26:52 2022

@author: ZSP
""" 

from itertools import combinations
from scipy.special import comb, perm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from split import splitspectrum
from PLS_CV import PLS_CV
'''
参数设置：
'''
def sipls_py(Xcal,ycal,Xtest,ytest,intervals,combined_intervals,max_comp=10,cv_split=5,cv_repeat=2,precision=4,sipls_fig=1):
    #error check
    
    if len(Xcal)!=len(ycal) or len(Xtest)!=len(ytest):
        raise ValueError("Inconsistent matrix dimensions")
    
    print("-----------总共%d模型即将被构建----------"%comb(intervals,combined_intervals))
    
    x_train_block, x_test_block = splitspectrum(intervals, Xcal, Xtest)
   
    for i in range(1,intervals+1):
        if max_comp>x_train_block[str(i)].shape[1]:
            raise ValueError("the number of componts must less than the variable in each interval")
    '''
    Let's GO!!!
    '''
    sipls_mse=[]#全部组合的mse
    sipls_comp=[]#对应的最小components
    min_comb=[]#最小的组合区间
    comb_list=list(combinations(range(1,intervals+1),combined_intervals))
    if combined_intervals<2:
        raise ValueError("the number of combined intervals must larger than 2")
        #先组合区间再进行计算
    if combined_intervals==2:
        for i in range(0,len(comb_list)):
            print("---正在努力构建第%d个模型---"%(i+1))
            X_train_block_comb=pd.concat([x_train_block[str(comb_list[i][0])],
                                          x_train_block[str(comb_list[i][1])]],axis=1)
            si_min_mse,si_min_comp=PLS_CV(X_train_block_comb,ycal,max_comp,cv_split,cv_repeat,show_fig=0)
            sipls_mse.append(si_min_mse)
            sipls_comp.append(si_min_comp)
        min_sipls_mse=min(sipls_mse)
        min_sipls_mse_index=sipls_mse.index(min(sipls_mse))
        min_sipls_comp=sipls_comp[min_sipls_mse_index]
        min_comb=comb_list[min_sipls_mse_index]
        #建模并预测结果
        pls=PLSRegression(n_components=min_sipls_comp)
        X_calibration_comb=pd.concat([x_train_block[str(min_comb[0])], 
                                  x_train_block[str(min_comb[1])]],axis=1)
        X_test_comb=pd.concat([x_test_block[str(min_comb[0])], 
                                  x_test_block[str(min_comb[1])]],axis=1)
        pls.fit(X_calibration_comb, ycal)
        ypred=pls.predict(X_test_comb)
        RMSEP=np.sqrt(mean_squared_error(ytest, ypred))
        r2=r2_score(ytest,ypred)
        print("-----------%d模型已经全部构建完成----------"%comb(intervals,combined_intervals))
    if combined_intervals==3:
        for i in range(0,len(comb_list)):
            print("---正在努力构建第%d个模型---"%(i+1))
            X_train_block_comb=pd.concat([x_train_block[str(comb_list[i][0])],
                                          x_train_block[str(comb_list[i][1])],
                                          x_train_block[str(comb_list[i][2])]],axis=1)
            si_min_mse,si_min_comp=PLS_CV(X_train_block_comb,ycal,max_comp,cv_split,cv_repeat,show_fig=0)
            sipls_mse.append(si_min_mse)
            sipls_comp.append(si_min_comp)
        min_sipls_mse=min(sipls_mse)
        min_sipls_mse_index=sipls_mse.index(min(sipls_mse))
        min_sipls_comp=sipls_comp[min_sipls_mse_index]
        min_comb=comb_list[min_sipls_mse_index]
        
        pls=PLSRegression(n_components=min_sipls_comp)
        X_calibration_comb=pd.concat([x_train_block[str(min_comb[0])], 
                                  x_train_block[str(min_comb[1])],  
                                  x_train_block[str(min_comb[2])]],axis=1)
        X_test_comb=pd.concat([x_test_block[str(min_comb[0])], 
                                  x_test_block[str(min_comb[1])], 
                                  x_test_block[str(min_comb[2])]],axis=1)
        pls.fit(X_calibration_comb, ycal)
        ypred=pls.predict(X_test_comb)
        RMSEP=np.sqrt(mean_squared_error(ytest, ypred))
        r2=r2_score(ytest,ypred)
        print("-----------%d模型已经全部构建完成----------"%comb(intervals,combined_intervals))
    if combined_intervals==4:
        for i in range(0,len(comb_list)):
            print("---正在努力构建第%d个模型---"%(i+1))
            X_train_block_comb=pd.concat([x_train_block[str(comb_list[i][0])],
                                          x_train_block[str(comb_list[i][1])],
                                          x_train_block[str(comb_list[i][2])],
                                          x_train_block[str(comb_list[i][3])]],axis=1)
            si_min_mse,si_min_comp=PLS_CV(X_train_block_comb,ycal,max_comp,cv_split,cv_repeat,show_fig=0)
            sipls_mse.append(si_min_mse)
            sipls_comp.append(si_min_comp)
        min_sipls_mse=min(sipls_mse)
        min_sipls_mse_index=sipls_mse.index(min(sipls_mse))
        min_sipls_comp=sipls_comp[min_sipls_mse_index]
        min_comb=comb_list[min_sipls_mse_index]
        #确定主成分个数后建模并测试
        pls=PLSRegression(n_components=min_sipls_comp)
        X_calibration_comb=pd.concat([x_train_block[str(min_comb[0])], 
                                  x_train_block[str(min_comb[1])], 
                                  x_train_block[str(min_comb[2])],
                                  x_train_block[str(min_comb[3])]],axis=1)
        X_test_comb=pd.concat([x_test_block[str(min_comb[0])], 
                                  x_test_block[str(min_comb[1])], 
                                  x_test_block[str(min_comb[2])],
                                  x_test_block[str(min_comb[3])]],axis=1)
        pls.fit(X_calibration_comb, ycal)
        ypred=pls.predict(X_test_comb)
        RMSEP=np.sqrt(mean_squared_error(ytest, ypred))
        r2=r2_score(ytest,ypred)
        print("-----------%d模型已经全部构建完成----------"%comb(intervals,combined_intervals))
        if sipls_fig==1:
            plt.scatter(ytest,ypred)
    return min_sipls_mse,min_sipls_mse_index,min_comb,r2,RMSEP,ypred
            
    
'''  
os.chdir("D:\sipls_python")
Xcal=pd.read_csv('Xcal',header=None)
Xtest=pd.read_csv('Xtest',header=None)
ycal=pd.read_csv('ycal',header=None) 
ytest=pd.read_csv('ytest',header=None) 
min_sipls_mse3,min_sipls_comp3,min_comb3,r23,RMSEP3,ypred3=sipls_py(Xcal,ycal,Xtest,ytest,20,3
                                ,max_comp=10,cv_split=5,cv_repeat=1,precision=4)

min_sipls_mse4,min_sipls_comp4,min_comb4,r24,RMSEP4,ypred4=sipls_py(Xcal,ycal,Xtest,ytest,20,4
                                ,max_comp=10,cv_split=5,cv_repeat=1,precision=4)
'''